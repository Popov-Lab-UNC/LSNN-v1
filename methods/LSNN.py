from abc_sim_class import simulation_calc
from openff.toolkit.topology import Molecule
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from openmm import app, Platform, NonbondedForce, LangevinMiddleIntegrator
import torch
from MachineLearning.GNN_Models import GNN3_scale_96
import numpy as np
from openmm.unit import *
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import pickle as pkl
from openmm.app.internal.customgbforces import GBSAGBn2Force
from openmmtorch import TorchForce
import time

class LSNN(simulation_calc):

  @staticmethod
  def create_system_from_smiles(smiles, pdb_path):
    molecule = Molecule.from_smiles(smiles)
    pdb = app.PDBFile(pdb_path)
    smirnoff = SMIRNOFFTemplateGenerator(molecules=molecule)
    forcefield = app.ForceField()
    forcefield.registerTemplateGenerator(smirnoff.generator)
    system_F = forcefield.createSystem(pdb.topology)
    return system_F, molecule, pdb.topology

  def __init__(self, model_dict, lambda_electrostatics, lambda_sterics, smiles, name_path, file_path, device, n_steps, report_interval):
    super.__init__(lambda_electrostatics, lambda_sterics, name_path, file_path, n_steps, report_interval)

    match device:
      case "auto":
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if (self.device == torch.device("cpu")):
            self.platform = Platform.getPlatformByName('CPU')
        else:
            self.platform = Platform.getPlatformByName('CUDA')
      case "CPU":
        self.device = torch.device("cpu")
        self.platform = Platform.getPlatformByName("CPU")
      case "CUDA":
        self.device = torch.device("cuda")
        self.platform = Platform.getPlatformByName("CUDA")
      case _:
        raise ValueError("Device Not Supported or Invalid")

    self.model_dict = torch.load(model_dict, map_location=self.device)
    tot_unique = [
            0.14, 0.117, 0.155, 0.15, 0.21, 0.185, 0.18, 0.17, 0.12, 0.13
        ]
    self.model = GNN3_scale_96(max_num_neighbors=10000,
                                   parameters=None,
                                   device=self.device,
                                   fraction=0.5,
                                   unique_radii=tot_unique,
                                   jittable=True).to(self.device)
    self.model.load_state_dict(self.model_dict)
    self.model.to(self.device)
    self.model.eval()
    self.smiles = smiles
    self.charges = None

  def savePDB(self):
    m = Chem.MolFromSmiles(self.smiles)
    mh = Chem.AddHs(m)
    AllChem.EmbedMolecule(mh)
    Chem.MolToPDBFile(mh, self.file_path)

  def compute_atom_features(self):
        '''Calculates the atom features needed for the GNN to function. the GB Force has derived parameters requiring 
        the derived function to pass through to calculate the radindex based on the GB radius
        '''
        atom_features_path = os.path.join(self.name_path,
                                          f"{self.name}_gnn_params.pkl")

        if os.path.exists(atom_features_path):
            print("Found Existing Atom Features")
            with open(atom_features_path, 'rb') as f:
                data = pkl.load(f)

            return data.to(self.device)

        print("Calculating Atom Features for GNN")
        force = GBSAGBn2Force(cutoff=None,
                              SA="ACE",
                              soluteDielectric=1,
                              solventDielectric=78.5)
        print(78.5)
        gnn_params = np.array(force.getStandardParameters(self.topology))
        gnn_params = np.concatenate((np.reshape(self.charges,
                                                (-1, 1)), gnn_params),
                                    axis=1)
        force.addParticles(gnn_params)
        force.finalize()
        gbn2_parameters = np.array([
            force.getParticleParameters(i)
            for i in range(force.getNumParticles())
        ])
        print(f"Atom Features shape: {gbn2_parameters.shape}")

        with open(atom_features_path, 'wb') as f:
            pkl.dump(torch.from_numpy(gbn2_parameters), f)

        with open(atom_features_path, 'rb') as f:
            data = pkl.load(f)

        return data.to(self.device)

  def create_system(self):
    start = time.time()
    if not os.path.exists(self.file__path):
      print("-- PDB of Smile not Found, Creating Now -- ")
      self.savePDB()


    self.system, self.molecule, self.topology = self.create_system_from_smiles(self.smiles, self.file_path)

    nonbonded = [
            f for f in self.system.getForces()
            if isinstance(f, NonbondedForce)
        ][0]

    self.charges = np.array([
            tuple(nonbonded.getParticleParameters(idx))[0].value_in_unit(
                elementary_charge)
            for idx in range(self.system.getNumParticles())
        ])

    gnn_params_path = os.path.join(self.name_path,
                                  f"{self.name}_gnn_paramed_model.pt")

    if not os.path.exists(gnn_params_path):
      gnn_params = torch.tensor(self.compute_atom_features())

      self.model.gnn_params = gnn_params
      self.model.batch = torch.zeros(size=(len(gnn_params), )).to(torch.long)
      torch.jit.script(self.model).save(gnn_params_path)

    self.model_force = TorchForce(gnn_params_path)
    self.model_force.addGlobalParameter("lambda_sterics", 1.0)
    self.model_force.addGlobalParameter("lambda_electrostatics", 1.0)
    self.model_force.addGlobalParameter("retrieve_forces", 1.0)
    #self.model_force.addGlobalParameter("atom_features", 1.0)
    #self.model_force.addGlobalParameter('batch', -1.0)
    self.model_force.setOutputsForces(True)
    self.system.addForce(self.model_force)

    self.forces = {
        force.__class__.__name__: force
        for force in self.system.getForces()
    }

    setup_time = time.time()

    print(
        f" -- Finished Setup in {setup_time - start} seconds; Starting Simulation -- "
    )

  def AI_simulation(self, lambda_sterics, lambda_electrostatics, out):

    path = os.path.join(self.name_path, f"{out}.dcd")
    com = os.path.join(self.name_path, f"{out}.com")

    if os.path.exists(com):
        return

    integrator = LangevinMiddleIntegrator(
        self._T, 1 / picosecond,
        0.002 * picoseconds)  #reduced from 2 femtoseconds temporarily
    simulation = app.Simulation(self.topology,
                            self.system,
                            integrator,
                            platform=self.platform)
    simulation.context.setParameter("lambda_sterics", lambda_sterics)
    simulation.context.setParameter("lambda_electrostatics",
                                    lambda_electrostatics)
    simulation.context.setParameter("retrieve_forces", 1.0)

    simulation.context.setPositions(self.PDB.positions)

    simulation.minimizeEnergy()
    simulation.reporters.append(app.DCDReporter(path, self.report_interval))

    simulation.step(self.n_steps)
    with open(com, 'w'):
        pass


  def run_all_sims(self):
    start_time = time.time()

    print(" -- Starting Simulations -- Removing Electrostatics -- ")
    for lambda_elec in reversed(self.lambda_electrostatics):
        self.AI_simulation(1.0,
                            lambda_elec,
                            vaccum=0.0,
                            out=f"(1.0-{lambda_elec})_{self.name}")
    solv_elec_time = time.time()
    print(
            f" -- Time taken: {solv_elec_time - start_time} - Removing Sterics -- "
        )
    
    for lambda_ster in reversed(self.lambda_sterics):
      self.AI_simulation(lambda_ster,
                          0.0,
                          vaccum=0.0,
                          out=f"({lambda_ster}-0.0)_{self.name}")
    print(
            f" -- Finished Simulation -- Sterics Time: {time.time() - solv_elec_time}; Total Time: {time.time() - start_time} -- "
        )




