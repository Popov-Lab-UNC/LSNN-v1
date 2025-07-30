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
import alchemlyb
from alchemlyb.estimators import MBAR
import pandas as pd
from tqdm import tqdm
import mdtraj as md
from openmmtools.constants import kB

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

  def __init__(self, model_dict, method, lambda_electrostatics, lambda_sterics, smiles, name_path, file_path, device, n_steps, report_interval):
    super.__init__(lambda_electrostatics, lambda_sterics, name_path, file_path, n_steps, report_interval)

    self.method = method

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
    

  def calculate_energy_for_traj(self, traj, e_lambda_ster, e_lambda_elec):
    u = np.zeros(len(traj.time))
    e_lambda_ster = torch.scalar_tensor(e_lambda_ster).to(self.device)
    e_lambda_elec = torch.scalar_tensor(e_lambda_elec).to(self.device)

    for idx, coords in enumerate(traj.xyz):

        positions = torch.from_numpy(coords).to(self.device)
        batch = torch.zeros(size=(len(positions), )).to(torch.long) 

        factor = self.model(positions, e_lambda_ster, e_lambda_elec,
                            torch.tensor(1.0).to(self.device), True, batch, None)

        self.curr_simulation_vac.context.setPositions(coords)
        U = self.curr_simulation_vac.context.getState(
            getEnergy=True).getPotentialEnergy()
        val = (U +
                (factor[0].item() * kilojoule_per_mole)) / (kB * self._T)
        u[idx] = float(val)
    return u
  
  def u_nk_processing_df(self, df): 
    df.attrs = {
        "temperature": self._T,
        "energy_unit": "kT",
    }

    df = alchemlyb.preprocessing.decorrelate_u_nk(df, remove_burnin=True)
    return df
  
  def solv_u_nk(self):
        
    self.model.gnn_params = torch.tensor(self.compute_atom_features())

    cache_path = os.path.join(self.name_path, f"{self.name}_u_nk.pkl")
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)

    solv_u_nk_df = []

    #Recreate System for Vaccum -- LSNN Forces removed, you could just deepcopy it beforehand and save some time. 

    self.system, self.molecule, self.topology = self.create_system_from_smiles(self.smiles, self.file_path)

    integrator = LangevinMiddleIntegrator(self._T, 1 / picosecond,
                                    0.001 * picoseconds)
    
    self.curr_simulation_vac = app.Simulation(self.topology,
                                          self.system,
                                          integrator,
                                          platform=self.platform)


    for (lambda_ster,
          lambda_elec) in tqdm(self.get_solv_lambda_schedule()):
        dcd_file = os.path.join(
            self.name_path,
            f"({lambda_ster}-{lambda_elec})_{self.name}.dcd")
        pdb_file = os.path.join(self.name_path, f"{self.name}.pdb")
        traj = md.load(dcd_file, top=pdb_file)

        df = pd.DataFrame({
            "time": traj.time,
            "vdw-lambda": [lambda_ster] * len(traj.time),
            "coul-lambda": [lambda_elec] * len(traj.time),
        })
        df = df.set_index(["time", "vdw-lambda", "coul-lambda"])

        for (e_lambda_ster,
              e_lambda_elec) in self.get_solv_lambda_schedule():
            u = self.calculate_energy_for_traj(traj, e_lambda_ster,
                                                e_lambda_elec)
            df[(e_lambda_ster, e_lambda_elec)] = u

        df = self.u_nk_processing_df(df)

        solv_u_nk_df.append(df)

    solv_u_nk_df = alchemlyb.concat(solv_u_nk_df)

    #not sure what this does, is this just to add an extra index into the thing??
    new_index = []
    for i, index in enumerate(solv_u_nk_df.index):
        new_index.append((i, *index[1:]))
    solv_u_nk_df.index = pd.MultiIndex.from_tuples(
        new_index, names=solv_u_nk_df.index.names)

    solv_u_nk_df.to_pickle(cache_path)

    self.solv = solv_u_nk_df
  

  def meanCalculation(self, traj, lambda_ster, lambda_elec):
    lambda_ster = torch.scalar_tensor(lambda_ster).to(self.device)
    lambda_elec = torch.scalar_tensor(lambda_elec).to(self.device)
    sterics = []
    electrostatics = []

    traj.center_coordinates()
    ref = traj[0]
    aligned = traj.superpose(reference=ref)
    avg_coords = np.mean(aligned.xyz, axis=0)
    avg_structure = md.Trajectory(xyz=avg_coords.reshape(1, -1, 3),
                                  topology=traj.topology)

    rmsd_vals = md.rmsd(target=traj,
                        reference=avg_structure,
                        frame=0,
                        atom_indices=None,
                        precentered=True)

    rmsd_mean = np.mean(rmsd_vals)
    rmsd_std = np.std(rmsd_vals)

    start_index = next((index for index, rmsd in enumerate(rmsd_vals)
                        if rmsd < (rmsd_mean + rmsd_std)),
                        self.minimum_equil_frame)

    start_index = self.minimum_equil_frame if start_index < self.minimum_equil_frame else start_index

    print(start_index)

    gnn_params = torch.cat((
        torch.tensor(self.compute_atom_features()),
        torch.full((len(avg_structure.xyz[0]), 1), lambda_elec),
        torch.full((len(avg_structure.xyz[0]), 1), lambda_ster),
    ),
                            dim=-1)
    batch = torch.zeros(size=(len(gnn_params), )).to(torch.long)                   
    '''
    gnn_params = gnn_params.repeat(len(traj[start_index:]))

    batch = torch.arange(0, len(traj[start_index:]))
    batch = batch.repeat_interleave(len(traj[0].xyz))

    positions = torch.from_numpy(traj[start_index:].xyz).float().reshape(-1, 3)

    lambda_elecs = torch.full((len(traj[start_index:]), 1), lambda_elec)
    lambda_sters = torch.full((len(traj[start_index:]), 1), lambda_ster)

    _, _, sterics, electrostatics = self.model(positions, lambda_sters, lambda_elecs, torch.tensor(0.0), True, batch, gnn_params)

    return (lambda_elec.item(),
            (torch.mean(torch.tensor(electrostatics).detach()).item()),
            lambda_ster.item(),
            (torch.mean(torch.tensor(sterics).detach()).item()))
    
    '''
    for idx, coords in enumerate(traj[start_index:].xyz):
        positions = torch.from_numpy(coords).to(self.device)
        positions = positions.float()
        lambda_ster = lambda_ster.float()
        lambda_elec = lambda_elec.float()
        U, F, steric, electrostatic = self.model(
            positions, lambda_ster, lambda_elec,
            torch.tensor(0.0).to(self.device), True, batch, gnn_params)
        
        sterics.append(steric)
        electrostatics.append(electrostatic)
    return (lambda_elec.item(),
            (torch.mean(torch.tensor(electrostatics).detach()).item()),
            lambda_ster.item(),
            (torch.mean(torch.tensor(sterics).detach()).item()))
  
  def collateInfo(self):
    self.derivatives = []
    self.set_model()
    for (lambda_ster,
          lambda_elec) in tqdm(self.get_solv_lambda_schedule()):
        dcd_file = os.path.join(
            self.solv_path,
            f"({lambda_ster}-{lambda_elec})_{self.name}.dcd")
        pdb_file = os.path.join(self.solv_path, f"{self.name}.pdb")
        traj = md.load(dcd_file, top=pdb_file)
        self.derivatives.append(
            self.meanCalculation(traj, lambda_ster, lambda_elec))

  def TI(self):
    assert self.derivatives

    derivatives = np.array(self.derivatives)
    elec = derivatives[:, 0]
    elec_der = derivatives[:, 1]
    ster = derivatives[:, 2]
    ster_der = derivatives[:, 3]

    np_elec = np.array(elec)
    np_elec_der = np.array(elec_der)
    np_ster = np.array(ster)
    np_ster_der = np.array(ster_der)

    u_elec = np.unique(np_elec)
    u_ster = np.unique(np_ster)

    u_elec_der = np.array(
        [np.median(np_elec_der[np_elec == ux]) for ux in u_elec])
    u_ster_der = np.array(
        [np.median(np_ster_der[np_ster == ux]) for ux in u_ster])

    sort_elec = np.argsort(u_elec)
    sort_ster = np.argsort(u_ster)

    int_elec = np.trapz(u_elec_der[sort_elec], u_elec[sort_elec])
    int_ster = np.trapz(u_ster_der[sort_ster], u_ster[sort_ster])

    return ((int_elec + int_ster) / 4.1839)
  
  def MBAR(self):
    assert self.solv

    mbar_solv = MBAR()
    mbar_solv.fit(self.solv)

    F_solv_kt = mbar_solv.delta_f_[(0.0, 0.0)][(1.0, 1.0)]
    F_solv = F_solv_kt * self._T * kB
    return -F_solv.value_in_unit(kilojoule_per_mole) * 0.2390


  def computeEnergies(self):
    if(self.method == "LSNN-MBAR"):
       self.solv_u_nk()
    else:
       self.collateInfo()
  

  def computeG(self):
    if(self.method == "LSNN-MBAR"):
       return self.MBAR()
    else:
       return self.TI()



