from utils.fep import apply_fep, set_fep_lambdas
import copy
import openmm as mm
from openmm import app, Platform
from openmm.unit import *
from abc_sim_class import simulation_calc
from utils.lr_complex import LRComplex, get_lr_complex
from utils.misc import smi_to_protonated_sdf
import os
import pandas as pd
import mdtraj as md
from alchemlyb.estimators import MBAR
import alchemlyb
from alchemlyb.preprocessing.subsampling import decorrelate_u_nk
from openmmtools.constants import kB
import numpy as np
import tqdm as tqdm

def get_lig_and_water_indices(system):
  """ Returns lists of sets of the ligand and water atom indices.
  This is the format needed by fep.py """

  lig_indices = []
  water_indices = []
  for residue in system.topology.residues():
      in_lig = False
      atom_indices = set()
      for atom in residue.atoms():
          if atom.index in system.lig_indices:
              in_lig = True
          atom_indices.add(atom.index)
      if in_lig:
          lig_indices.append(atom_indices)
      else:
          water_indices.append(atom_indices)

  return lig_indices, water_indices

def make_alchemical_system(system, T):
  """ Created a new alchemically modified LR system"""
  lig_indices, water_indices = get_lig_and_water_indices(system)

  # create a new alchemical system
  alchemical_system = apply_fep(system.system, lig_indices, water_indices)

  integrator = mm.LangevinIntegrator(T,
                                      1.0/unit.picosecond,
                                      4.0*unit.femtosecond)
  simulation = app.Simulation(system.topology, alchemical_system, integrator, system.platform)

  lr_system = copy.copy(system)
  lr_system.system = alchemical_system
  lr_system.integrator = integrator
  lr_system.simulation = simulation

  return lr_system

class TIP3P(simulation_calc):
  def __init__(self, lambda_electrostatics, lambda_sterics, smiles, name_path, file_path, device, n_steps, report_interval):
    super.__init__(lambda_electrostatics, lambda_sterics, name_path, file_path, n_steps, report_interval)
    match device:
      case "auto":
        self.platform = None
      case "CPU":
        self.platform = Platform.getPlatformByName("CPU")
      case "CUDA":
        self.platform = Platform.getPlatformByName("CUDA")
      case _:
        raise ValueError("Device Not Supported or Invalid")
    
    self.smiles = smiles
  
  
  def create_system(self):
    if(not os.path.exist(self.file_path)):
       smi_to_protonated_sdf(self.smiles, self.file_path)
    kwargs = {
          "nonbonded_cutoff": 0.9 * unit.nanometer,
          # "nonbonded_cutoff": 1.5*unit.nanometer,
          "constraints": app.HBonds,
          "box_padding": 1.6 * unit.nanometer,
          # "box_padding": 2.0*unit.nanometer,
          "lig_ff": "gaff",
          "cache_dir": self.name_path,
      }
    system = get_lr_complex(None,
                                self.file_path,
                                solvent="tip3p",
                                nonbonded_method=app.NoCutoff,
                                **kwargs)
    system_vac = get_lr_complex(None, self.file_path, solvent="none", **kwargs)
    system.save(os.path.join(self.name_path, "system"))
    system_vac.save(os.path.join(self.name_path, "system_vac"))

    system.save_to_pdb(os.path.join(self.name_path, "system.pdb"))
    system_vac.save_to_pdb(os.path.join(self.name_path, "system_vac.pdb"))

    self.system = make_alchemical_system(system, self._T)

    self.system_vac = make_alchemical_system(system_vac, self._T)

    self.system.set_positions(system.get_positions())

  def get_sim_prefix(self,
                       lambda_sterics,
                       lambda_electrostatics,
                       vacuum=False):
    """ Returns the prefix for the simulation file """
    prefix = f"sim_{lambda_sterics}_{lambda_electrostatics}"
    if vacuum:
      prefix += "_vac"
    return prefix
  
  def minimize(self):
    cache_file = os.path.join(self.name_path, "minimized.pkl")
    tol = 1.0
    try:
        self.system.minimize_cached(cache_file, tol)
    except mm.OpenMMException:
        # in case we create a system with a different number of waters
        os.remove(cache_file)
        self.system.minimize_cached(cache_file, tol)


  def simulate(self,
                 lambda_sterics,
                 lambda_electrostatics,
                 prefix=None,
                 vacuum=False):
    """ Run the simulation for n_steps, saving the trajectory
    to name_path/prefix.dcd """

    if prefix is None:
      prefix = self.get_sim_prefix(lambda_sterics, lambda_electrostatics, vacuum)

    out_dcd = os.path.join(self.name_path, f"{prefix}.dcd")
    out_com = os.path.join(self.name_path, f"{prefix}.completed")

    if os.path.exists(out_com):
      print(f"Skipping {prefix} -- already completed")
      return
    
    if vacuum:
        simulation = self.system_vac.simulation
    else:
        simulation = self.system.simulation

    set_fep_lambdas(simulation.context, lambda_sterics,
                        lambda_electrostatics)
    simulation.reporters.clear()
    simulation.reporters.append(app.DCDReporter(out_dcd, 100))
    simulation.step(self.n_steps)

    with open(out_com, "w") as f:
        f.write("completed")


  def run_all_sims(self):
    self.minimize()

    print("Removing electrostatics")
    lambda_ster = 1.0
    for lambda_elec in reversed(self.lambda_electrostatics):
        print(f"Running {lambda_ster=}, {lambda_elec=}")
        self.simulate(lambda_ster, lambda_elec)

    print("Removing sterics")
    lambda_elec = 0.0
    for lambda_ster in reversed(self.lambda_sterics):
        print(f"Running {lambda_ster=}, {lambda_elec=}")
        self.simulate(lambda_ster, lambda_elec)

    print("Re-adding electrostatics in vacuum")
    lambda_ster = 0.0
    lig_pos = self.system.get_lig_positions()
    self.system_vac.set_positions(lig_pos)
    for lambda_elec in self.lambda_electrostatics:
        print(f"Running {lambda_ster=}, {lambda_elec=}")
        self.simulate(
                      lambda_ster,
                      lambda_elec,
                      vacuum=True)
        

  def get_vac_u_nk(self, lambda_elec):
    """ Returns the u_nk dataframe for the vacuum simulation 
    run at lambda_elec. This returns the energy of the system
    from _all_ the lambda_elec values for the simulation """

    prefix = self.get_sim_prefix(0.0, lambda_elec, vacuum=True)
    cache_fname = os.path.join(self.name_path, f"{prefix}_u_nk.pkl")
    if os.path.exists(cache_fname):
        return pd.read_pickle(cache_fname)

    dcd_file = prefix + ".dcd"
    dcd_file = os.path.join(self.name_path, dcd_file)
    pdb_file = os.path.join(self.name_path, "system_vac.pdb")

    traj = md.load(dcd_file, top=pdb_file)
    df = pd.DataFrame({
        "time": traj.time,
        "fep-lambda": [lambda_elec]*len(traj.time),
    })
    df = df.set_index(["time", "fep-lambda"])

    for energy_lambda_elec in self.lambda_electrostatics:
        u = np.zeros(len(traj.time))
        set_fep_lambdas(self.system_vac.simulation.context, 0.0, energy_lambda_elec)
        for i, coords in enumerate(traj.xyz):
            self.system_vac.set_positions(coords*unit.nanometer)
            U = self.system_vac.simulation.context.getState(getEnergy=True).getPotentialEnergy()
            # reduced energy (divided by kT)
            u[i] = U / (kB * self._T)
        df[energy_lambda_elec] = u

    df.attrs = {
        "temperature": self._T.value_in_unit(unit.kelvin),
        "energy_unit": "kT",
    }

    df = decorrelate_u_nk(df, remove_burnin=True)

    df.to_pickle(cache_fname)

    return df
        
  def get_all_vac_u_nk(self):
      df = alchemlyb.concat([self.get_vac_u_nk(lambda_elec) for lambda_elec in self.lambda_electrostatics])
      # make sure time is increasing
      new_index = []
      for i, index in enumerate(df.index):
          new_index.append((i, *index[1:]))
      df.index = pd.MultiIndex.from_tuples(new_index, names=df.index.names)
      return df
  
  def get_solv_u_nk(self, lambda_ster, lambda_elec):
    """ Returns the u_nk dataframe for the solvation simulation run
    at lambda_ster and lambda_elec. """
        
    prefix = self.get_sim_prefix(lambda_ster, lambda_elec) 
    cache_fname = os.path.join(self.name_path, f"{prefix}_u_nk.pkl")
    if os.path.exists(cache_fname):
        return pd.read_pickle(cache_fname)

    dcd_file = prefix + ".dcd"
    dcd_file = os.path.join(self.name_path, dcd_file)
    pdb_file = os.path.join(self.name_path, "system.pdb")

    traj = md.load(dcd_file, top=pdb_file)
    df = pd.DataFrame({
        "time": traj.time,
        "vdw-lambda": [lambda_ster]*len(traj.time),
        "coul-lambda": [lambda_elec]*len(traj.time),
    })
    df = df.set_index(["time", "vdw-lambda", "coul-lambda"])

    for energy_lambda_ster, energy_lambda_elec in self.get_solv_lambda_schedule():
        u = np.zeros(len(traj.time))
        set_fep_lambdas(self.system.simulation.context, energy_lambda_ster, energy_lambda_elec)
        for i, coords in enumerate(traj.xyz):
            self.system.set_positions(coords*unit.nanometer)
            U = self.system.simulation.context.getState(getEnergy=True).getPotentialEnergy()
            # reduced energy (divided by kT)
            u[i] = U / (kB * self._T)
        df[(energy_lambda_ster, energy_lambda_elec)] = u

    df.attrs = {
        "temperature": self._T.value_in_unit(unit.kelvin),
        "energy_unit": "kT",
    }

    df = decorrelate_u_nk(df, remove_burnin=True)

    df.to_pickle(cache_fname)

    return df

  def get_all_solv_u_nk(self):
    df = alchemlyb.concat([self.get_solv_u_nk(lambda_ster, lambda_elec) for lambda_ster, lambda_elec in tqdm(self.get_solv_lambda_schedule())])
    # make sure time is increasing
    new_index = []
    for i, index in enumerate(df.index):
        new_index.append((i, *index[1:]))
    df.index = pd.MultiIndex.from_tuples(new_index, names=df.index.names)
    return df

  

  def computeEnergies(self):
    self.u_nk_vac = self.get_all_vac_u_nk()
    self.u_nk_solv = self.get_all_solv_u_n
  
 
  def computeG(self):
    assert self.u_nk_solv, self.u_nk_vac
    T = self.u_nk_vac.attrs["temperature"] * unit.kelvin

    mbar_vac = MBAR()
    mbar_vac.fit(self.u_nk_vac)

    mbar_solv = MBAR()
    mbar_solv.fit(self.u_nk_solv)

    F_solv_kt = mbar_vac.delta_f_[0][1] - mbar_solv.delta_f_[(0,0)][(1,1)]
    F_solv = F_solv_kt*T*kB

    return F_solv

    
