from abc_sim_class import simulation_calc
from openmm import Platform, app, LangevinMiddleIntegrator, LocalEnergyMinimizer
from openmm.unit import *
import os
import subprocess
from utils.lr_complex import LRComplex, get_lr_complex
from openmmtools import alchemy
from openmmtools.states import ThermodynamicState, CompoundThermodynamicState, SamplerState
from utils.reporter import PositionsReporter
from openmmtools.constants import kB
import numpy as np
import alchemlyb
from alchemlyb.estimators import MBAR
import h5py
import pandas as pd

class ImplicitSolv(simulation_calc):
  @staticmethod
  def make_inital_system(out_folder, lig_file, smile, solvent):
    kwargs = {
      "nonbonded_cutoff": 0.9 * unit.nanometer,
      # "nonbonded_cutoff": 1.5*unit.nanometer,
      "constraints": app.HBonds,
      "box_padding": 1.6 * unit.nanometer,
      # "box_padding": 2.0*unit.nanometer,
      "lig_ff": "gaff",
      "cache_dir": out_folder,
    }
    lig_file = os.path.join(out_folder, "ligand.sdf")
    if not os.path.exists(lig_file):
      cmd = f'obabel "-:{smile}" -O "{lig_file}" --gen3d --pH 7'
      subprocess.run(cmd, check=True, shell=True, timeout=60)
    system = get_lr_complex(
      None,
      lig_file,
      solvent=solvent,
      include_barostat=True if solvent == "tip3p" else False,
      **kwargs)
    system_vac = get_lr_complex(None, lig_file, solvent="none", **kwargs)

    system.save(os.path.join(out_folder, "system"))
    system_vac.save(os.path.join(out_folder, "system_vac"))

    system.save_to_pdb(os.path.join(out_folder, "system.pdb"))
    system_vac.save_to_pdb(os.path.join(out_folder, "system_vac.pdb"))
    return system, system_vac
  
  def __init__(self, solvent, lambda_electrostatics, lambda_sterics, smiles, name_path, file_path, device, n_steps, report_interval):
    super.__init__(lambda_electrostatics, lambda_sterics, name_path, file_path, n_steps, report_interval)
    self.solvent = solvent
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
    system_path = os.path.join(self.name_path, "system")
    lig_path = os.path.join(self.name_path, "ligand.sdf")

    if (not os.path.exists(self.name_path)):
      os.mkdir(self.name_path)
    if (not os.path.exists(system_path)):
        system, system_vac = self.create_system(self.name_path, lig_path,
                                                self.smile, self.solvent)  
    else:
      try:
          system_vac_path = os.path.join(self.name_path, "system_vac")
          system = LRComplex.load(system_path)
          system_vac = LRComplex.load(system_vac_path)
      except Exception as e:
          print(f"Existing files corrupted... Continuing...: {str(e)}")
          return
      
      self.PDB = app.PDBFile(os.path.join(self.name_path, "system.pdb"))
      self.mol_indices = system_vac.lig_indices


    factory = alchemy.AbsoluteAlchemicalFactory(
            consistent_exceptions=False,
            alchemical_pme_treatment='direct-space',
            disable_alchemical_dispersion_correction=True,
            split_alchemical_forces=True)

    system_region = alchemy.AlchemicalRegion(
        alchemical_atoms=system.lig_indices,
        annihilate_electrostatics=True,
        annihilate_sterics=False)
    system_vac_region = alchemy.AlchemicalRegion(
        alchemical_atoms=system_vac.lig_indices,
        annihilate_electrostatics=True,
        annihilate_sterics=False)

    print("Alchemical Atoms:", system_region.alchemical_atoms)

    self.alchemy_system = factory.create_alchemical_system(
        system.system, system_region)
    self.alchemy_system_vac = factory.create_alchemical_system(
        system_vac.system, system_vac_region)

    therm_state = ThermodynamicState(self.alchemy_system, self._T)
    therm_vac_state = ThermodynamicState(self.alchemy_system_vac, self._T)

    alchemical_state = alchemy.AlchemicalState.from_system(self.alchemy_system)

    alchemical_state_vac = alchemy.AlchemicalState.from_system(
        self.alchemy_system_vac)

    self.compound_state = CompoundThermodynamicState(
        thermodynamic_state=therm_state,
        composable_states=[alchemical_state])
    self.compound_state_vac = CompoundThermodynamicState(
        thermodynamic_state=therm_vac_state,
        composable_states=[alchemical_state_vac])
    
  def run_sim(self, lambda_elec, lambda_ster, vacuum=False):
    if vacuum:
      compound_state = self.compound_state_vac
      file_name = os.path.join(
          self.name_path, f"{self.name}_vac_{lambda_elec}_{lambda_ster}")
    else:
      compound_state = self.compound_state
      file_name = os.path.join(
          self.name_path, f"{self.name}_{lambda_elec}_{lambda_ster}")

    compound_state.lambda_electrostatics = lambda_elec
    compound_state.lambda_sterics = lambda_ster

    com_file_name = file_name + ".com"
    file_name += ".h5"

    if (os.path.exists(com_file_name)):
      print("File Already Exists, Continuing...")
      return

    integrator = LangevinMiddleIntegrator(self._T, 1.0 / unit.picoseconds,
                                          0.002 * unit.picoseconds)

    self.curr_context = compound_state.create_context(
      integrator, self.platform)
    print(self.PDB.positions)
    self.curr_context.setPositions(self.PDB.positions)
    LocalEnergyMinimizer.minimize(self.curr_context)
    reporter_solv = PositionsReporter(file_name, self.mol_indices,
                                      self.report_interval)

    for _ in range(0, self.n_steps, self.report_interval):
      integrator.step(self.report_interval)
      state = self.curr_context.getState(getPositions=True,
                                          getForces=True,
                                          getEnergy=True,
                                          getParameters=True,
                                          getParameterDerivatives=True)
        #reporter_dcd.report(context, state)
      reporter_solv.report(self.curr_context, state)

    with open(com_file_name,  'w'):
        pass

  def run_all_sims(self):
    print("Removing electrostatics")
    lambda_ster = 1.0
    for lambda_elec in reversed(self.lambda_electrostatics):
        print(f"Running {lambda_ster=}, {lambda_elec=}")
        self.run_sim(lambda_elec, lambda_ster)

    print("Removing sterics")
    lambda_elec = 0.0
    for lambda_ster in reversed(self.lambda_sterics):
        print(f"Running {lambda_ster=}, {lambda_elec=}")
        self.run_sim(lambda_elec, lambda_ster)

    print("Re-adding electrostatics in vacuum")
    lambda_ster = 0.0
    for lambda_elec in self.lambda_electrostatics:
        print(f"Running {lambda_ster=}, {lambda_elec=}")
        self.run_sim(lambda_elec, lambda_ster, vacuum=True)


  def calculate_energy_for_traj(self, pos, energy_context, e_lambda_ster, e_lambda_elec, vacuum):
    u = np.zeros(len(pos))
    if vacuum:
        compound_state = self.compound_state_vac
    else:
        compound_state = self.compound_state

    compound_state.lambda_electrostatics = e_lambda_elec
    compound_state.lambda_sterics = e_lambda_ster
    compound_state.apply_to_context(energy_context)

    for idx, coords in enumerate(pos):
        energy_context.setPositions(coords)
        #LocalEnergyMinimizer.minimize(energy_context)
        U = energy_context.getState(getEnergy=True).getPotentialEnergy()

        u[idx] = U / (kB * self._T)
    return u
  
  def u_nk_processing_df(self, df):
    df.attrs = {
        "temperature": self._T,
        "energy_unit": "kT",
    }

    df = alchemlyb.preprocessing.decorrelate_u_nk(df, remove_burnin=True)
    return df
  
  def get_vac_u_nk(self):
    cache_path = os.path.join(self.name_path, f"{self.name}_vac_u_nk.pkl")
    '''
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)
    '''
    
    integrator = LangevinMiddleIntegrator(self._T, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds)
    energy_context = self.compound_state_vac.create_context(integrator, self.platform)

    vac_u_nk_df = []

    for lambda_elec in self.electrostatics:
        indiv_path = os.path.join(
            self.name_path,
            f"{self.name}_{lambda_elec}_vac_u_nk.pkl")
        if os.path.exists(indiv_path):
            df = pd.read_pickle(indiv_path)
            df = self.u_nk_processing_df(df)
            vac_u_nk_df.append(df)
        file_name = os.path.join(
        self.name_path, f"{self.name}_vac_{lambda_elec}_0.0")
        file_name += ".h5"

        with h5py.File(file_name, 'r') as f:
            pos = f['positions'][:]
        df = pd.DataFrame({
            "time": [
                self.report_interval * idx * 0.002
                for idx, _ in enumerate(pos)
            ],
            "fep-lambda": [lambda_elec] * len(pos),
        })
        df = df.set_index(["time", "fep-lambda"])

        for e_lambda_elec in self.electrostatics:
            u = self.calculate_energy_for_traj(pos,energy_context, 0.0,
                                                e_lambda_elec, True)
            df[(e_lambda_elec)] = u

        #df.to_pickle(indiv_path)
        df = self.u_nk_processing_df(df)

        vac_u_nk_df.append(df)

    vac_u_nk_df = alchemlyb.concat(vac_u_nk_df)

    new_index = []
    for i, index in enumerate(vac_u_nk_df.index):
        new_index.append((i, *index[1:]))
    vac_u_nk_df.index = pd.MultiIndex.from_tuples(
        new_index, names=vac_u_nk_df.index.names)

    vac_u_nk_df.to_pickle(cache_path)

    return vac_u_nk_df
  

  
  def get_solv_u_nk(self):

    cache_path = os.path.join(self.name_path, f"{self.name}_u_nk.pkl")

    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)
    
    integrator = LangevinMiddleIntegrator(self._T, 1.0 / unit.picoseconds, 0.002 * unit.picoseconds)
    energy_context = self.compound_state.create_context(integrator, self.platform)
        

    solv_u_nk_df = []

    for (lambda_ster, lambda_elec) in self.get_solv_lambda_schedule():
        indiv_path = os.path.join(
            self.name_path,
            f"{self.name}_{lambda_elec}_{lambda_ster}_u_nk.pkl")
        if os.path.exists(indiv_path):
            df = pd.read_pickle(indiv_path)
            df = self.u_nk_processing_df(df)
            solv_u_nk_df.append(df)
            continue
        file_name = os.path.join(
            self.name_path, f"{self.name}_{lambda_elec}_{lambda_ster}")
        file_name += ".h5"
        with h5py.File(file_name, 'r') as f:
            pos = f['positions'][:]
        df = pd.DataFrame({
            "time": [
                self.report_interval * idx * 0.002
                for idx, _ in enumerate(pos)
            ],
            "vdw-lambda": [lambda_ster] * len(pos),
            "coul-lambda": [lambda_elec] * len(pos),
        })
        df = df.set_index(["time", "vdw-lambda", "coul-lambda"])

        for (e_lambda_ster,
              e_lambda_elec) in self.get_solv_lambda_schedule():
            u = self.calculate_energy_for_traj(pos,energy_context, e_lambda_ster,
                                                e_lambda_elec, False)
            df[(e_lambda_ster, e_lambda_elec)] = u

        #df.to_pickle(indiv_path)
        df = self.u_nk_processing_df(df)
        solv_u_nk_df.append(df)

    solv_u_nk_df = alchemlyb.concat(solv_u_nk_df)
    new_index = []
    for i, index in enumerate(solv_u_nk_df.index):
        new_index.append((i, *index[1:]))
    solv_u_nk_df.index = pd.MultiIndex.from_tuples(
        new_index, names=solv_u_nk_df.index.names)

    solv_u_nk_df.to_pickle(cache_path)

    return solv_u_nk_df
  
  def computeEnergies(self):
    self.u_nk_vac = self.get_vac_u_nk()
    self.u_nk_solv = self.get_solv_u_nk()
  
  def computeG(self):
    assert self.u_nk_vac, self.u_nk_solv
    mbar_vac = MBAR()
    mbar_vac.fit(self.u_nk_vac)

    mbar_solv = MBAR()
    mbar_solv.fit(self.u_nk_solv)

    F_solv_kt = -mbar_solv.delta_f_[
        (0, 0)][(1, 1)] + mbar_vac.delta_f_[0][1]
    F_solv = F_solv_kt * kB * self._T

    return F_solv.value_in_unit(
            unit.kilojoule_per_mole) * 0.239006
    

    
