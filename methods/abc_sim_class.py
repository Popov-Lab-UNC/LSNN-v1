from abc import ABC, abstractmethod
from openmm.unit import *


class simulation_calc(ABC):

  def __init__(self, lambda_electrostatics, lambda_sterics, name_path, file_path, n_steps, report_interval):
    self.lambda_sterics = lambda_sterics
    self.lambda_electrostatics = lambda_electrostatics
    self.name_path = name_path
    self.file_path = file_path
    self.n_steps = n_steps
    self.report_interval = report_interval
    self._T = 300 * kelvin
    return

  @classmethod
  def get_solv_lambda_schedule(self):
    """ Returns a list of tuples of (lambda_ster, lambda_elec) 
    for the solvation simulations """

    lambda_schedule = []
    lambda_ster = 1.0
    for lambda_elec in reversed(self.lambda_electrostatics):
        lambda_schedule.append((lambda_ster, lambda_elec))

    lambda_elec = 0.0
    for lambda_ster in reversed(self.lambda_sterics):
        lambda_schedule.append((lambda_ster, lambda_elec))

    return lambda_schedule
  
  @abstractmethod
  def create_system(self):
    pass
  @abstractmethod
  def run_all_sims(self):
    pass
  @abstractmethod
  def computeEnergies(self):
    pass
  
  @abstractmethod
  def computeG(self):
    pass


