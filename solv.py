import yaml
import sys
import utils.lr_complex
import os


class solvation_calculation():

  def __init__(self):
    with open('path_to_your_file.yaml', 'r') as f:
      config = yaml.safe_load(f)

    self.name = config.molecule.name
    cache_path = config.molecule.cache_path
    self.name_path = os.path.join(cache_path, self.name)

    self.smiles = config.molecule.smiles
    self.file_path = config.molecule.file_path

    if (not os.path.exist()):
      os.mkdir(self.name_path)


    self.n_steps = config.calculation.steps
    
    self.report_interval = config.calculation.report_interval
    self.lambda_electrostatics = config.calculation.lambda_electrostatics
    self.lambda_sterics = config.calculation.lambda_sterics


    sim_args = (self.lambda_electrostatics, 
                self.lambda_sterics, 
                self.smiles, 
                self.name_path, 
                self.file_path, 
                config.calculation.device, 
                self.n_steps,
                self.report_interval)

    match config.calculation.solvent:
      case "LSNN":
          from methods.LSNN import LSNN
          simulation_calc = LSNN(config.calculation.model_dict_path, *sim_args)
      case "OBC2" | "GBn2":
          from methods.OBC2GBN2 import ImplicitSolv
          simulation_calc = ImplicitSolv(config.calculation.solvent, *sim_args)
      case "TIP3P":
          from methods.TIP3P import TIP3P
          simulation_calc = TIP3P(*sim_args)
      case _:
          raise ValueError("Invalid Solvent Provided.")
       
    method_args = ()
    
    match config.calculation.method:
      case "MBAR":
        from methods.MBAR import MBARCalc
        method_calc = MBARCalc(config.calculation.solvent, *method_args)
      case "TI":
        from methods.TI import TICalc
        method_calc = TICalc(config.calculation.solvent, *method_args)
      case _:
        raise ValueError("Invalid Method Provided")
    

    
        


  
  


if __name__ == '__main__':
  yaml_file = str(sys.argv(1))
  solvation_calculation()
