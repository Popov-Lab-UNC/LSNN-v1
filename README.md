# LSNN-v1

## Abstract

The implicit solvent approach offers a computationally efficient framework to model solvation effects in molecular simulations. However, its accuracy often falls short compared to explicit solvent models, limiting its broader utility. Recent advancements in machine learning (ML) present an opportunity to overcome these limitations by leveraging neural networks to develop more precise implicit solvent potentials for diverse applications. However, a major drawback of current approaches is their reliance on force-matching alone, resulting in energy predictions that are offset by an arbitrary constant, limiting their use in free energy calculations. Here, we introduce a novel methodology with a graph neural network (GNN)-based implicit solvent model, dubbed $\lambda$-Solvation Neural Network (LSNN), built upon a comprehensive non-bonded PMF function, incorporating electrostatic and steric interactions derived from alchemically modified explicit-water simulations. This approach enables accurate potential calculations across diverse molecular microstates. Trained on a dataset of approximately 300,000 small molecules, LSNN achieves free energy predictions with accuracy comparable to explicit-solvent alchemical simulations, while offering a significant computational speedup and establishing a foundational framework for future applications in drug discovery.


## Usage
Download the necessary dependencies:
```bash
mamba env create -f environment.yml
```

Configure the config.yml file:

```bash
molecule:
  name: null
  cache_path: null #all files will save under cache_path/{name}/
  smiles: null
  file_path: null #save directory for SMILES
  expt: null

calculation:
  solvent: LSNN-MBAR #obc2, gbn2, tip3p, LSNN-MBAR, LSNN-TI
  device: auto #CUDA, CPU, auto 
  lambda_electrostatics: [0.0, 1.0]
  lambda_sterics: [0.0, 1.0]
  model_dict_path: LSNN-v1/Best_Trained_Models/280KDATASET2Kv3model.dict
  
  report_interval: 100
  steps: 300
```

Run the command: 
```bash
python solv.py config-yml
```

