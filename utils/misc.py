import subprocess

def smi_to_protonated_sdf(smiles, output_path):
    cmd = f'obabel "-:{smiles}" -O "{output_path}" --gen3d --pH 7'
    subprocess.run(cmd, check=True, shell=True, timeout=60)
