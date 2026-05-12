# Databricks notebook source
# MAGIC %md
# MAGIC # **ParametrizANI: Fast, Accurate and Free Parametrization for Small Molecules**
# MAGIC
# MAGIC Dihedral parametrization using **Psi4** for RESP charges combined with **TorchANI** for structural optimization.
# MAGIC
# MAGIC **Reference:** Arantes et al. *J. Chem. Inf. Model.* (2025) doi: [10.1021/acs.jcim.5c01957](http://pubs.acs.org/doi/abs/10.1021/acs.jcim.5c01957)

# COMMAND ----------

# MAGIC %md
# MAGIC # **Setting the environment**

# COMMAND ----------

#@title **Install ParametrizANI and dependencies**
!pip install -q condacolab
import condacolab
condacolab.install_from_url("https://github.com/conda-forge/miniforge/releases/download/25.3.1-0/Miniforge3-Linux-x86_64.sh")

# COMMAND ----------

#@title **Install packages (including Psi4)**
import subprocess
subprocess.run("mamba install -c conda-forge ambertools openmm rdkit openbabel openmmforcefields -y", shell=True)
subprocess.run("pip install torchani ase py3Dmol parmed openff-toolkit", shell=True)
subprocess.run("pip install git+https://github.com/pablo-arantes/ParametrizANI.git", shell=True)
print("\n\u2713 Main dependencies installed!")
print("\nInstalling Psi4 environment...")

# COMMAND ----------

# MAGIC %%bash
# MAGIC #@title **Install Psi4 environment**
# MAGIC mamba create -n psi4_env python=3.11 psi4 resp -c conda-forge --yes > /dev/null 2>&1
# MAGIC source activate psi4_env > /dev/null 2>&1
# MAGIC pip install rdkit Cython > /dev/null 2>&1
# MAGIC mamba install -c conda-forge parmed openbabel --yes > /dev/null 2>&1
# MAGIC echo "✓ Psi4 environment ready"

# COMMAND ----------

#@title **Import ParametrizANI**
from parametrizani import (
    ConformerGenerator,
    ReferenceEnergyCalculator,
    EnergyMinimizer,
    DihedralOptimizer,
    ParameterValidator,
    TopologyGenerator,
    get_dihedral_atom_types,
)
import numpy as np
import os
workDir = "/content/"
print("\u2713 ParametrizANI imported!")

# COMMAND ----------

# MAGIC %md
# MAGIC # **Step 1: Molecule Input & RESP Charges (Psi4)**

# COMMAND ----------

#@title **Molecule Input**
Type = "smiles" #@param ["smiles", "pdb", "mol"]
smiles_or_filename = "COc3ccc2c(=O)cc(c1ccccc1)oc2c3" #@param {type:"string"}

gen = ConformerGenerator(smiles_or_filename, Type, workDir)
conf = gen.run()
print(f"Molecule: {conf['smiles']} ({conf['num_atoms']} atoms)")

import py3Dmol
with open(conf['pdb_file']) as f:
    pdb_data = f.read()
view = py3Dmol.view(width=400, height=300)
view.addModel(pdb_data, "pdb")
view.setStyle({'stick': {}})
view.addPropertyLabels("index", {}, {'fontColor': 'black', 'fontSize': 10})
view.zoomTo()
view.show()

# COMMAND ----------

# MAGIC %%bash
# MAGIC #@title **Calculate RESP Charges with Psi4**
# MAGIC #@markdown Method and basis set for RESP:
# MAGIC #@markdown - Method: HF, B3LYP, or MP2
# MAGIC #@markdown - Basis: 6-31G*
# MAGIC source activate psi4_env
# MAGIC
# MAGIC python << 'EOF'
# MAGIC import psi4
# MAGIC import resp
# MAGIC from rdkit import Chem
# MAGIC from rdkit.Chem import AllChem
# MAGIC import numpy as np
# MAGIC
# MAGIC # Read molecule
# MAGIC mol = Chem.MolFromMolFile("/content/molecule.mol", removeHs=False)
# MAGIC if mol is None:
# MAGIC     mol = Chem.MolFromPDBFile("/content/molecule.pdb", removeHs=False)
# MAGIC
# MAGIC # Get XYZ coordinates
# MAGIC num_atoms = mol.GetNumAtoms()
# MAGIC xyz_string = ""
# MAGIC for i in range(num_atoms):
# MAGIC     pos = mol.GetConformer().GetAtomPosition(i)
# MAGIC     xyz_string += f"{mol.GetAtomWithIdx(i).GetSymbol()} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}\n"
# MAGIC
# MAGIC # Psi4 calculation
# MAGIC psi4.set_memory("2 GB")
# MAGIC psi4.set_num_threads(2)
# MAGIC mol_psi4 = psi4.geometry(f"""
# MAGIC 0 1
# MAGIC {xyz_string}""")
# MAGIC
# MAGIC # RESP options
# MAGIC options = {'BASIS_ESP': '6-31G*', 'METHOD_ESP': 'HF', 'RESP_A': 0.0005, 'RESP_B': 0.1}
# MAGIC charges = resp.resp([mol_psi4], options)
# MAGIC
# MAGIC # Save charges
# MAGIC np.savetxt("/content/resp_charges.dat", charges[0][1])
# MAGIC print(f"RESP charges calculated for {num_atoms} atoms")
# MAGIC print(charges[0][1])
# MAGIC EOF

# COMMAND ----------

# MAGIC %md
# MAGIC # **Step 2: Dihedral Scan & Reference Energies**

# COMMAND ----------

#@title **Set Dihedral and Generate Conformers**
atom1 = 8 #@param {type:"integer"}
atom2 = 9 #@param {type:"integer"}
atom3 = 10 #@param {type:"integer"}
atom4 = 15 #@param {type:"integer"}
degrees_steps = 30 #@param [5, 10, 15, 20, 30, 45, 60] {type:"raw"}
min_deg = -180 #@param {type:"slider", min:-180, max:180, step:10}
max_deg = 180 #@param {type:"slider", min:-180, max:180, step:10}

dihedral_indices = [atom1, atom2, atom3, atom4]
scan = gen.generate_dihedral_conformers(dihedral_indices, min_deg, max_deg, int(degrees_steps))
print(f"\u2713 Generated {len(scan['angles'])} conformers")

# COMMAND ----------

#@title **Reference Energy (TorchANI optimization + energy)**
model_name = "torchani" #@param ["torchani", "mace", "xtb", "aimnet2"]

calc = ReferenceEnergyCalculator(model_name, workDir)
ref = calc.scan_dihedral(scan['conformers'], scan['angles'], optimize=True)

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
plt.plot(ref['angles'], ref['energies_relative'], 'o-', lw=1.5, label=model_name)
plt.xlabel('Dihedral Angle (degrees)')
plt.ylabel('Relative Energy (kcal/mol)')
plt.legend()
plt.savefig('reference_profile.png', dpi=300, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # **Step 3: MM Minimization & Optimization**

# COMMAND ----------

#@title **Generate GAFF2 Topology with RESP Charges & Minimize**
topo = TopologyGenerator(workDir, force_field='gaff2')
amber_files = topo.generate_amber(conf['mol_file'], charge_method='resp')

# Automatically detect atom types from MOL2
atom_types = get_dihedral_atom_types(amber_files['mol2'], dihedral_indices)
print(f"  Atom types for dihedral: {atom_types}")

minimizer = EnergyMinimizer('gaff2', workDir)
mm = minimizer.minimize_scan(
    amber_files['prmtop'], amber_files['inpcrd'],
    scan['pdb_files'], dihedral_indices,
    angles=scan['angles'], zero_dihedral=True
)
print(f"\u2713 MM energy range: {max(mm['energies_relative']):.3f} kcal/mol")

# COMMAND ----------

#@title **Optimize & Validate**
max_terms = 4 #@param [1, 2, 3, 4, 5, 6] {type:"raw"}

optimizer = DihedralOptimizer(max_terms=int(max_terms), work_dir=workDir)
result = optimizer.run_optimization(
    ref['angles'], ref['energies_relative'],
    mm_energies=mm['energies_relative'],
    atom_types=atom_types
)

validator = ParameterValidator(workDir)
val = validator.validate_parameters(ref['angles'], ref['energies_relative'], result['best_fit'])

print(f"RMSE: {result['rmse']:.4f} kcal/mol  |  Quality: {val['quality']}  |  R²: {val['r_squared']:.4f}")
print(f"\nParameters:\n{result['frcmod_params']}")

plt.figure(figsize=(8, 5))
plt.plot(ref['angles'], ref['energies_relative'], 'o-', lw=1.5, label=model_name)
plt.plot(mm['angles'], mm['energies_relative'], 's--', lw=1.0, label="GAFF2", alpha=0.7)
plt.plot(result['angles'], result['best_fit'], 'D-', lw=1.5, label="Optimized")
plt.xlabel('Dihedral Angle (degrees)')
plt.ylabel('Relative Energy (kcal/mol)')
plt.legend()
plt.title(f'TorchANI+Psi4: RMSE={result["rmse"]:.4f} kcal/mol')
plt.savefig('psi4_optimized.png', dpi=300, bbox_inches='tight')
plt.show()

# COMMAND ----------

#@title **Download Results**
from google.colab import files
import zipfile
with zipfile.ZipFile("parametrizani_psi4_results.zip", 'w') as zf:
    for f in os.listdir(workDir):
        path = os.path.join(workDir, f)
        if os.path.isfile(path) and (f.endswith('.dat') or f.endswith('.png') or f.endswith('.txt') or f.endswith('.frcmod')):
            zf.write(path, f)
files.download("parametrizani_psi4_results.zip")