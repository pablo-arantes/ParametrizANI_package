# Databricks notebook source
# MAGIC %md
# MAGIC # **ParametrizANI: Fast, Accurate and Free Parametrization for Small Molecules**
# MAGIC
# MAGIC Dihedral parametrization of small molecules for **OpenFF** force fields using state-of-the-art reference methods: TorchANI, AIMNet2, MACE-OFF, or GFN2-xTB.
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

#@title **Install packages**
import subprocess
subprocess.run("mamba install -c conda-forge ambertools rdkit openbabel openmm openmmforcefields -y", shell=True)
subprocess.run("conda install -c conda-forge xtb-python -y", shell=True)
subprocess.run("pip install torchani ase py3Dmol parmed openff-toolkit", shell=True)
subprocess.run("pip install mace-torch e3nn==0.4.4", shell=True)
subprocess.run("pip install git+https://github.com/pablo-arantes/ParametrizANI.git", shell=True)
print("\n\u2713 All dependencies installed!")

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
# MAGIC # **Step 1: Molecule Input**

# COMMAND ----------

#@title **Provide Your Molecule**
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

# MAGIC %md
# MAGIC # **Step 2: Dihedral Selection & Conformer Generation**

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

# Automatically detect GAFF2 atom types for FRCMOD output
topo = TopologyGenerator(workDir, force_field='gaff2')
amber_files = topo.generate_amber(conf['mol_file'], charge_method='am1bcc')
atom_types = get_dihedral_atom_types(amber_files['mol2'], dihedral_indices)
print(f"  Atom types for dihedral: {atom_types}")

# COMMAND ----------

# MAGIC %md
# MAGIC # **Step 3: Reference Energy Calculation**

# COMMAND ----------

#@title **Calculate Reference Energies**
model_name = "torchani" #@param ["torchani", "mace", "xtb", "aimnet2"]
device = "cpu" #@param ["cpu", "cuda"]

calc = ReferenceEnergyCalculator(model_name, workDir, device=device)
ref = calc.scan_dihedral(scan['conformers'], scan['angles'], optimize=True)

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(8, 5))
plt.plot(ref['angles'], ref['energies_relative'], 'o-', linewidth=1.5, label=model_name)
plt.xlabel('Dihedral Angle (degrees)')
plt.ylabel('Relative Energy (kcal/mol)')
plt.legend(frameon=True)
plt.savefig(f'{model_name}_reference.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"\u2713 Energy range: {max(ref['energies_relative']):.3f} kcal/mol")

# COMMAND ----------

# MAGIC %md
# MAGIC # **Step 4: OpenFF Minimization**

# COMMAND ----------

#@title **OpenMM Minimization with OpenFF Force Field**
Force_field = "openff-2.0.0" #@param ["openff-2.0.0", "openff-1.3.1", "openff-1.2.0", "smirnoff99Frosst-1.1.0"]
Dihedral_potential_energy = "0" #@param ["0", "Default"]
Force_constant = 1000 #@param {type:"slider", min:100, max:2000, step:100}
opt_tol = 0.001 #@param {type:"slider", min:0.001, max:0.1, step:0.001}

minimizer = EnergyMinimizer(Force_field, workDir)
mm = minimizer.minimize_scan_openff(
    smiles_or_filename if Type == "smiles" else conf['smiles'],
    scan['pdb_files'],
    dihedral_indices,
    angles=scan['angles'],
    zero_dihedral=(Dihedral_potential_energy == "0"),
    force_constant=Force_constant,
    opt_tol=opt_tol,
)

plt.figure(figsize=(8, 5))
plt.plot(ref['angles'], ref['energies_relative'], 'o-', linewidth=1.5, label=model_name)
plt.plot(mm['angles'], mm['energies_relative'], 's-', linewidth=1.5, label=Force_field)
plt.xlabel('Dihedral Angle (degrees)')
plt.ylabel('Relative Energy (kcal/mol)')
plt.legend(frameon=True)
plt.savefig('ref_vs_openff.png', dpi=300, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # **Step 5: Optimize Dihedral Parameters**

# COMMAND ----------

#@title **Dihedral Optimization**
max_terms = 4 #@param [1, 2, 3, 4, 5, 6] {type:"raw"}

optimizer = DihedralOptimizer(max_terms=int(max_terms), work_dir=workDir)
result = optimizer.run_optimization(
    ref['angles'], ref['energies_relative'],
    mm_energies=mm['energies_relative'],
    atom_types=atom_types
)

print(f"Best RMSE ({max_terms} terms): {result['rmse']:.4f} kcal/mol")
print(f"\nParameters:")
print(result['frcmod_params'])

plt.figure(figsize=(8, 5))
plt.plot(ref['angles'], ref['energies_relative'], 'o-', lw=1.5, label=model_name)
plt.plot(mm['angles'], mm['energies_relative'], 's--', lw=1.0, label=Force_field, alpha=0.7)
plt.plot(result['angles'], result['best_fit'], 'D-', lw=1.5, label="Optimized")
plt.xlabel('Dihedral Angle (degrees)')
plt.ylabel('Relative Energy (kcal/mol)')
plt.legend(frameon=True)
plt.title(f'Optimization Result (RMSE: {result["rmse"]:.4f} kcal/mol)')
plt.savefig('optimized_openff.png', dpi=300, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # **Step 6: Validation**

# COMMAND ----------

#@title **Validate Parameters**
validator = ParameterValidator(workDir)
val = validator.validate_parameters(
    ref['angles'], ref['energies_relative'], result['best_fit'],
    mm_energies=mm['energies_relative']
)
print(f"Quality: {val['quality']}  |  RMSE: {val['rmse']:.4f}  |  R²: {val['r_squared']:.4f}")

# COMMAND ----------

#@title **Download Results**
from google.colab import files
import zipfile

with zipfile.ZipFile("parametrizani_openff_results.zip", 'w') as zf:
    for f in os.listdir(workDir):
        path = os.path.join(workDir, f)
        if os.path.isfile(path) and (f.endswith('.dat') or f.endswith('.png') or f.endswith('.txt')):
            zf.write(path, f)
files.download("parametrizani_openff_results.zip")