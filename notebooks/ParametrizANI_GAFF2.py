# Databricks notebook source
# MAGIC %md
# MAGIC # **ParametrizANI: Fast, Accurate and Free Parametrization for Small Molecules**
# MAGIC
# MAGIC Dihedral parametrization of small molecules for **GAFF2** force field using state-of-the-art reference methods: TorchANI, AIMNet2, MACE-OFF, or GFN2-xTB.
# MAGIC
# MAGIC **Reference:** Arantes et al. *J. Chem. Inf. Model.* (2025) doi: [10.1021/acs.jcim.5c01957](http://pubs.acs.org/doi/abs/10.1021/acs.jcim.5c01957)
# MAGIC
# MAGIC ---
# MAGIC ## Workflow:
# MAGIC ```
# MAGIC SMILES/PDB/MOL → Conformer Generation → Reference Energy (ML) → MM Minimization → Dihedral Optimization → Validation → Topology
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC # **Setting the environment**
# MAGIC
# MAGIC Install the `parametrizani` package and all required dependencies.

# COMMAND ----------

#@title **Install ParametrizANI and dependencies**
#@markdown This may take a few minutes.

!pip install -q condacolab
import condacolab
condacolab.install_from_url("https://github.com/conda-forge/miniforge/releases/download/25.3.1-0/Miniforge3-Linux-x86_64.sh")

# COMMAND ----------

#@title **Install molecular dynamics and ML packages**
#@markdown Please wait while dependencies are installed.

import subprocess, sys

# Install conda packages
subprocess.run("mamba install -c conda-forge ambertools openmm rdkit openbabel openmmforcefields -y", shell=True)
subprocess.run("conda install -c conda-forge xtb-python -y", shell=True)

# Install pip packages
subprocess.run("pip install torchani ase py3Dmol parmed", shell=True)
subprocess.run("pip install openff-toolkit", shell=True)

# Install MACE-OFF
subprocess.run("pip install mace-torch e3nn==0.4.4", shell=True)

# Install parametrizani
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
    read_energy_file,
)
import numpy as np
import os

workDir = "/content/"
print("\u2713 ParametrizANI imported successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC # **Step 1: Provide Your Molecule**
# MAGIC
# MAGIC Enter the SMILES string or filename of your molecule.

# COMMAND ----------

#@title **Molecule Input**
#@markdown Enter SMILES or filename (PDB/MOL). Upload files to Colab first.

Type = "smiles" #@param ["smiles", "pdb", "mol"]
smiles_or_filename = "COc3ccc2c(=O)cc(c1ccccc1)oc2c3" #@param {type:"string"}

# Generate conformer
gen = ConformerGenerator(smiles_or_filename, Type, workDir)
conf = gen.run()

print(f"Molecule: {conf['smiles']}") 
print(f"Atoms: {conf['num_atoms']}")
print(f"PDB: {conf['pdb_file']}")

# Visualize
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
# MAGIC # **Step 2: Set Dihedral Angle**
# MAGIC
# MAGIC Specify the 4 atom indices defining the dihedral to parametrize.

# COMMAND ----------

#@title **Dihedral Selection**
#@markdown Atom indices (0-based) for the dihedral: atom1-atom2-atom3-atom4

atom1 = 8 #@param {type:"integer"}
atom2 = 9 #@param {type:"integer"}
atom3 = 10 #@param {type:"integer"}
atom4 = 15 #@param {type:"integer"}

#@markdown Step size for dihedral scan (degrees):
degrees_steps = 30 #@param [1, 2, 3, 4, 5, 10, 15, 20, 30, 45, 60] {type:"raw"}

#@markdown Angle range:
min_deg = -180 #@param {type:"slider", min:-180, max:180, step:10}
max_deg = 180 #@param {type:"slider", min:-180, max:180, step:10}

dihedral_indices = [atom1, atom2, atom3, atom4]
print(f"Dihedral: {dihedral_indices}")
print(f"Scan: {min_deg}° to {max_deg}°, step {degrees_steps}°")

# COMMAND ----------

# MAGIC %md
# MAGIC # **Step 3: Generate Conformers**
# MAGIC
# MAGIC Generate structures by rotating the dihedral angle.

# COMMAND ----------

#@title **Generate Dihedral Conformers**

scan = gen.generate_dihedral_conformers(
    dihedral_indices,
    min_angle=min_deg,
    max_angle=max_deg,
    step=int(degrees_steps)
)

print(f"\u2713 Generated {len(scan['angles'])} conformers")
print(f"  Angles: {scan['angles']}")
print(f"  Files in: {os.path.join(workDir, 'mol_files/')}")

# COMMAND ----------

# MAGIC %md
# MAGIC # **Step 4: Calculate Reference Energy Profile**
# MAGIC
# MAGIC Use a ML potential as the reference for parametrization.

# COMMAND ----------

#@title **Reference Energy Calculation**
#@markdown Select the ML method for reference energies:

model_name = "torchani" #@param ["torchani", "mace", "xtb", "aimnet2"]
#@markdown Use GPU if available:
device = "cpu" #@param ["cpu", "cuda"]

calc = ReferenceEnergyCalculator(model_name, workDir, device=device)
ref = calc.scan_dihedral(
    scan['conformers'],
    scan['angles'],
    optimize=True
)

print(f"\u2713 Reference energies calculated with {model_name}")
print(f"  Energy range: {max(ref['energies_relative']):.3f} kcal/mol")
if model_name == "torchani":
    print(f"  Max RHO (reliability): {max(ref['rho_values']):.4f} kcal/mol")
    if max(ref['rho_values']) > 0.6:
        print("  ⚠️ WARNING: High RHO - results may be unreliable!")

# COMMAND ----------

#@title **Visualize Reference Energy Profile**

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')

plt.figure(figsize=(8, 5))
plt.plot(ref['angles'], ref['energies_relative'], 'o-', linewidth=1.5, label=model_name)
plt.xticks(np.arange(min_deg, max_deg+1, 60.0))
plt.xlabel('Dihedral Angle (degrees)')
plt.ylabel('Relative Energy (kcal/mol)')
plt.legend(frameon=True)
plt.title('Reference Energy Profile')
plt.savefig(f'{model_name}_reference.png', dpi=300, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # **Step 5: GAFF2 Topology & MM Minimization**
# MAGIC
# MAGIC Generate GAFF2 topology and minimize with the target dihedral zeroed out.

# COMMAND ----------

#@title **Generate GAFF2 Topology**
#@markdown Charge model:
charge_model = "AM1-BCC" #@param ["AM1-BCC", "RESP"]

topo = TopologyGenerator(workDir, force_field='gaff2')
amber_files = topo.generate_amber(
    conf['mol_file'],
    charge_method='am1bcc' if charge_model == "AM1-BCC" else 'resp'
)

# Automatically detect atom types from MOL2
atom_types = get_dihedral_atom_types(amber_files['mol2'], dihedral_indices)

print(f"\u2713 GAFF2 topology generated")
print(f"  Atom types for dihedral: {atom_types}")
for key, path in amber_files.items():
    if os.path.exists(str(path)):
        print(f"  {key}: {path}")

# COMMAND ----------

#@title **OpenMM Minimization (Dihedral Zeroed)**
#@markdown Set dihedral potential to 0 for reparametrization:
Dihedral_potential_energy = "0" #@param ["0", "Default"]
#@markdown Force constant for dihedral restraint:
Force_constant = 1000 #@param {type:"slider", min:100, max:2000, step:100}
#@markdown Convergence tolerance:
opt_tol = 0.001 #@param {type:"slider", min:0.001, max:0.1, step:0.001}

minimizer = EnergyMinimizer('gaff2', workDir)
mm = minimizer.minimize_scan(
    amber_files['prmtop'],
    amber_files['inpcrd'],
    scan['pdb_files'],
    dihedral_indices,
    angles=scan['angles'],
    zero_dihedral=(Dihedral_potential_energy == "0"),
    force_constant=Force_constant,
    opt_tol=opt_tol,
)

print(f"\u2713 MM minimization complete")
print(f"  MM energy range: {max(mm['energies_relative']):.3f} kcal/mol")

# COMMAND ----------

#@title **Visualize Reference vs MM Energy Profiles**

plt.figure(figsize=(8, 5))
plt.plot(ref['angles'], ref['energies_relative'], 'o-', linewidth=1.5, label=model_name)
plt.plot(mm['angles'], mm['energies_relative'], 's-', linewidth=1.5, label="GAFF2")
plt.xticks(np.arange(min_deg, max_deg+1, 60.0))
plt.xlabel('Dihedral Angle (degrees)')
plt.ylabel('Relative Energy (kcal/mol)')
plt.legend(frameon=True)
plt.title('Reference vs GAFF2 Energy Profile')
plt.savefig(f'{model_name}_vs_gaff2.png', dpi=300, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # **Step 6: Optimize Dihedral Parameters**
# MAGIC
# MAGIC Fit Fourier series to reproduce the reference energy profile.

# COMMAND ----------

#@title **Dihedral Optimization (Rotational Profiler)**
#@markdown Maximum Fourier terms:
max_terms = 4 #@param [1, 2, 3, 4, 5, 6] {type:"raw"}

optimizer = DihedralOptimizer(max_terms=int(max_terms), work_dir=workDir)
result = optimizer.run_optimization(
    ref['angles'],
    ref['energies_relative'],
    mm_energies=mm['energies_relative'],
    atom_types=atom_types
)

print(f"\u2713 Optimization complete!")
print(f"\nRMSE per number of terms:")
for i, rmse in enumerate(result['all_rmse'], 1):
    print(f"  {i} terms: {rmse:.4f} kcal/mol")
print(f"\nBest fit ({max_terms} terms): RMSE = {result['rmse']:.4f} kcal/mol")
print(f"\nFRCMOD Parameters:")
print(result['frcmod_params'])

# COMMAND ----------

#@title **Visualize Optimized Dihedral Profile**

plt.figure(figsize=(8, 5))
plt.plot(ref['angles'], ref['energies_relative'], 'o-', linewidth=1.5, label=model_name)
plt.plot(mm['angles'], mm['energies_relative'], 's--', linewidth=1.0, label="GAFF2 (original)", alpha=0.7)
plt.plot(result['angles'], result['best_fit'], 'D-', linewidth=1.5, label="Optimized")
plt.xticks(np.arange(min_deg, max_deg+1, 60.0))
plt.xlabel('Dihedral Angle (degrees)')
plt.ylabel('Relative Energy (kcal/mol)')
plt.legend(frameon=True)
plt.title(f'Dihedral Optimization (RMSE: {result["rmse"]:.4f} kcal/mol)')
plt.savefig('optimized_profile.png', dpi=300, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # **Step 7: Validation**

# COMMAND ----------

#@title **Validate Optimized Parameters**

validator = ParameterValidator(workDir)
val = validator.validate_parameters(
    ref['angles'],
    ref['energies_relative'],
    result['best_fit'],
    mm_energies=mm['energies_relative'],
    labels={'reference': model_name, 'fitted': 'Optimized', 'mm': 'GAFF2 original'}
)

print(f"\n✓ Validation Results:")
print(f"  Quality: {val['quality']}")
print(f"  RMSE: {val['rmse']:.4f} kcal/mol")
print(f"  MAE: {val['mae']:.4f} kcal/mol")
print(f"  R²: {val['r_squared']:.4f}")
print(f"  Correlation: {val['correlation']:.4f}")
print(f"\n  Report: {val['report_file']}")
print(f"  Plot: {val['plot_file']}")

# COMMAND ----------

# MAGIC %md
# MAGIC # **Step 8: Generate Updated Topology**
# MAGIC
# MAGIC Write the FRCMOD with optimized parameters and generate final topology files.

# COMMAND ----------

#@title **Write FRCMOD and Generate Topology**
#@markdown IDIVF (scaling factor for equivalent torsions):
IDIVF = 1 #@param [1, 2, 3, 4] {type:"raw"}
#@markdown Output format:
program = "GROMACS" #@param ["AMBER", "GROMACS", "OpenMM"]

# Write optimized FRCMOD
frcmod_file = optimizer.write_frcmod(result, idivf=int(IDIVF))
print(f"\u2713 FRCMOD written: {frcmod_file}")

# Update topology with new parameters
updated_frcmod = topo.update_frcmod(amber_files['frcmod'], result['frcmod_params'])

# Generate topology in requested format
try:
    if program == "AMBER":
        files = topo.generate_amber(conf['mol_file'], updated_frcmod)
        print(f"\u2713 AMBER files: {files['prmtop']}")
    elif program == "GROMACS":
        amber = topo.generate_amber(conf['mol_file'], updated_frcmod)
        files = topo.generate_gromacs(amber['prmtop'], amber['inpcrd'])
        print(f"\u2713 GROMACS files: {files['top']}")
    elif program == "OpenMM":
        amber = topo.generate_amber(conf['mol_file'], updated_frcmod)
        files = topo.generate_openmm(amber['prmtop'], amber['inpcrd'])
        print(f"\u2713 OpenMM files: {files['xml']}")
except Exception as e:
    print(f"Topology generation: {e}")

# COMMAND ----------

#@title **Download Results**
from google.colab import files
import zipfile

# Package all results
zip_name = "parametrizani_results.zip"
with zipfile.ZipFile(zip_name, 'w') as zf:
    for f in os.listdir(workDir):
        path = os.path.join(workDir, f)
        if os.path.isfile(path) and (f.endswith('.dat') or f.endswith('.png') or f.endswith('.frcmod') or f.endswith('.txt')):
            zf.write(path, f)
    # Add topology files
    tleap_dir = os.path.join(workDir, 'tleap_output')
    if os.path.exists(tleap_dir):
        for f in os.listdir(tleap_dir):
            zf.write(os.path.join(tleap_dir, f), f"topology/{f}")

files.download(zip_name)
print(f"\u2713 Downloaded: {zip_name}")