# Databricks notebook source
# MAGIC %md
# MAGIC # **ParametrizANI: Rotational Profile Fitting**
# MAGIC
# MAGIC Fit an empirical energy profile to a reference profile using the Rotational Profiler algorithm.
# MAGIC
# MAGIC The reference profile can be obtained from:
# MAGIC - Quantum mechanical (QM) calculations
# MAGIC - Machine learning models (TorchANI, MACE, xTB)
# MAGIC - Experimental data
# MAGIC
# MAGIC **Reference:** Arantes et al. *J. Chem. Inf. Model.* (2025) doi: [10.1021/acs.jcim.5c01957](http://pubs.acs.org/doi/abs/10.1021/acs.jcim.5c01957)

# COMMAND ----------

#@title **Install ParametrizANI**
!pip install git+https://github.com/pablo-arantes/ParametrizANI.git
print("\u2713 ParametrizANI installed!")

# COMMAND ----------

#@title **Import**
from parametrizani import (
    DihedralOptimizer,
    ParameterValidator,
    read_energy_file,
    write_energy_file,
)
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
import os

workDir = "/content/"
print("\u2713 Ready!")

# COMMAND ----------

# MAGIC %md
# MAGIC # **Upload Energy Files**
# MAGIC
# MAGIC Upload your reference (QM) and empirical (MM/force field) energy files.
# MAGIC
# MAGIC **File format:** Two columns - `angle(degrees) energy(kcal/mol)`
# MAGIC
# MAGIC Example:
# MAGIC ```
# MAGIC 0      0.0005
# MAGIC 30     5.7147
# MAGIC 60    21.6567
# MAGIC ...
# MAGIC ```
# MAGIC
# MAGIC You can find examples at:
# MAGIC - [qm.dat](https://github.com/pablo-arantes/ParametrizANI/raw/main/examples/qm.dat)
# MAGIC - [mm.dat](https://github.com/pablo-arantes/ParametrizANI/raw/main/examples/mm.dat)

# COMMAND ----------

#@title **Upload Reference and MM Energy Files**
#@markdown Run this cell, then upload your files:
#@markdown 1. First upload: **Reference energy file** (QM/ML)
#@markdown 2. Second upload: **MM energy file** (force field)

from google.colab import files
import sys

print("Upload your REFERENCE energy file (QM/ML):")
uploaded = files.upload()
for fn in uploaded.keys():
    os.rename(fn, "QM.dat")
print(f"  \u2713 Saved as QM.dat")

print("\nUpload your MM energy file (force field):")
uploaded = files.upload()
for fn in uploaded.keys():
    os.rename(fn, "MM.dat")
print(f"  \u2713 Saved as MM.dat")

# COMMAND ----------

#@title **Load and Preview Data**

ref_angles, ref_energies = read_energy_file("QM.dat")
mm_angles, mm_energies = read_energy_file("MM.dat")

print(f"Reference: {len(ref_angles)} points, range {min(ref_angles):.0f}° to {max(ref_angles):.0f}°")
print(f"  Energy range: {min(ref_energies):.3f} to {max(ref_energies):.3f} kcal/mol")
print(f"\nMM: {len(mm_angles)} points")
print(f"  Energy range: {min(mm_energies):.3f} to {max(mm_energies):.3f} kcal/mol")

plt.figure(figsize=(10, 5))
plt.plot(ref_angles, ref_energies, 'o-', lw=1.5, label='Reference (QM/ML)')
plt.plot(mm_angles, mm_energies, 's-', lw=1.5, label='MM (Force Field)')
plt.xlabel('Dihedral Angle (degrees)', fontsize=12)
plt.ylabel('Energy (kcal/mol)', fontsize=12)
plt.legend(fontsize=12)
plt.title('Input Energy Profiles')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # **Fit Dihedral Parameters**

# COMMAND ----------

#@title **Optimize Dihedral Parameters**
#@markdown Maximum number of Fourier terms to fit:
max_terms = 6 #@param [1, 2, 3, 4, 5, 6] {type:"raw"}

#@markdown Atom types for FRCMOD output (from antechamber MOL2):
atom_type_1 = "ca" #@param {type:"string"}
atom_type_2 = "ca" #@param {type:"string"}
atom_type_3 = "ca" #@param {type:"string"}
atom_type_4 = "ca" #@param {type:"string"}

optimizer = DihedralOptimizer(max_terms=int(max_terms), work_dir=workDir)
result = optimizer.run_optimization(
    ref_angles, ref_energies,
    mm_energies=mm_energies,
    atom_types=[atom_type_1, atom_type_2, atom_type_3, atom_type_4]
)

print("\n" + "="*60)
print("OPTIMIZATION RESULTS")
print("="*60)
print(f"\nRMSE per number of Fourier terms:")
for i, rmse in enumerate(result['all_rmse'], 1):
    marker = " ← best" if i == int(max_terms) else ""
    print(f"  {i} terms: {rmse:.4f} kcal/mol{marker}")

print(f"\nBest fit ({max_terms} terms):")
print(f"  RMSE = {result['rmse']:.4f} kcal/mol")
print(f"\nOptimized FRCMOD Parameters:")
print(result['frcmod_params'])

# Write FRCMOD file
frcmod_file = optimizer.write_frcmod(result)
print(f"\n\u2713 FRCMOD saved: {frcmod_file}")

# COMMAND ----------

#@title **Visualize All Fitted Curves**

plt.figure(figsize=(12, 6))
plt.plot(ref_angles, ref_energies, 'o-', lw=2, ms=6, label='Reference (QM/ML)', color='red')
plt.plot(mm_angles, mm_energies, 's-', lw=2, ms=6, label='MM (Force Field)', color='blue')

colors = ['#4CAF50', '#FF9800', '#9C27B0', '#00BCD4', '#E91E63', '#795548']
for i, fitted in enumerate(result['energies_fitted']):
    rmse = result['all_rmse'][i]
    plt.plot(ref_angles, fitted, '^--', lw=1.2, ms=4, 
             label=f'{i+1} terms (RMSE={rmse:.3f})', color=colors[i % len(colors)], alpha=0.8)

plt.xlabel('Dihedral Angle (degrees)', fontsize=14)
plt.ylabel('Energy (kcal/mol)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=10, loc='best')
plt.title('Rotational Profile Fitting', fontsize=14)
plt.tight_layout()
plt.savefig('fitted_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # **Validation**

# COMMAND ----------

#@title **Validate Best Fit**

validator = ParameterValidator(workDir)
val = validator.validate_parameters(
    ref_angles, ref_energies, result['best_fit'],
    mm_energies=mm_energies,
    labels={'reference': 'QM/ML', 'fitted': 'Optimized', 'mm': 'MM (original)'}
)

print("\n" + "="*60)
print("VALIDATION REPORT")
print("="*60)
print(f"  Quality Rating: {val['quality']}")
print(f"  RMSE:          {val['rmse']:.4f} kcal/mol")
print(f"  MAE:           {val['mae']:.4f} kcal/mol")
print(f"  R²:            {val['r_squared']:.4f}")
print(f"  Correlation:   {val['correlation']:.4f}")
print(f"  Max Error:     {val['max_error']:.4f} kcal/mol")
print(f"\n  Report: {val['report_file']}")
print(f"  Plot: {val['plot_file']}")

# COMMAND ----------

#@title **Download Results**
from google.colab import files
import zipfile

with zipfile.ZipFile("rotprof_results.zip", 'w') as zf:
    for f in os.listdir(workDir):
        path = os.path.join(workDir, f)
        if os.path.isfile(path) and (f.endswith('.dat') or f.endswith('.png') or 
                                      f.endswith('.txt') or f.endswith('.frcmod')):
            zf.write(path, f)

files.download("rotprof_results.zip")
print("\u2713 Results downloaded!")