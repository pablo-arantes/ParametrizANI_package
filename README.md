# ParametrizANI: Fast and Accessible Dihedral Parametrization for Small Molecules

![alt text](https://github.com/pablo-arantes/ParametrizANI/blob/main/TOC_graphic.png)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Hi there!

Welcome to ParametrizANI, an innovative and free tool designed to address the growing demand for accurate parametrization of small molecules in molecular studies. Our goal is to democratize research by providing a research-friendly environment that is free from resource constraints, enabling teams of all sizes to perform dihedral parametrization with DFT-level accuracy.

We have now integrated **TorchANI2** into our pipeline, along with the latest improved ANI models trained on the expanded 2× dataset (**ANI-2xr, ANI-2dr, ANI-2xr-Snn, and ANI-mbis**).
These models were trained at the B97-3c level of theory and include explicit repulsion and dispersion corrections, smoother potential energy surfaces (PES), and MBIS-derived charges.

ParametrizANI is a production-ready Python package for dihedral parameter optimization using neural network potentials and molecular mechanics. It provides an end-to-end pipeline from SMILES strings to force field parameters compatible with AMBER, GROMACS, and OpenMM.

## Notebooks

**Notebook A** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-arantes/ParametrizANI/blob/main/ParametrizANI_GAFF2.ipynb) - `Dihedral parametrization of small molecules for GAFF force field using state-of-the-art reference methods such as TorchANI, AIMNet2, MACE-OFF or GFN2-xTB.`

**Notebook B** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-arantes/ParametrizANI/blob/main/ParametrizANI_OpenFF.ipynb) - `Dihedral parametrization of small molecules for OpenFF force fields using state-of-the-art reference methods such as TorchANI, AIMNet2, MACE-OFF or GFN2-xTB.`

**Notebook C** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-arantes/ParametrizANI/blob/main/ParametrizANI_TorchANI%2BPsi4.ipynb) - `Dihedral parametrization of small molecules using a reference potential computed with Psi4, combined with structural optimization from TorchANI.`

**Notebook D** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-arantes/ParametrizANI/blob/main/ParametrizANI_RotProf.ipynb) - `Rotational Profile – fits an empirical energy profile to a reference profile, which can be obtained experimentally, through quantum mechanical (QM) calculations, or using machine learning models such as TorchANI.`

**Notebook E** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pablo-arantes/ParametrizANI/blob/main/ParametrizANI_RESP_charges.ipynb) - `Perform a geometry optimization with TorchANI, AIMNet2, or GFN2-xTB, then compute RESP charges to generate the GAFF topology parameters.`

## Key Features and Benefits

**• Robust Backbone:** ParametrizANI leverages TorchANI, a robust PyTorch-based deep learning program, as its benchmark to ensure precision in parametrization tasks. TorchANI is crucial for training and inference of ANI (ANAKIN-ME) deep learning models, which are fundamental for predicting potential energy surfaces and other molecular system attributes.

**• Accuracy and Efficiency:** ParametrizANI establishes detailed protocols for dihedral parametrization using both GAFF and OpenFF force fields. By integrating TorchANI's predictive power, ParametrizANI offers a streamlined and accurate approach to parametrization, especially for small molecules. TorchANI's neural network models predict molecular energies and properties with high accuracy and efficiency, significantly reducing computation time compared to traditional Quantum Mechanical (QM) methods.

**• Cloud-Based Accessibility:** The tool harnesses the power of Google Colaboratory (Colab), a hosted Jupyter Notebook service that provides free access to computing resources. This makes ParametrizANI a feasible, cost-effective, and accessible approach to compound parametrization, particularly beneficial for investigators worldwide, including those with limited resources. Our notebooks are designed to run efficiently on CPU cores, requiring no heavy parallel processing.

**• Comprehensive Workflow:** ParametrizANI provides comprehensive workflows implemented in Google Colab notebooks, exemplifying a complete pipeline for dihedral parametrization from SMILES strings generation to force field parameter optimization. These workflows enable researchers to efficiently perform accurate and reliable dihedral parametrization.

**• Versatile and Customizable:** The notebooks are designed for ease of use, following the Jupyter Notebook structure, with an initial configuration step taking less than 5 minutes. Users can select between GAFF and OpenFF force fields, choose charge calculation methods (AM1-BCC or RESP), and even upload their own reference energy profiles calculated using external software (e.g., Gaussian, GAMESS). This flexibility allows for customization to specific research requirements and professional use.

**• Broad Applicability:** ParametrizANI is not only suited for advanced molecular dynamics research and computational drug discovery but also serves as an excellent tool for educational purposes. It allows students to independently run the entire parametrization process without local software compilation or extensive coding experience, with embedded visualization at each step.

## Installation

### Step 1: Install conda dependencies (required)

Several packages (`openmm`, `openff-toolkit`, `ambertools`, `rdkit`, `xtb-python`) are only available via conda-forge:

```bash
conda install -c conda-forge ambertools openmm openmmforcefields rdkit openbabel openff-toolkit xtb-python -y
```

### Step 2: Install parametrizani

```bash
cd parametrizani
pip install -e .
```

### Step 3: Install ML potentials (pip)

```bash
# TorchANI
pip install torch torchani

# MACE-OFF (optional)
pip install mace-torch e3nn==0.4.4

# AIMNet2 (optional)
pip install aimnet2
```

### Quick install (all pip extras at once)

```bash
pip install -e ".[full]"
```

> **Note:** The `[full]` extra installs only pip-available packages (`torchani`, `mace-torch`, `parmed`, `ase`). You still need conda for `openmm`, `openff-toolkit`, `ambertools`, `rdkit`, and `xtb-python`.

### Google Colab

On Colab, everything is handled automatically via condacolab. See the provided notebooks.

## Quick Start

```python
from parametrizani import (
    ConformerGenerator, TopologyGenerator,
    calculate_reference_energies, optimize_dihedral, get_dihedral_atom_types,
)

# 1. Generate a 3D conformer and dihedral scan
gen = ConformerGenerator('CC(=O)OC', 'smiles', './work')
conf = gen.run()
scan = gen.generate_dihedral_conformers([0, 1, 2, 3], step=15)

# 2. Detect atom types automatically from MOL2 (via antechamber)
topo = TopologyGenerator('./work', force_field='gaff2')
amber_files = topo.generate_amber(conf['mol_file'])
atom_types = get_dihedral_atom_types(amber_files['mol2'], [0, 1, 2, 3])

# 3. Calculate reference energies
ref = calculate_reference_energies(scan['conformers'], scan['angles'], method='torchani')

# 4. Optimize dihedral parameters (atom_types detected automatically)
opt = optimize_dihedral(ref['angles'], ref['energies_relative'], atom_types=atom_types)
print(f"RMSE: {opt['rmse']:.4f} kcal/mol")
print(f"Atom types: {atom_types}")  # e.g. ['c3', 'c', 'o', 'c3']
```

### Full Workflow with Classes

```python
from parametrizani import (
    ConformerGenerator,
    ReferenceEnergyCalculator,
    EnergyMinimizer,
    DihedralOptimizer,
    ParameterValidator,
    TopologyGenerator,
    get_dihedral_atom_types,
)

# 1. Generate conformer
gen = ConformerGenerator('COc3ccc2c(=O)cc(c1ccccc1)oc2c3', 'smiles', './work')
conf = gen.run()
scan = gen.generate_dihedral_conformers([8, 9, 10, 15], step=30)

# 2. Generate topology & detect atom types automatically
topo = TopologyGenerator('./work', force_field='gaff2')
amber_files = topo.generate_amber(conf['mol_file'])
atom_types = get_dihedral_atom_types(amber_files['mol2'], [8, 9, 10, 15])
print(f"Detected atom types: {atom_types}")  # e.g. ['ca', 'ca', 'os', 'ca']

# 3. Calculate reference energies with TorchANI
calc = ReferenceEnergyCalculator('torchani', './work')
ref = calc.scan_dihedral(scan['conformers'], scan['angles'])

# 4. Optimize dihedral parameters
opt = DihedralOptimizer(max_terms=4, work_dir='./work')
result = opt.run_optimization(
    ref['angles'], ref['energies_relative'],
    atom_types=atom_types  # Automatically detected!
)

# 5. Validate
val = ParameterValidator('./work')
validation = val.validate_parameters(
    ref['angles'], ref['energies_relative'], result['best_fit']
)
print(f"Quality: {validation['quality']} (RMSE: {validation['rmse']:.4f} kcal/mol)")

# 6. Generate topology files
files = topo.generate_all(conf['mol_file'], result['output_file'])
```

## Modules

| Module | Class | Description |
|--------|-------|-------------|
| `conformer_gen` | `ConformerGenerator` | Generate 3D conformers from SMILES/PDB/MOL |
| `reference_energy` | `ReferenceEnergyCalculator` | ML potential energy calculations |
| `energy_minimization` | `EnergyMinimizer` | OpenMM minimization with restraints |
| `dihedral_optimizer` | `DihedralOptimizer` | Least-squares dihedral fitting |
| `validation` | `ParameterValidator` | Quality metrics and visualization |
| `topology_gen` | `TopologyGenerator` | AMBER/GROMACS/OpenMM topology generation |
| `utils` | `get_dihedral_atom_types` | Auto-detect atom types from MOL2 |

## Supported ML Methods

| Method | Package | Level of Theory |
|--------|---------|------------------|
| ANI-2x | TorchANI | ωB97X/6-31G* |
| ANI-1x | TorchANI | ωB97X/6-31G* |
| ANI-1ccx | TorchANI | CCSD(T)/CBS |
| ANI-2xr | TorchANI2 | B97-3c |
| ANI-2dr | TorchANI2 | B97-3c |
| ANI-2xr-Snn | TorchANI2 | B97-3c |
| ANI-mbis | TorchANI2 | B97-3c |
| MACE-OFF | MACE | DFT (SPICE) |
| GFN2-xTB | xtb-python | GFN2-xTB |
| AIMNet2 | AIMNet2 | ωB97M-D3 |

### How to select a model

The ML model is selected via the `method` parameter in `ReferenceEnergyCalculator`:

```python
from parametrizani import ReferenceEnergyCalculator

# Use any of the supported method strings:
calc = ReferenceEnergyCalculator('ani2xr', './work')  # TorchANI2 ANI-2xr
```

**Available `method` values:**

| `method` string | Model | Package required |
|-----------------|-------|------------------|
| `torchani` or `ani2x` | ANI-2x (default) | `torchani` |
| `ani1x` | ANI-1x | `torchani` |
| `ani1ccx` | ANI-1ccx | `torchani` |
| `ani2xr` | ANI-2xr | `torchani2` |
| `ani2dr` | ANI-2dr | `torchani2` |
| `ani2xr_snn` | ANI-2xr-Snn | `torchani2` |
| `ani_mbis` | ANI-mbis | `torchani2` |
| `mace` | MACE-OFF (medium) | `mace-torch` |
| `xtb` or `gfn2xtb` | GFN2-xTB | `xtb-python` (conda) |
| `aimnet2` | AIMNet2 | `aimnet2` |

In the Colab notebooks, select the model from the dropdown widget in the "Reference Energy Calculation" cell.

### How to select the force field (GAFF2 vs OpenFF)

The force field choice affects two steps: **topology generation** and **MM minimization**.

**GAFF2 workflow** (uses antechamber + tleap):

```python
from parametrizani import TopologyGenerator, EnergyMinimizer

# Topology with GAFF2
topo = TopologyGenerator('./work', force_field='gaff2')
amber_files = topo.generate_amber(conf['mol_file'], charge_method='am1bcc')

# MM minimization with GAFF2
minimizer = EnergyMinimizer('gaff2', './work')
mm = minimizer.minimize_scan(
    amber_files['prmtop'], amber_files['inpcrd'],
    scan['pdb_files'], dihedral_indices,
    angles=scan['angles'], zero_dihedral=True
)
```

**OpenFF workflow** (uses SMIRNOFF force fields):

```python
from parametrizani import EnergyMinimizer

# MM minimization with OpenFF (no topology step needed)
minimizer = EnergyMinimizer('openff-2.0.0', './work')
mm = minimizer.minimize_scan_openff(
    smiles,                  # SMILES string for the molecule
    scan['pdb_files'], dihedral_indices,
    angles=scan['angles'], zero_dihedral=True
)
```

**Available force field values for `EnergyMinimizer`:**

| Value | Force Field | Notes |
|-------|-------------|-------|
| `gaff2` | GAFF2 | Requires AmberTools (antechamber, tleap) |
| `openff-2.0.0` | OpenFF 2.0.0 (Sage) | Recommended OpenFF version |
| `openff-1.3.1` | OpenFF 1.3.1 | Legacy |
| `openff-1.2.0` | OpenFF 1.2.0 | Legacy |
| `smirnoff99Frosst-1.1.0` | SMIRNOFF99Frosst | Original SMIRNOFF |

> **Key difference:** GAFF2 requires generating topology files first (`TopologyGenerator.generate_amber()`), while OpenFF works directly from the SMILES string — no separate topology step needed for minimization.

In the Colab notebooks, use **Notebook A** for GAFF2 or **Notebook B** for OpenFF.

### How to select the charge method

The charge method is set via the `charge_method` parameter in `TopologyGenerator.generate_amber()`:

```python
from parametrizani import TopologyGenerator

topo = TopologyGenerator('./work', force_field='gaff2')

# AM1-BCC charges (default, fast)
amber_files = topo.generate_amber(conf['mol_file'], charge_method='am1bcc')

# RESP charges (more accurate, requires Psi4 or external calculation)
amber_files = topo.generate_amber(conf['mol_file'], charge_method='resp')
```

**Available `charge_method` values:**

| Value | Method | Description |
|-------|--------|-------------|
| `am1bcc` | AM1-BCC | Fast semi-empirical charges (default). Uses antechamber. |
| `resp` | RESP | Restrained Electrostatic Potential charges. More accurate, slower. Requires Psi4 or pre-computed charges. |
| `gasteiger` | Gasteiger | Fastest, least accurate. Useful for quick tests. |

> **Recommendation:** Use `am1bcc` for most applications. Use `resp` when higher accuracy is needed (e.g., for charged molecules or hydrogen bonding). See **Notebook C** (TorchANI+Psi4) for the RESP workflow with Psi4.


## Workflow

```
SMILES/PDB/MOL
     ↓
1. Conformer Generation (RDKit + MMFF94)
     ↓
2. Dihedral Scan (constrained optimization)
     ↓
3. Topology Generation & Atom Type Detection (antechamber)
     ↓
4. Reference Energy Calculation (TorchANI/MACE/xTB/AIMNet2)
     ↓
5. MM Energy Minimization (OpenMM, optional)
     ↓
6. Dihedral Optimization (Least-squares fitting, Rotational Profiler)
     ↓
7. Validation (RMSE, MAE, R², quality rating)
     ↓
8. Topology Generation (AMBER/GROMACS/OpenMM)
```

## Quality Ratings

| Rating | RMSE (kcal/mol) | Interpretation |
|--------|-----------------|----------------|
| Excellent | ≤ 0.25 | Near-QM accuracy |
| Good | ≤ 0.50 | Suitable for most applications |
| Acceptable | ≤ 1.00 | Usable with caution |
| Poor | > 1.00 | Consider different parameters |

## Bugs

- If you encounter any bugs, please report the issue to https://github.com/pablo-arantes/ParametrizANI/issues

## Acknowledgments

- ParametrizANI by **Pablo R. Arantes** ([@pablitoarantes](https://twitter.com/pablitoarantes)), **Souvik Sinha** and **Giulia Palermo**
- We would like to thank the OpenMM team for developing an excellent and open source engine.
- We would like to thank the [Psi4](https://psicode.org/) team for developing an excellent and open source suite of ab initio quantum chemistry.
- We would like to thank the [Roitberg](https://roitberg.chem.ufl.edu/) team for developing the fantastic [TorchANI](https://github.com/aiqm/torchani).
- We would like to thank the [Xavier Barril](http://www.ub.edu/bl/) team for their protocol on dihedrals parametrization and for the genetic algorithm script.
- We would like to thank [iwatobipen](https://twitter.com/iwatobipen) for his fantastic [blog](https://iwatobipen.wordpress.com/) on chemoinformatics.
- Also, credit to [David Koes](https://github.com/dkoes) for his awesome [py3Dmol](https://3dmol.csb.pitt.edu/) plugin.
- Finally, we would like to thank [Making it rain](https://github.com/pablo-arantes/making-it-rain) team, **Pablo R. Arantes** ([@pablitoarantes](https://twitter.com/pablitoarantes)), **Marcelo D. Polêto** ([@mdpoleto](https://twitter.com/mdpoleto)), **Conrado Pedebos** ([@ConradoPedebos](https://twitter.com/ConradoPedebos)) and **Rodrigo Ligabue-Braun** ([@ligabue_braun](https://twitter.com/ligabue_braun)), for their amazing work.

## How should I reference this work?

- For **ParametrizANI**, please cite:
  Arantes et al. "ParametrizANI: Fast and Accessible Dihedral Parametrization for Small Molecules."
  *Journal of Chemical Information and Modeling* (2025) doi: [10.1021/acs.jcim.5c01957](http://pubs.acs.org/doi/abs/10.1021/acs.jcim.5c01957)

- For **TorchANI**, please cite:
  Gao et al. "TorchANI: A Free and Open Source PyTorch-Based Deep Learning Implementation of the ANI Neural Network Potentials."
  *Journal of Chemical Information and Modeling* (2020) doi: [10.1021/acs.jcim.0c00451](https://doi.org/10.1021/acs.jcim.0c00451)

- For **OpenMM**, please also cite:
  Eastman et al. "OpenMM 7: Rapid development of high performance algorithms for molecular dynamics."
  *PLOS Computational Biology* (2017) doi: [10.1371/journal.pcbi.1005659](https://doi.org/10.1371/journal.pcbi.1005659)

- For **Molecular Dynamics Notebook**, please also cite:
  Arantes et al. "Making it rain: cloud-based molecular simulations for everyone."
  *Journal of Chemical Information and Modeling* (2021) doi: [10.1021/acs.jcim.1c00998](https://doi.org/10.1021/acs.jcim.1c00998)

## License

MIT License - see [LICENSE](LICENSE) for details.
