# ParametrizANI

**Fast and Accessible Dihedral Parametrization for Small Molecules**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ParametrizANI is a production-ready Python package for dihedral parameter optimization using neural network potentials and molecular mechanics. It provides an end-to-end pipeline from SMILES strings to force field parameters compatible with AMBER, GROMACS, and OpenMM.

## Citation

> Arantes et al. "ParametrizANI: Fast and Accessible Dihedral Parametrization for Small Molecules."
> *Journal of Chemical Information and Modeling* (2025) doi: [10.1021/acs.jcim.5c01957](http://pubs.acs.org/doi/abs/10.1021/acs.jcim.5c01957)

## Installation

```bash
cd parametrizani
pip install -e .
```

### Full Installation (all ML methods + MD engines)

```bash
pip install -e ".[full]"
```

### Conda dependencies (for xTB and AmberTools)

```bash
conda install -c conda-forge xtb-python ambertools
```

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
| ANI-2x | TorchANI | \u03c9B97X/6-31G* |
| ANI-1x | TorchANI | \u03c9B97X/6-31G* |
| ANI-1ccx | TorchANI | CCSD(T)/CBS |
| ANI-2xr | TorchANI2 | B97-3c |
| ANI-2dr | TorchANI2 | B97-3c |
| MACE-OFF | MACE | DFT (SPICE) |
| GFN2-xTB | xtb-python | GFN2-xTB |
| AIMNet2 | AIMNet2 | \u03c9B97M-D3 |

## Workflow

```
SMILES/PDB/MOL
     \u2193
1. Conformer Generation (RDKit + MMFF94)
     \u2193
2. Dihedral Scan (constrained optimization)
     \u2193
3. Topology Generation & Atom Type Detection (antechamber)
     \u2193
4. Reference Energy Calculation (TorchANI/MACE/xTB/AIMNet2)
     \u2193
5. MM Energy Minimization (OpenMM, optional)
     \u2193
6. Dihedral Optimization (Least-squares fitting, Rotational Profiler)
     \u2193
7. Validation (RMSE, MAE, R\u00b2, quality rating)
     \u2193
8. Topology Generation (AMBER/GROMACS/OpenMM)
```

## Quality Ratings

| Rating | RMSE (kcal/mol) | Interpretation |
|--------|-----------------|----------------|
| Excellent | \u2264 0.25 | Near-QM accuracy |
| Good | \u2264 0.50 | Suitable for most applications |
| Acceptable | \u2264 1.00 | Usable with caution |
| Poor | > 1.00 | Consider different parameters |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Authors

- **Pablo R. Arantes** ([@pablitoarantes](https://twitter.com/pablitoarantes))
- **Souvik Sinha**
- **Giulia Palermo**
