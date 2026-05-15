"""
ParametrizANI - Fast and Accessible Dihedral Parametrization for Small Molecules
=================================================================================

A production-ready Python package for dihedral parameter optimization using
neural network potentials (TorchANI, AIMNet2, MACE-OFF, GFN2-xTB) and
molecular mechanics (AMBER, OpenFF).

Main Classes
------------
ConformerGenerator : Generate 3D conformers from SMILES/PDB/MOL
ReferenceEnergyCalculator : Calculate energies with ML potentials
EnergyMinimizer : OpenMM energy minimization with restraints
DihedralOptimizer : Optimize dihedral parameters via least-squares fitting
ParameterValidator : Validate optimized parameters with metrics & plots
TopologyGenerator : Generate AMBER/GROMACS/OpenMM topology files

References
----------
Arantes et al. "ParametrizANI: Fast and Accessible Dihedral Parametrization
for Small Molecules." J. Chem. Inf. Model. (2025) doi: 10.1021/acs.jcim.5c01957
"""

__version__ = "1.0.0"
__author__ = "Pablo R. Arantes, Souvik Sinha, Giulia Palermo"
__license__ = "MIT"

# Core classes
from .conformer_gen import ConformerGenerator
from .reference_energy import ReferenceEnergyCalculator
from .energy_minimization import EnergyMinimizer
from .dihedral_optimizer import DihedralOptimizer
from .validation import ParameterValidator
from .topology_gen import TopologyGenerator

# Utility functions
from .utils import (
    read_energy_file,
    write_energy_file,
    relative_energies,
    extract_atom_info_from_pdb,
    get_atom_types_from_mol2,
    get_dihedral_atom_types,
)


def generate_conformer(molecule_input, input_type='smiles', work_dir='./work', optimize=True):
    """Quick conformer generation from SMILES/PDB/MOL."""
    gen = ConformerGenerator(molecule_input, input_type, work_dir)
    return gen.run(optimize=optimize)


def calculate_reference_energies(conformer_files, angles=None, method='torchani',
                                  work_dir='./work', optimize=True, device='cpu',
                                  dihedral_indices=None):
    """Quick reference energy calculation for a dihedral scan.
    
    Parameters
    ----------
    dihedral_indices : list of int, optional
        Four atom indices defining the dihedral to constrain during optimization.
    """
    calc = ReferenceEnergyCalculator(method, work_dir, device)
    return calc.scan_dihedral(conformer_files, angles, optimize=optimize,
                              dihedral_indices=dihedral_indices)


def optimize_dihedral(ref_angles, ref_energies, atom_types=None,
                      mm_energies=None, max_terms=4, work_dir='./work'):
    """Quick dihedral parameter optimization."""
    opt = DihedralOptimizer(max_terms=max_terms, work_dir=work_dir)
    return opt.run_optimization(ref_angles, ref_energies, mm_energies, atom_types)


def validate_parameters(angles, ref_energies, fitted_energies, work_dir='./work'):
    """Quick parameter validation."""
    val = ParameterValidator(work_dir)
    return val.validate_parameters(angles, ref_energies, fitted_energies)
