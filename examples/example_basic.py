"""ParametrizANI - Basic Example

Demonstrates dihedral optimization using pre-computed energy files.
"""
from parametrizani import optimize_dihedral, validate_parameters, read_energy_file

def main():
    # Read pre-computed energy profiles
    ref_angles, ref_energies = read_energy_file('examples/qm.dat')
    mm_angles, mm_energies = read_energy_file('examples/mm.dat')

    # Optimize dihedral parameters
    result = optimize_dihedral(
        ref_angles, ref_energies,
        atom_types=['ca', 'ca', 'ca', 'ca'],
        mm_energies=mm_energies,
        max_terms=6,
        work_dir='./output'
    )

    print(f"Best RMSE ({len(result['best_parameters'])} terms): {result['rmse']:.4f} kcal/mol")
    print(f"\nFRCMOD Parameters:")
    print(result['frcmod_params'])

    # Validate
    val = validate_parameters(ref_angles, ref_energies, result['best_fit'], './output')
    print(f"\nQuality: {val['quality']} (R²: {val['r_squared']:.4f})")

if __name__ == '__main__':
    main()
