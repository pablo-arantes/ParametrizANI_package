"""ParametrizANI - Full Workflow

Complete pipeline from SMILES to topology files.
Requires: pip install -e ".[full]"
          conda install -c conda-forge ambertools
"""
from parametrizani import (
    ConformerGenerator,
    ReferenceEnergyCalculator,
    DihedralOptimizer,
    ParameterValidator,
    TopologyGenerator,
    get_dihedral_atom_types,
)

def main():
    smiles = 'C=CC(=O)OC'  # Vinyl acetate
    dihedral_indices = [0, 1, 2, 3]
    work_dir = './full_workflow_output'

    # 1. Generate conformer & scan
    gen = ConformerGenerator(smiles, 'smiles', work_dir)
    conf = gen.run()
    scan = gen.generate_dihedral_conformers(dihedral_indices, step=15)
    print(f"[1/6] Generated {len(scan['angles'])} conformers")

    # 2. Generate topology & detect atom types automatically
    topo = TopologyGenerator(work_dir, force_field='gaff2')
    amber_files = topo.generate_amber(conf['mol_file'], charge_method='am1bcc')
    atom_types = get_dihedral_atom_types(amber_files['mol2'], dihedral_indices)
    print(f"[2/6] Atom types detected: {atom_types}")

    # 3. Calculate reference energies
    calc = ReferenceEnergyCalculator('torchani', work_dir)
    ref = calc.scan_dihedral(scan['conformers'], scan['angles'])
    print(f"[3/6] Energy range: {max(ref['energies_relative']):.3f} kcal/mol")

    # 4. Optimize (atom_types automatically from MOL2)
    optimizer = DihedralOptimizer(max_terms=4, work_dir=work_dir)
    result = optimizer.run_optimization(
        ref['angles'], ref['energies_relative'], atom_types=atom_types
    )
    frcmod_file = optimizer.write_frcmod(result)
    print(f"[4/6] RMSE: {result['rmse']:.4f} kcal/mol")

    # 5. Validate
    validator = ParameterValidator(work_dir)
    val = validator.validate_parameters(ref['angles'], ref['energies_relative'], result['best_fit'])
    print(f"[5/6] Quality: {val['quality']} (R\u00b2: {val['r_squared']:.4f})")

    # 6. Generate final topology with optimized parameters
    try:
        files = topo.generate_all(conf['mol_file'], frcmod_file)
        print(f"[6/6] Topology files generated")
    except Exception as e:
        print(f"[6/6] Topology skipped: {e}")

    print(f"\nDone! Results in: {work_dir}")

if __name__ == '__main__':
    main()
