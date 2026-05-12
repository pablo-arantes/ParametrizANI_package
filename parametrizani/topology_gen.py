"""
ParametrizANI - Topology Generator
====================================

Generate topology files for MD simulations in AMBER, GROMACS, and OpenMM formats.
"""

import os
import logging
import subprocess
import shutil
import zipfile
from typing import Dict, Any, Optional, List

from .utils import create_work_dir

logger = logging.getLogger(__name__)


class TopologyGenerator:
    """
    Generate topology files for MD simulations.
    
    Supports AMBER (prmtop, inpcrd, frcmod, lib), GROMACS (top, gro, pdb),
    and OpenMM (xml, pdb) formats.
    
    Parameters
    ----------
    work_dir : str
        Working directory.
    force_field : str
        Force field to use. Default 'gaff2'.
    """
    
    def __init__(self, work_dir: str = './work', force_field: str = 'gaff2'):
        self.work_dir = create_work_dir(work_dir)
        self.force_field = force_field
        self.tleap_dir = os.path.join(self.work_dir, 'tleap_output')
        os.makedirs(self.tleap_dir, exist_ok=True)
    
    def generate_amber(self, mol_file: str, frcmod_file: Optional[str] = None,
                       charge_method: str = 'am1bcc', mol2_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate AMBER topology files using tLEaP."""
        if mol2_file is None:
            mol2_file = self._generate_mol2_with_charges(mol_file, charge_method)
        
        paths = {
            'prmtop': os.path.join(self.tleap_dir, 'SYS.prmtop'),
            'inpcrd': os.path.join(self.tleap_dir, 'SYS.crd'),
            'pdb': os.path.join(self.tleap_dir, 'SYS.pdb'),
            'lib': os.path.join(self.tleap_dir, 'lig.lib'),
            'mol2': mol2_file,
            'frcmod': frcmod_file or os.path.join(self.tleap_dir, 'ligand.frcmod'),
        }
        
        if frcmod_file is None:
            self._run_parmchk2(mol2_file, paths['frcmod'])
        else:
            shutil.copy(frcmod_file, paths['frcmod'])
        
        tleap_input = f"""source leaprc.protein.ff19SB
source leaprc.{self.force_field}
LIG = loadmol2 {paths['mol2']}
loadamberparams {paths['frcmod']}
saveoff LIG {paths['lib']}
saveamberparm LIG {paths['prmtop']} {paths['inpcrd']}
savepdb LIG {paths['pdb']}
quit"""
        
        tleap_file = os.path.join(self.tleap_dir, 'tleap.in')
        with open(tleap_file, 'w') as f:
            f.write(tleap_input)
        
        subprocess.run(f"tleap -f {tleap_file}", shell=True, capture_output=True, text=True)
        logger.info("AMBER topology files generated")
        return paths
    
    def generate_gromacs(self, prmtop_file: str, inpcrd_file: str) -> Dict[str, Any]:
        """Generate GROMACS topology files from AMBER topology."""
        import parmed
        gmx_dir = os.path.join(self.work_dir, 'gromacs')
        os.makedirs(gmx_dir, exist_ok=True)
        amber_struct = parmed.load_file(prmtop_file, inpcrd_file)
        paths = {
            'top': os.path.join(gmx_dir, 'system.top'),
            'gro': os.path.join(gmx_dir, 'system.gro'),
            'pdb': os.path.join(gmx_dir, 'system.pdb'),
        }
        amber_struct.save(paths['top'], overwrite=True)
        amber_struct.save(paths['gro'], overwrite=True)
        amber_struct.save(paths['pdb'], overwrite=True)
        logger.info("GROMACS topology files generated")
        return paths
    
    def generate_openmm(self, prmtop_file: str, inpcrd_file: str) -> Dict[str, Any]:
        """Generate OpenMM XML system file."""
        import openmm as mm
        from openmm.app import AmberPrmtopFile, AmberInpcrdFile, PDBFile
        from openmm import unit, XmlSerializer
        
        omm_dir = os.path.join(self.work_dir, 'openmm')
        os.makedirs(omm_dir, exist_ok=True)
        
        prmtop = AmberPrmtopFile(prmtop_file)
        inpcrd = AmberInpcrdFile(inpcrd_file)
        system = prmtop.createSystem(nonbondedCutoff=1*unit.nanometer, constraints=None)
        xml_content = XmlSerializer.serialize(system)
        
        paths = {
            'xml': os.path.join(omm_dir, 'system.xml'),
            'pdb': os.path.join(omm_dir, 'system.pdb'),
        }
        with open(paths['xml'], 'w') as f:
            f.write(xml_content)
        
        integrator = mm.LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
        simulation = mm.app.Simulation(prmtop.topology, system, integrator)
        simulation.context.setPositions(inpcrd.positions)
        with open(paths['pdb'], 'w') as f:
            PDBFile.writeFile(simulation.topology, inpcrd.positions, f)
        
        logger.info("OpenMM system files generated")
        return paths
    
    def generate_all(self, mol_file: str, frcmod_file=None, charge_method='am1bcc', formats=None):
        """Generate topology files in all requested formats."""
        if formats is None:
            formats = ['amber', 'gromacs', 'openmm']
        results = {}
        amber_result = self.generate_amber(mol_file, frcmod_file, charge_method)
        results['amber'] = amber_result
        if 'gromacs' in formats:
            results['gromacs'] = self.generate_gromacs(amber_result['prmtop'], amber_result['inpcrd'])
        if 'openmm' in formats:
            results['openmm'] = self.generate_openmm(amber_result['prmtop'], amber_result['inpcrd'])
        return results
    
    def package_results(self, results, output_name='topology_files'):
        """Package all generated files into a ZIP archive."""
        zip_path = os.path.join(self.work_dir, f'{output_name}.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for format_name, files in results.items():
                for file_type, file_path in files.items():
                    if file_path and os.path.exists(file_path):
                        arcname = f'{format_name}/{os.path.basename(file_path)}'
                        zf.write(file_path, arcname)
        logger.info(f"Results packaged: {zip_path}")
        return zip_path
    
    def update_frcmod(self, original_frcmod: str, optimized_params: str, output_file=None):
        """Update FRCMOD file with optimized dihedral parameters."""
        if output_file is None:
            output_file = os.path.join(self.tleap_dir, 'ligand_new.frcmod')
        shutil.copy(original_frcmod, output_file)
        with open(output_file, 'r') as f:
            content = f.read()
        if 'DIHE' in content:
            parts = content.split('DIHE')
            if len(parts) > 1:
                rest = parts[1]
                sections = ['IMPROPER', 'NONBON', '\n\n']
                next_pos = len(rest)
                for section in sections:
                    pos = rest.find(section)
                    if pos != -1 and pos < next_pos:
                        next_pos = pos
                content = parts[0] + optimized_params + "\n" + rest[next_pos:]
        else:
            content += "\n" + optimized_params + "\n"
        with open(output_file, 'w') as f:
            f.write(content)
        return output_file
    
    def _generate_mol2_with_charges(self, mol_file, charge_method):
        """Generate MOL2 file with charges using antechamber."""
        mol2_output = os.path.join(self.tleap_dir, 'ligand.mol2')
        ext = os.path.splitext(mol_file)[1].lower()
        input_format = {'.mol': 'mdl', '.pdb': 'pdb', '.mol2': 'mol2', '.sdf': 'sdf'}.get(ext, 'pdb')
        charge_flag = {'am1bcc': 'bcc', 'bcc': 'bcc', 'resp': 'resp', 'gasteiger': 'gas'}.get(charge_method.lower(), 'bcc')
        cmd = f"antechamber -i {mol_file} -fi {input_format} -o {mol2_output} -fo mol2 -c {charge_flag} -at {self.force_field} -pf y"
        subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if not os.path.exists(mol2_output):
            raise RuntimeError(f"antechamber failed to generate MOL2 file.")
        return mol2_output
    
    def _run_parmchk2(self, mol2_file, frcmod_output):
        """Run parmchk2 to generate missing parameters."""
        cmd = f"parmchk2 -i {mol2_file} -f mol2 -o {frcmod_output} -s {self.force_field}"
        subprocess.run(cmd, shell=True, capture_output=True, text=True)
