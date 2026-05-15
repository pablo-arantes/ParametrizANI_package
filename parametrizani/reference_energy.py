"""
ParametrizANI - Reference Energy Calculator
=============================================

Calculate reference energies using ML potentials:
- TorchANI (ANI-2x, ANI-1x, ANI-1ccx)
- TorchANI2 (ANI-2xr, ANI-2dr, ANI-2xr-Snn, ANI-mbis)
- AIMNet2
- MACE-OFF
- GFN2-xTB
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from .utils import create_work_dir, relative_energies, write_energy_file, HARTREE_TO_KCAL

logger = logging.getLogger(__name__)


class ReferenceEnergyCalculator:
    """
    Calculate reference energies using machine learning potentials.
    
    Supported methods:
    - 'torchani' or 'ani2x': TorchANI with ANI-2x model
    - 'ani1x': TorchANI with ANI-1x model  
    - 'ani1ccx': TorchANI with ANI-1ccx model
    - 'ani2xr', 'ani2dr', 'ani2xr_snn', 'ani_mbis': TorchANI2 models
    - 'aimnet2': AIMNet2 neural network potential
    - 'mace': MACE-OFF potential
    - 'xtb' or 'gfn2xtb': GFN2-xTB semi-empirical method
    
    Parameters
    ----------
    method : str
        Calculation method to use.
    work_dir : str
        Working directory for output files.
    device : str
        Device for computation ('cpu' or 'cuda'). Default 'cpu'.
    """
    
    SUPPORTED_METHODS = [
        'torchani', 'ani2x', 'ani1x', 'ani1ccx',
        'ani2xr', 'ani2dr', 'ani2xr_snn', 'ani_mbis',
        'aimnet2', 'mace', 'xtb', 'gfn2xtb'
    ]
    
    def __init__(self, method: str = 'torchani', work_dir: str = './work',
                 device: str = 'cpu'):
        self.method = method.lower().replace('-', '_').replace(' ', '_')
        self.work_dir = create_work_dir(work_dir)
        self.device = device
        self.calculator = None
        
        if self.method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method: {method}. "
                f"Supported: {self.SUPPORTED_METHODS}"
            )
        
        self._setup_calculator()
    
    def _setup_calculator(self):
        """Initialize the ML potential calculator."""
        if self.method in ['torchani', 'ani2x']:
            self._setup_torchani('ANI2x')
        elif self.method == 'ani1x':
            self._setup_torchani('ANI1x')
        elif self.method == 'ani1ccx':
            self._setup_torchani('ANI1ccx')
        elif self.method in ['ani2xr', 'ani2dr', 'ani2xr_snn', 'ani_mbis']:
            self._setup_torchani2()
        elif self.method == 'aimnet2':
            self._setup_aimnet2()
        elif self.method == 'mace':
            self._setup_mace()
        elif self.method in ['xtb', 'gfn2xtb']:
            self._setup_xtb()
    
    def _setup_torchani(self, model_name: str):
        """Set up TorchANI calculator."""
        import torch
        import torchani
        self.torch = torch
        self.torchani = torchani
        if model_name == 'ANI2x':
            self.model = torchani.models.ANI2x(periodic_table_index=True).to(self.device)
        elif model_name == 'ANI1x':
            self.model = torchani.models.ANI1x(periodic_table_index=True).to(self.device)
        elif model_name == 'ANI1ccx':
            self.model = torchani.models.ANI1ccx(periodic_table_index=True).to(self.device)
        logger.info(f"TorchANI {model_name} calculator initialized on {self.device}")
    
    def _setup_torchani2(self):
        """Set up TorchANI2 calculator."""
        try:
            import torchani2
            import torch
            self.torch = torch
            model_map = {
                'ani2xr': 'ANI-2xr', 'ani2dr': 'ANI-2dr',
                'ani2xr_snn': 'ANI-2xr-Snn', 'ani_mbis': 'ANI-mbis',
            }
            model_name = model_map.get(self.method, 'ANI-2xr')
            self.model = torchani2.load(model_name).to(self.device)
            logger.info(f"TorchANI2 {model_name} calculator initialized")
        except ImportError:
            raise ImportError("TorchANI2 not installed. Install with: pip install torchani2")
    
    def _setup_aimnet2(self):
        """Set up AIMNet2 calculator."""
        import torch
        self.torch = torch
        
        # Try multiple ways to load AIMNet2
        self.aimnet2_model = None
        
        # Method 1: Try the aimnet2 pip package
        try:
            import aimnet2
            self.aimnet2_model = torch.jit.load(
                aimnet2.download_model('aimnet2_wb97m-d3_ens'),
                map_location=self.device
            )
            logger.info("AIMNet2 loaded from pip package")
            return
        except (ImportError, AttributeError, Exception):
            pass
        
        # Method 2: Try loading from cloned repo path
        model_paths = [
            'AIMNet2/models/aimnet2_wb97m-d3_ens.jpt',
            '/content/AIMNet2/models/aimnet2_wb97m-d3_ens.jpt',
            os.path.join(self.work_dir, 'AIMNet2/models/aimnet2_wb97m-d3_ens.jpt'),
        ]
        for path in model_paths:
            if os.path.exists(path):
                self.aimnet2_model = torch.jit.load(path, map_location=self.device)
                logger.info(f"AIMNet2 loaded from {path}")
                return
        
        # Method 3: Clone the repo
        import subprocess
        clone_dir = os.path.join(self.work_dir, 'AIMNet2')
        if not os.path.exists(clone_dir):
            logger.info("Cloning AIMNet2 repository...")
            subprocess.run(
                f"git clone https://github.com/isayevlab/AIMNet2.git {clone_dir}",
                shell=True, capture_output=True
            )
        
        model_file = os.path.join(clone_dir, 'models', 'aimnet2_wb97m-d3_ens.jpt')
        if os.path.exists(model_file):
            self.aimnet2_model = torch.jit.load(model_file, map_location=self.device)
            logger.info(f"AIMNet2 loaded from cloned repo")
        else:
            raise RuntimeError(
                "Could not load AIMNet2 model. Try:\n"
                "  git clone https://github.com/isayevlab/AIMNet2.git\n"
                "  # or: pip install aimnet2"
            )
    
    def _setup_mace(self):
        """Set up MACE-OFF calculator."""
        try:
            # Fix potential circular import with torchvision on Colab
            import torch
            try:
                import torchvision
            except (ImportError, AttributeError):
                pass
            
            from mace.calculators import mace_off
            self.mace_calc = mace_off(model="medium", device=self.device)
            logger.info("MACE-OFF calculator initialized")
        except ImportError:
            raise ImportError(
                "MACE not installed. Install with:\n"
                "  pip install mace-torch\n"
                "If you get a torchvision error, try:\n"
                "  pip install --upgrade torchvision\n"
                "  pip install mace-torch"
            )
        except AttributeError as e:
            if 'torchvision' in str(e):
                raise ImportError(
                    "Circular import with torchvision. Fix by running:\n"
                    "  pip install --upgrade torchvision\n"
                    "Then restart the runtime and try again."
                )
            raise
    
    def _setup_xtb(self):
        """Set up GFN2-xTB calculator."""
        try:
            from xtb.ase.calculator import XTB
            self.xtb_calc = XTB(method="GFN2-xTB")
            logger.info("GFN2-xTB calculator initialized")
        except ImportError:
            raise ImportError(
                "xtb-python not installed. Install: conda install -c conda-forge xtb-python"
            )
    
    def calculate_energy(self, mol_file: str, optimize: bool = True,
                        fmax: float = 0.05, steps: int = 200) -> Dict[str, Any]:
        """
        Calculate energy of a single conformer.
        
        Parameters
        ----------
        mol_file : str
            Path to MOL/PDB file.
        optimize : bool
            Whether to optimize geometry. Default True.
        fmax : float
            Force convergence criterion for optimization.
        steps : int
            Maximum optimization steps.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with 'energy' (Hartree), 'energy_kcal' (kcal/mol), 'rho'.
        """
        from ase.io import read as ase_read
        atoms = ase_read(mol_file)
        
        if self.method in ['torchani', 'ani2x', 'ani1x', 'ani1ccx']:
            return self._calc_torchani(atoms, optimize, fmax, steps)
        elif self.method in ['ani2xr', 'ani2dr', 'ani2xr_snn', 'ani_mbis']:
            return self._calc_torchani2(atoms, optimize, fmax, steps)
        elif self.method == 'mace':
            return self._calc_mace(atoms, optimize, fmax, steps)
        elif self.method in ['xtb', 'gfn2xtb']:
            return self._calc_xtb(atoms, optimize, fmax, steps)
        elif self.method == 'aimnet2':
            return self._calc_aimnet2(atoms, optimize, fmax, steps)
    
    def _calc_torchani(self, atoms, optimize, fmax, steps) -> Dict[str, Any]:
        """Calculate energy using TorchANI."""
        import torch
        
        # Optimize geometry using ASE + TorchANI calculator
        if optimize:
            from ase.optimize import BFGS
            # Use model.ase() - compatible with torchani >= 2.2
            atoms.calc = self.model.ase()
            opt = BFGS(atoms, logfile=None)
            opt.run(fmax=fmax, steps=steps)
        
        # Calculate energy with the model directly
        species = torch.tensor([atoms.get_atomic_numbers()], dtype=torch.long, device=self.device)
        coordinates = torch.tensor(
            [atoms.get_positions()], dtype=torch.float32,
            device=self.device, requires_grad=True
        )
        energy = self.model((species, coordinates)).energies
        energy_hartree = energy.item()
        
        # Calculate ensemble disagreement (rho) for reliability estimate
        rho = 0.0
        try:
            member_energies = []
            # Access individual ensemble members via the model's internals
            aev_result = self.model.aev_computer((species, coordinates))
            for i, nn in enumerate(self.model.neural_networks):
                member_species_energies = nn(aev_result)
                member_shifted = self.model.energy_shifter(member_species_energies)
                member_energies.append(member_shifted.energies.item())
            if len(member_energies) > 1:
                rho = float(np.std(member_energies)) * HARTREE_TO_KCAL
        except Exception:
            # If ensemble access fails (API changed), skip rho calculation
            pass
        
        return {
            'energy': energy_hartree,
            'energy_kcal': energy_hartree * HARTREE_TO_KCAL,
            'rho': rho,
            'positions': atoms.get_positions(),
        }
    
    def _calc_torchani2(self, atoms, optimize, fmax, steps) -> Dict[str, Any]:
        """Calculate energy using TorchANI2."""
        import torch
        species = torch.tensor([atoms.get_atomic_numbers()], dtype=torch.long, device=self.device)
        coordinates = torch.tensor(
            [atoms.get_positions()], dtype=torch.float32,
            device=self.device, requires_grad=True
        )
        if optimize:
            from ase.optimize import BFGS
            atoms.calc = self.model.ase()
            opt = BFGS(atoms, logfile=None)
            opt.run(fmax=fmax, steps=steps)
            coordinates = torch.tensor(
                [atoms.get_positions()], dtype=torch.float32,
                device=self.device, requires_grad=True
            )
        result = self.model(species, coordinates)
        energy_hartree = result.energies.item()
        return {'energy': energy_hartree, 'energy_kcal': energy_hartree * HARTREE_TO_KCAL, 'rho': 0.0, 'positions': atoms.get_positions()}
    
    def _calc_mace(self, atoms, optimize, fmax, steps) -> Dict[str, Any]:
        """Calculate energy using MACE-OFF."""
        atoms.calc = self.mace_calc
        if optimize:
            from ase.optimize import BFGS
            opt = BFGS(atoms, logfile=None)
            opt.run(fmax=fmax, steps=steps)
        energy_ev = atoms.get_potential_energy()
        energy_kcal = energy_ev * 23.0609  # eV to kcal/mol
        return {'energy': energy_ev, 'energy_kcal': energy_kcal, 'rho': 0.0, 'positions': atoms.get_positions()}
    
    def _calc_xtb(self, atoms, optimize, fmax, steps) -> Dict[str, Any]:
        """Calculate energy using GFN2-xTB."""
        from xtb.ase.calculator import XTB
        atoms.calc = XTB(method="GFN2-xTB")
        if optimize:
            from ase.optimize import BFGS
            opt = BFGS(atoms, logfile=None)
            opt.run(fmax=fmax, steps=steps)
        energy_ev = atoms.get_potential_energy()
        energy_kcal = energy_ev * 23.0609
        return {'energy': energy_ev, 'energy_kcal': energy_kcal, 'rho': 0.0, 'positions': atoms.get_positions()}
    
    def _calc_aimnet2(self, atoms, optimize, fmax, steps) -> Dict[str, Any]:
        """Calculate energy using AIMNet2."""
        import torch
        
        if optimize:
            # Use ASE interface for optimization
            from ase.optimize import BFGS
            from ase.calculators.calculator import Calculator, all_changes
            
            class AIMNet2ASECalc(Calculator):
                implemented_properties = ['energy', 'forces']
                def __init__(self, model, device):
                    Calculator.__init__(self)
                    self.model = model
                    self.device = device
                def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
                    Calculator.calculate(self, atoms, properties, system_changes)
                    species = torch.tensor(self.atoms.get_atomic_numbers(), dtype=torch.long, device=self.device)
                    coords = torch.tensor(
                        self.atoms.get_positions(), dtype=torch.float64,
                        device=self.device
                    ).unsqueeze(0).requires_grad_(True)
                    inp = {'coord': coords, 'numbers': species, 'charge': torch.tensor([0.0], device=self.device)}
                    res = self.model(inp)
                    self.results['energy'] = res['energy'].item() * HARTREE_TO_KCAL * 4.184 / 96.485  # Hartree to eV
                    if 'forces' in res:
                        self.results['forces'] = -res['forces'].squeeze(0).detach().cpu().numpy() * HARTREE_TO_KCAL * 4.184 / 96.485 / 0.529177
            
            atoms.calc = AIMNet2ASECalc(self.aimnet2_model, self.device)
            opt = BFGS(atoms, logfile=None)
            opt.run(fmax=fmax, steps=steps)
        
        # Final energy calculation
        species = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=self.device)
        coordinates = torch.tensor(
            atoms.get_positions(), dtype=torch.float64, device=self.device
        ).unsqueeze(0).requires_grad_(True)
        
        input_dict = {
            'coord': coordinates,
            'numbers': species,
            'charge': torch.tensor([0.0], device=self.device)
        }
        result = self.aimnet2_model(input_dict)
        energy_hartree = result['energy'].item()
        
        return {
            'energy': energy_hartree,
            'energy_kcal': energy_hartree * HARTREE_TO_KCAL,
            'rho': 0.0,
            'positions': atoms.get_positions(),
        }
    
    def scan_dihedral(
        self,
        conformer_files: List[str],
        angles: Optional[List[float]] = None,
        optimize: bool = True,
        fmax: float = 0.05,
        steps: int = 200
    ) -> Dict[str, Any]:
        """
        Calculate energies for a dihedral scan.
        
        Parameters
        ----------
        conformer_files : List[str]
            List of MOL/PDB files for each dihedral angle.
        angles : Optional[List[float]]
            Corresponding dihedral angles. If None, extracted from filenames.
        optimize : bool
            Whether to optimize each conformer.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with 'angles', 'energies_relative', 'rho_values', 'output_file'.
        """
        energies = []
        rho_values = []
        
        if angles is None:
            angles = []
            for f in conformer_files:
                basename = os.path.splitext(os.path.basename(f))[0]
                try:
                    angles.append(float(basename.split('_')[0]))
                except (ValueError, IndexError):
                    angles.append(float(basename))
        
        logger.info(f"Scanning {len(conformer_files)} conformers with {self.method}...")
        
        for i, mol_file in enumerate(conformer_files):
            try:
                result = self.calculate_energy(mol_file, optimize=optimize, fmax=fmax, steps=steps)
                energies.append(result['energy_kcal'])
                rho_values.append(result.get('rho', 0.0))
                if (i + 1) % 5 == 0:
                    logger.info(f"  Processed {i+1}/{len(conformer_files)} conformers")
            except Exception as e:
                logger.warning(f"Error processing {mol_file}: {e}")
                energies.append(np.nan)
                rho_values.append(np.nan)
        
        energies_arr = np.array(energies)
        energies_relative = relative_energies(energies_arr)
        
        max_rho = max(rho_values) if rho_values else 0.0
        if max_rho > 0.6:
            logger.warning(f"High RHO={max_rho:.3f} > 0.6 kcal/mol. Results may be unreliable.")
        
        output_file = os.path.join(self.work_dir, f'{self.method}_scan.dat')
        write_energy_file(output_file, angles, energies_relative.tolist())
        
        logger.info(f"Dihedral scan complete. Energy range: {energies_relative.max():.3f} kcal/mol")
        
        return {
            'angles': angles,
            'energies': energies,
            'energies_relative': energies_relative.tolist(),
            'rho_values': rho_values,
            'output_file': output_file,
            'method': self.method,
        }
