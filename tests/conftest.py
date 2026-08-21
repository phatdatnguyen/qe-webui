"""Shared fixtures.

The tests never touch Quantum Espresso binaries, the network, or ./data — every
case runs against a temporary working directory and a fake pseudopotential root,
so `pytest` is safe to run on a machine without QE installed.
"""
import os
import sys
import warnings

import pytest

# The app modules are top-level files in the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pymatgen/nglview are chatty about deprecations at import time.
warnings.filterwarnings("ignore")


# A minimal rocksalt NaCl cell — small enough that PWInput/HighSymmKpath stay fast.
NACL_CIF = """data_NaCl
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   5.64056
_cell_length_b   5.64056
_cell_length_c   5.64056
_cell_angle_alpha   90.0
_cell_angle_beta    90.0
_cell_angle_gamma   90.0
_symmetry_Int_Tables_number   1
_chemical_formula_structural   NaCl
_chemical_formula_sum   'Na4 Cl4'
loop_
 _symmetry_equiv_pos_as_xyz
  'x, y, z'
loop_
 _atom_site_label
 _atom_site_type_symbol
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
  Na0  Na  0.0  0.0  0.0
  Na1  Na  0.0  0.5  0.5
  Na2  Na  0.5  0.0  0.5
  Na3  Na  0.5  0.5  0.0
  Cl0  Cl  0.5  0.5  0.5
  Cl1  Cl  0.5  0.0  0.0
  Cl2  Cl  0.0  0.5  0.0
  Cl3  Cl  0.0  0.0  0.5
"""


@pytest.fixture
def working_dir(tmp_path):
    """An empty working directory, as the app would create under ./data/<name>."""
    path = tmp_path / "wd"
    path.mkdir()
    return str(path)


@pytest.fixture
def structure_file(working_dir):
    """A NaCl .cif inside the working directory; returns its bare file name.

    This is the 8-site *conventional* cell — the common case for a downloaded CIF,
    and the one a labelled bands k-path may not be written for.
    """
    with open(os.path.join(working_dir, "NaCl.cif"), "w") as fh:
        fh.write(NACL_CIF)
    return "NaCl.cif"


@pytest.fixture
def structure(structure_file, working_dir):
    from pymatgen.core import Structure
    return Structure.from_file(os.path.join(working_dir, structure_file))


@pytest.fixture
def primitive_structure(structure):
    """The 2-site standard primitive cell of the same NaCl — bands-ready."""
    import calculation
    return calculation.primitive_standard_structure(structure)


@pytest.fixture
def primitive_structure_file(working_dir, primitive_structure):
    """The primitive cell written into the working directory as a .cif."""
    name = "NaCl_primitive.cif"
    primitive_structure.to(filename=os.path.join(working_dir, name))
    return name


@pytest.fixture
def pseudo_root(tmp_path, monkeypatch):
    """A fake $ESPRESSO_PSEUDO tree, patched into calculation.PSEUDO_ROOT.

    Layout (deliberately unsorted on disk):
        <root>/SSSP-lib-pbe-eff-v2/{Na,Cl}.upf
        <root>/SG14_ONCV/{Na_ONCV_PBE-1.2,Cl_ONCV_PBE-1.2}.upf
        <root>/empty-set/            (no .UPF files)
    """
    import calculation

    root = tmp_path / "pseudo"
    (root / "SSSP-lib-pbe-eff-v2").mkdir(parents=True)
    (root / "SG14_ONCV").mkdir()
    (root / "empty-set").mkdir()
    for name in ("Na.upf", "Cl.upf"):
        (root / "SSSP-lib-pbe-eff-v2" / name).write_text("fake UPF\n")
    for name in ("Na_ONCV_PBE-1.2.upf", "Cl_ONCV_PBE-1.2.upf"):
        (root / "SG14_ONCV" / name).write_text("fake UPF\n")

    monkeypatch.setattr(calculation, "PSEUDO_ROOT", str(root))
    return str(root)


@pytest.fixture
def pw_input_args(working_dir, structure, pseudo_root):
    """Keyword arguments for a plain scf generate_pw_input_file() call."""
    return dict(
        working_directory_path=working_dir,
        calc_type="scf",
        structure=structure,
        pseudo_dir="SSSP-lib-pbe-eff-v2",
        ecutwfc=50,
        ecutrho=400,
        kgrid=(4, 4, 4),
        input_dft=None,
        is_metagga=False,
        prefix="pwscf",
        input_file_name="scf.in",
    )
