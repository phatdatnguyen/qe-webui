"""calculation.py — input generation, pseudopotential resolution, run plumbing.

Nothing here needs QE: the one place a process would be spawned (run_qe_stream)
is exercised against a fake subprocess.
"""
import os

import pytest

import calculation as C


# --------------------------------------------------------------------------- #
# Naming helpers
# --------------------------------------------------------------------------- #

class TestNames:
    @pytest.mark.parametrize("calc_type,expected", [
        ("scf", "scf.in"), ("relax", "relax.in"), ("vc-relax", "vc-relax.in"),
        ("nscf", "nscf.in"), ("bands", "bands.in"),
        ("dos", "dos.in"), ("projwfc", "projwfc.in"),
        ("bands.x", "bandsx.in"), ("pp.x", "pp.in"),
    ])
    def test_default_input_name(self, calc_type, expected):
        assert C.default_input_name(calc_type) == expected

    def test_post_type_names_are_valid_file_names(self):
        # 'bands.x'/'pp.x' would otherwise produce a confusing 'bands.x.in'.
        for calc_type in C.POST_TYPES:
            name = C.default_input_name(calc_type)
            assert validate_ok(name), name

    @pytest.mark.parametrize("input_name,expected", [
        ("scf.in", "scf.out"), ("bandsx.in", "bandsx.out"),
        ("no_extension", "no_extension.out"), ("", ""), (None, ""),
    ])
    def test_default_output_name(self, input_name, expected):
        assert C.default_output_name(input_name) == expected

    def test_on_select_input_file_suggests_matching_log(self):
        assert C.on_select_input_file("relax.in")["value"] == "relax.out"

    def test_every_calc_type_maps_to_an_executable(self):
        for calc_type in C.CALC_TYPES:
            assert C.CALC_EXECUTABLE[calc_type] in C.QE_EXECUTABLES

    def test_pw_types_all_run_pw_x(self):
        assert {C.CALC_EXECUTABLE[t] for t in C.PW_TYPES} == {"pw.x"}


def validate_ok(name):
    from utils import validate_name
    return validate_name(name, "input file name") is None


# --------------------------------------------------------------------------- #
# Functionals
# --------------------------------------------------------------------------- #

class TestResolveFunctional:
    def test_default_keeps_qe_default(self):
        assert C.resolve_functional(C.DEFAULT_FUNCTIONAL, "") == (None, False)

    def test_unknown_choice_falls_back_to_default(self):
        assert C.resolve_functional("not a choice", "") == (None, False)

    @pytest.mark.parametrize("label,value", [
        ("GGA — PBE", "pbe"), ("LDA — PZ", "pz"), ("hybrid — HSE06", "hse"),
    ])
    def test_plain_functionals(self, label, value):
        assert C.resolve_functional(label, "") == (value, False)

    @pytest.mark.parametrize("label", ["meta-GGA — SCAN", "meta-GGA — TPSS",
                                       "meta-GGA — M06-L"])
    def test_meta_gga_is_flagged(self, label):
        value, is_metagga = C.resolve_functional(label, "")
        assert value in C.META_GGA and is_metagga

    def test_custom_uses_the_free_text_box(self):
        assert C.resolve_functional("Custom (Libxc / input_dft)", " r2scan ") == \
            ("r2scan", True)   # 'scan' in the name marks it meta-GGA

    def test_custom_libxc_code_is_not_metagga_by_default(self):
        value, is_metagga = C.resolve_functional("Custom (Libxc / input_dft)",
                                                 "XC-000i-000i-101L-130L")
        assert value == "XC-000i-000i-101L-130L" and not is_metagga

    @pytest.mark.parametrize("custom", ["", "   ", None])
    def test_custom_without_text_is_an_error(self, custom):
        with pytest.raises(Exception, match="Custom input_dft"):
            C.resolve_functional("Custom (Libxc / input_dft)", custom)


# --------------------------------------------------------------------------- #
# Extra namelist settings
# --------------------------------------------------------------------------- #

class TestParseExtraSettings:
    @pytest.mark.parametrize("text", ["", "   ", None, "# only a comment\n! and another"])
    def test_nothing_to_parse(self, text):
        assert C.parse_extra_settings(text) == {}

    def test_groups_by_namelist_and_types_values(self):
        parsed = C.parse_extra_settings(
            "system.nbnd = 24\n"
            "electrons.conv_thr = 1e-8\n"
            "SYSTEM.nspin = 2\n"
            "control.tprnfor = .true.\n"
            "control.restart_mode = 'from_scratch'\n")
        assert parsed == {
            "system": {"nbnd": 24, "nspin": 2},
            "electrons": {"conv_thr": 1e-8},
            "control": {"tprnfor": True, "restart_mode": "from_scratch"},
        }

    def test_value_may_contain_an_equals_sign(self):
        assert C.parse_extra_settings("control.title = a=b") == \
            {"control": {"title": "a=b"}}

    @pytest.mark.parametrize("line", ["nbnd = 24", "system nbnd 24", "system.nbnd"])
    def test_malformed_lines_raise(self, line):
        with pytest.raises(Exception, match="Cannot parse extra setting"):
            C.parse_extra_settings(line)

    @pytest.mark.parametrize("raw,expected", [
        (".true.", True), ("true", True), (".false.", False), ("false", False),
        ("24", 24), ("-3", -3), ("1e-8", 1e-8), ("0.7", 0.7),
        ("'quoted'", "quoted"), ('"quoted"', "quoted"), ("bare", "bare"),
    ])
    def test_coerce(self, raw, expected):
        assert C._coerce(raw) == expected


# --------------------------------------------------------------------------- #
# Pseudopotentials
# --------------------------------------------------------------------------- #

class TestPseudoSets:
    def test_lists_subdirectories_sorted_by_name(self, pseudo_root):
        assert C.list_pseudo_sets() == ["empty-set", "SG14_ONCV", "SSSP-lib-pbe-eff-v2"]

    def test_root_itself_is_offered_first_when_it_holds_upfs(self, pseudo_root):
        open(os.path.join(pseudo_root, "Si.upf"), "w").close()
        sets = C.list_pseudo_sets()
        assert sets[0] == C.PSEUDO_ROOT_CHOICE
        assert sets[1:] == ["empty-set", "SG14_ONCV", "SSSP-lib-pbe-eff-v2"]

    def test_missing_root_leaves_the_dropdown_empty(self, monkeypatch, tmp_path):
        monkeypatch.setattr(C, "PSEUDO_ROOT", str(tmp_path / "nope"))
        assert C.list_pseudo_sets() == []
        assert C.default_pseudo_set() is None

    def test_default_is_the_first_set(self, pseudo_root):
        assert C.default_pseudo_set() == "empty-set"


class TestResolvePseudoDir:
    def test_set_name_resolves_under_the_root(self, pseudo_root):
        assert C.resolve_pseudo_dir("SG14_ONCV") == \
            os.path.join(pseudo_root, "SG14_ONCV")

    def test_dot_means_the_root_itself(self, pseudo_root):
        assert C.resolve_pseudo_dir(C.PSEUDO_ROOT_CHOICE) == pseudo_root

    def test_absolute_path_is_used_as_typed(self, pseudo_root):
        assert C.resolve_pseudo_dir("/opt/pseudos") == "/opt/pseudos"

    def test_tilde_is_expanded(self, pseudo_root):
        assert C.resolve_pseudo_dir("~/pseudos") == \
            os.path.join(os.path.expanduser("~"), "pseudos")

    def test_surrounding_whitespace_is_ignored(self, pseudo_root):
        assert C.resolve_pseudo_dir("  SG14_ONCV  ") == \
            os.path.join(pseudo_root, "SG14_ONCV")

    @pytest.mark.parametrize("choice", [None, "", "   "])
    def test_nothing_selected_explains_espresso_pseudo(self, choice):
        with pytest.raises(Exception, match="ESPRESSO_PSEUDO"):
            C.resolve_pseudo_dir(choice)


class TestMatchPseudopotentials:
    def test_matches_each_element_by_file_stem(self, structure, pseudo_root):
        matched = C.match_pseudopotentials(
            structure, os.path.join(pseudo_root, "SSSP-lib-pbe-eff-v2"))
        assert matched == {"Na": "Na.upf", "Cl": "Cl.upf"}

    def test_stem_is_split_on_dot_underscore_and_dash(self, structure, pseudo_root):
        matched = C.match_pseudopotentials(
            structure, os.path.join(pseudo_root, "SG14_ONCV"))
        assert matched == {"Na": "Na_ONCV_PBE-1.2.upf", "Cl": "Cl_ONCV_PBE-1.2.upf"}

    def test_missing_element_is_named_in_the_error(self, pseudo_root):
        from pymatgen.core import Lattice, Structure
        silicon = Structure(Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])
        with pytest.raises(Exception, match="Si"):
            C.match_pseudopotentials(
                silicon, os.path.join(pseudo_root, "SSSP-lib-pbe-eff-v2"))

    def test_set_without_upf_files(self, structure, pseudo_root):
        with pytest.raises(Exception, match="No .UPF files"):
            C.match_pseudopotentials(structure, os.path.join(pseudo_root, "empty-set"))

    def test_missing_directory(self, structure, pseudo_root):
        with pytest.raises(Exception, match="not found"):
            C.match_pseudopotentials(structure, os.path.join(pseudo_root, "nope"))


# --------------------------------------------------------------------------- #
# pw.x input generation
# --------------------------------------------------------------------------- #

class TestGeneratePwInput:
    def test_writes_the_file_and_the_core_settings(self, pw_input_args, working_dir):
        path = C.generate_pw_input_file(**pw_input_args)
        assert path == os.path.join(working_dir, "scf.in")
        text = open(path).read()
        assert "calculation = 'scf'" in text
        assert "prefix = 'pwscf'" in text
        assert f"outdir = '{C.OUTDIR}'" in text
        assert "SSSP-lib-pbe-eff-v2" in text
        assert "ecutwfc = 50" in text and "ecutrho = 400" in text
        assert "Na.upf" in text and "Cl.upf" in text

    def test_automatic_kpoints_use_the_grid(self, pw_input_args):
        pw_input_args["kgrid"] = (6, 6, 2)
        text = open(C.generate_pw_input_file(**pw_input_args)).read()
        assert "K_POINTS automatic" in text
        assert "6 6 2" in text

    def test_relax_adds_ion_dynamics(self, pw_input_args):
        pw_input_args.update(calc_type="relax", input_file_name="relax.in")
        text = open(C.generate_pw_input_file(**pw_input_args)).read()
        assert "ion_dynamics = 'bfgs'" in text
        assert "forc_conv_thr" in text
        assert "cell_dynamics" not in text

    def test_vc_relax_adds_cell_dynamics(self, pw_input_args):
        pw_input_args.update(calc_type="vc-relax", input_file_name="vc.in")
        text = open(C.generate_pw_input_file(**pw_input_args)).read()
        assert "cell_dynamics = 'bfgs'" in text
        assert "press_conv_thr" in text

    def test_nscf_switches_to_tetrahedra(self, pw_input_args):
        pw_input_args.update(calc_type="nscf", input_file_name="nscf.in")
        text = open(C.generate_pw_input_file(**pw_input_args)).read()
        assert "occupations = 'tetrahedra'" in text
        assert "smearing" not in text and "degauss" not in text
        assert "nosym = .TRUE." in text

    def test_functional_is_written_as_input_dft(self, pw_input_args):
        pw_input_args["input_dft"] = "pbesol"
        text = open(C.generate_pw_input_file(**pw_input_args)).read()
        assert "input_dft = 'pbesol'" in text

    def test_meta_gga_lowers_mixing_beta(self, pw_input_args):
        pw_input_args.update(input_dft="scan", is_metagga=True)
        text = open(C.generate_pw_input_file(**pw_input_args)).read()
        assert "mixing_beta = 0.3" in text

    def test_extras_override_the_defaults(self, pw_input_args):
        pw_input_args.update(input_dft="scan", is_metagga=True, extras={
            "system": {"nbnd": 24},
            "electrons": {"mixing_beta": 0.5, "conv_thr": 1e-10},
        })
        text = open(C.generate_pw_input_file(**pw_input_args)).read()
        assert "nbnd = 24" in text
        assert "mixing_beta = 0.5" in text      # beats the meta-GGA default
        assert "mixing_beta = 0.3" not in text

    def test_unknown_namelist_in_extras_raises(self, pw_input_args):
        pw_input_args["extras"] = {"bogus": {"key": 1}}
        with pytest.raises(Exception, match="Unknown namelist"):
            C.generate_pw_input_file(**pw_input_args)

    def test_missing_pseudo_set_raises_before_writing(self, pw_input_args, working_dir):
        pw_input_args["pseudo_dir"] = None
        with pytest.raises(Exception, match="ESPRESSO_PSEUDO"):
            C.generate_pw_input_file(**pw_input_args)
        assert not os.path.exists(os.path.join(working_dir, "scf.in"))


class TestBandsKpath:
    def test_path_is_labelled_and_ends_each_branch_with_one_point(self,
                                                                  primitive_structure):
        kpath = C.build_kpath_crystal_b(primitive_structure)
        assert len(kpath) > 2
        assert all(label for *_c, label in kpath)
        assert kpath[-1][3] == 1                       # final vertex
        assert all(count >= 1 for *_c, count, _l in kpath)

    def test_repeated_vertices_within_a_branch_are_dropped(self, primitive_structure):
        kpath = C.build_kpath_crystal_b(primitive_structure)
        coords = [k[:3] for k in kpath]
        assert all(a != b for a, b in zip(coords, coords[1:]))

    def test_card_renders_a_count_and_a_label_comment_per_point(self,
                                                                primitive_structure):
        kpath = C.build_kpath_crystal_b(primitive_structure)
        lines = C._kpoints_crystal_b_block(kpath).splitlines()
        assert lines[0] == "K_POINTS crystal_b"
        assert lines[1] == str(len(kpath))
        assert len(lines) == len(kpath) + 2
        assert all(" ! " in line for line in lines[2:])

    def test_bands_input_carries_the_labelled_card(self, pw_input_args,
                                                   primitive_structure):
        pw_input_args.update(calc_type="bands", input_file_name="bands.in",
                             structure=primitive_structure)
        text = open(C.generate_pw_input_file(**pw_input_args)).read()
        assert "calculation = 'bands'" in text
        assert "K_POINTS crystal_b" in text
        assert "! \\Gamma" in text
        # The card must sit between the positions and the cell, and only once.
        assert text.count("K_POINTS") == 1
        assert text.index("ATOMIC_POSITIONS") < text.index("K_POINTS") \
            < text.index("CELL_PARAMETERS")

    def test_the_written_path_is_readable_by_pymatgen_io_espresso(
            self, pw_input_args, primitive_structure):
        from pymatgen.io.espresso.inputs.pwin import PWin
        pw_input_args.update(calc_type="bands", input_file_name="bands.in",
                             structure=primitive_structure)
        path = C.generate_pw_input_file(**pw_input_args)
        k_card = PWin.from_file(path).k_points
        assert k_card.option.name.lower() == "crystal_b"
        assert any(label.strip() for label in k_card.labels)

    def test_the_path_coordinates_are_in_the_written_cells_basis(
            self, pw_input_args, primitive_structure):
        """The bug this guards: QE reads crystal_b coordinates in the basis of the
        cell written to the input file. Interpreted in that cell, FCC's X point has
        to land on the zone-boundary face at |k| = 2*pi/a_cubic; read in the
        conventional cell instead it would come out somewhere else entirely."""
        import numpy as np
        pw_input_args.update(calc_type="bands", input_file_name="bands.in",
                             structure=primitive_structure)
        C.generate_pw_input_file(**pw_input_args)
        kpath = C.build_kpath_crystal_b(primitive_structure)
        reciprocal = primitive_structure.lattice.reciprocal_lattice
        x_point = next(k for k in kpath if k[4] == "X")
        cartesian = reciprocal.get_cartesian_coords(x_point[:3])
        a_cubic = 5.64056
        assert np.linalg.norm(cartesian) == pytest.approx(2 * np.pi / a_cubic,
                                                          rel=1e-3)


class TestPrimitiveCell:
    """A crystal_b path is only meaningful in the cell HighSymmKpath labels it in."""

    def test_a_conventional_cell_is_not_the_standard_primitive(self, structure):
        assert len(structure) == 8
        assert not C.is_standard_primitive(structure)

    def test_the_primitive_cell_is_recognised(self, primitive_structure):
        assert len(primitive_structure) == 2
        assert C.is_standard_primitive(primitive_structure)

    def test_the_conversion_is_idempotent(self, primitive_structure):
        again = C.primitive_standard_structure(primitive_structure)
        assert C.is_standard_primitive(primitive_structure, again)

    def test_a_cif_round_trip_stays_bands_ready(self, working_dir,
                                                primitive_structure_file):
        from pymatgen.core import Structure
        reloaded = Structure.from_file(
            os.path.join(working_dir, primitive_structure_file))
        assert C.is_standard_primitive(reloaded)

    def test_a_supercell_is_rejected_too(self, primitive_structure):
        supercell = primitive_structure.copy()
        supercell.make_supercell([2, 2, 2])
        assert not C.is_standard_primitive(supercell)

    def test_ensure_bands_cell_passes_the_kpath_through(self, primitive_structure):
        assert C.ensure_bands_cell(primitive_structure).kpath["kpoints"]

    def test_ensure_bands_cell_explains_what_to_do(self, structure):
        with pytest.raises(Exception, match="Save Primitive Cell") as raised:
            C.ensure_bands_cell(structure)
        message = str(raised.value)
        assert "8 sites" in message and "primitive cell has 2" in message

    def test_a_rotated_cell_is_still_accepted(self, primitive_structure):
        # Rotating a cell leaves fractional coordinates — and so the path — intact.
        from pymatgen.core import Lattice
        rotated = primitive_structure.copy()
        rotated.lattice = Lattice(primitive_structure.lattice.matrix @ [
            [0, -1, 0], [1, 0, 0], [0, 0, 1]])
        assert C.is_standard_primitive(rotated)


class TestBandsGenerationGuard:
    def test_a_bands_input_is_refused_for_a_conventional_cell(self, pw_input_args,
                                                              working_dir):
        pw_input_args.update(calc_type="bands", input_file_name="bands.in")
        with pytest.raises(Exception, match="not the standard primitive cell"):
            C.generate_pw_input_file(**pw_input_args)
        assert not os.path.exists(os.path.join(working_dir, "bands.in"))

    @pytest.mark.parametrize("calc_type", ["scf", "relax", "vc-relax", "nscf"])
    def test_other_calculation_types_are_unaffected(self, pw_input_args, calc_type):
        pw_input_args.update(calc_type=calc_type, input_file_name=f"{calc_type}.in")
        assert os.path.exists(C.generate_pw_input_file(**pw_input_args))

    def test_the_wrapper_reports_it_as_a_red_status(self, working_dir, structure_file,
                                                   pseudo_root):
        status, files, _dropdown = generate(
            working_dir, "SSSP-lib-pbe-eff-v2", calc_type="bands",
            input_file_name="bands.in")
        assert "color:red" in status and "Save Primitive Cell" in status
        assert files == ["NaCl.cif"]


class TestOnSavePrimitiveCell:
    def test_writes_the_primitive_cif_and_selects_it(self, working_dir,
                                                     structure_file):
        status, files, dropdown = C.on_save_primitive_cell(working_dir,
                                                           structure_file)
        assert "color:green" in status and "8 → 2 sites" in status
        assert files == ["NaCl.cif", "NaCl_primitive.cif"]
        assert dropdown["value"] == "NaCl_primitive.cif"
        assert dropdown["choices"] == ["NaCl.cif", "NaCl_primitive.cif"]

    def test_the_saved_cell_can_then_generate_a_bands_input(self, working_dir,
                                                            structure_file,
                                                            pseudo_root):
        C.on_save_primitive_cell(working_dir, structure_file)
        status, _files, _dropdown = generate(
            working_dir, "SSSP-lib-pbe-eff-v2", calc_type="bands",
            input_structure_file_name="NaCl_primitive.cif",
            input_file_name="bands.in")
        assert "color:green" in status
        assert "K_POINTS crystal_b" in open(os.path.join(working_dir,
                                                         "bands.in")).read()

    def test_an_already_primitive_cell_is_left_alone(self, working_dir,
                                                     primitive_structure_file):
        status, files, dropdown = C.on_save_primitive_cell(working_dir,
                                                           primitive_structure_file)
        assert "already the standard primitive cell" in status
        assert "color:red" not in status
        # No second copy is written and the selection is left alone.
        assert files == ["NaCl.cif", primitive_structure_file]
        assert dropdown == gr_update_noop()

    def test_no_working_directory(self):
        status, files, _dropdown = C.on_save_primitive_cell("", "NaCl.cif")
        assert "color:red" in status and "No working directory" in status
        assert files == []

    def test_no_structure_selected(self, working_dir):
        status, _files, _dropdown = C.on_save_primitive_cell(working_dir, None)
        assert "color:red" in status and "select an input structure" in status

    def test_an_unreadable_structure_is_reported_not_raised(self, working_dir):
        with open(os.path.join(working_dir, "broken.cif"), "w") as fh:
            fh.write("not a cif\n")
        status, _files, _dropdown = C.on_save_primitive_cell(working_dir,
                                                             "broken.cif")
        assert "color:red" in status and "Could not save the primitive cell" in status

    @pytest.mark.parametrize("source,expected", [
        ("NaCl.cif", "NaCl_primitive.cif"),
        ("POSCAR", "POSCAR_primitive.cif"),
        ("cell.vasp", "cell_primitive.cif"),
    ])
    def test_the_derived_file_name(self, source, expected):
        assert C.primitive_cif_name(source) == expected


def gr_update_noop():
    import gradio as gr
    return gr.update()


# --------------------------------------------------------------------------- #
# Post-processing inputs
# --------------------------------------------------------------------------- #

class TestPostInputs:
    @pytest.mark.parametrize("calc_type,namelist", [
        ("dos", "&DOS"), ("projwfc", "&PROJWFC"),
        ("bands.x", "&BANDS"), ("pp.x", "&INPUTPP"),
    ])
    def test_namelist_and_prefix(self, calc_type, namelist):
        text = C.build_post_input(calc_type, "myrun")
        assert text.startswith(namelist)
        assert "prefix = 'myrun'" in text
        assert f"outdir = '{C.OUTDIR}'" in text

    def test_dos_names_the_fildos_after_the_prefix(self):
        assert "fildos = 'myrun.dos'" in C.build_post_input("dos", "myrun")

    def test_projwfc_sets_both_filpdos_and_filproj(self):
        text = C.build_post_input("projwfc", "myrun")
        assert "filpdos = 'myrun'" in text and "filproj = 'myrun'" in text

    def test_pp_writes_a_cube(self):
        text = C.build_post_input("pp.x", "myrun")
        assert "output_format = 6" in text and "fileout = 'myrun.cube'" in text

    def test_unknown_type_raises(self):
        with pytest.raises(Exception, match="Unknown post-processing type"):
            C.build_post_input("nope", "myrun")

    def test_write_post_input_file(self, working_dir):
        path = C.write_post_input_file(working_dir, "dos", "myrun", "dos.in")
        assert path == os.path.join(working_dir, "dos.in")
        assert "fildos = 'myrun.dos'" in open(path).read()


# --------------------------------------------------------------------------- #
# The Gradio wrapper around generation
# --------------------------------------------------------------------------- #

def generate(working_dir, pseudo_set, **overrides):
    """on_generate_qe_input with sensible defaults, positional order preserved."""
    kwargs = dict(calc_type="scf", input_structure_file_name="NaCl.cif",
                  pseudo_set=pseudo_set, ecutwfc=50, ecutrho=400, kx=4, ky=4, kz=4,
                  extra_settings="", input_file_name="scf.in", output_name="pwscf",
                  functional=C.DEFAULT_FUNCTIONAL, custom_functional="")
    kwargs.update(overrides)
    return C.on_generate_qe_input(
        working_dir, kwargs["calc_type"], kwargs["input_structure_file_name"],
        kwargs["pseudo_set"], kwargs["ecutwfc"], kwargs["ecutrho"],
        kwargs["kx"], kwargs["ky"], kwargs["kz"], kwargs["extra_settings"],
        kwargs["input_file_name"], kwargs["output_name"],
        kwargs["functional"], kwargs["custom_functional"])


class TestOnGenerateQeInput:
    def test_success_reports_green_and_refreshes_the_file_list(
            self, working_dir, structure_file, pseudo_root):
        status, files, dropdown = generate(working_dir, "SSSP-lib-pbe-eff-v2")
        assert "color:green" in status and "scf.in" in status
        assert files == ["NaCl.cif", "scf.in"]                 # sorted by name
        assert dropdown["choices"] == ["scf.in"]
        assert dropdown["value"] == "scf.in"

    def test_the_chosen_functional_is_reported(self, working_dir, structure_file,
                                               pseudo_root):
        status, _files, _d = generate(working_dir, "SSSP-lib-pbe-eff-v2",
                                      functional="GGA — PBE")
        assert "input_dft = 'pbe'" in status

    def test_post_types_need_no_structure(self, working_dir, pseudo_root):
        status, files, dropdown = generate(
            working_dir, None, calc_type="dos", input_structure_file_name=None,
            input_file_name="dos.in")
        assert "color:green" in status
        assert files == ["dos.in"] and dropdown["value"] == "dos.in"

    def test_the_dropdown_lists_every_input_file(self, working_dir, structure_file,
                                                 pseudo_root):
        generate(working_dir, "SSSP-lib-pbe-eff-v2")
        _s, _f, dropdown = generate(working_dir, "SSSP-lib-pbe-eff-v2",
                                    calc_type="nscf", input_file_name="nscf.in")
        assert dropdown["choices"] == ["nscf.in", "scf.in"]
        assert dropdown["value"] == "nscf.in"

    @pytest.mark.parametrize("overrides,expected", [
        ({"input_file_name": "../evil.in"}, "invalid characters"),
        ({"input_file_name": ""}, "Please provide"),
        ({"output_name": "out/side"}, "invalid characters"),
        ({"input_structure_file_name": None}, "select an input structure"),
        ({"input_structure_file_name": "missing.cif"}, "Error generating input"),
        ({"extra_settings": "nbnd = 24"}, "Cannot parse extra setting"),
        ({"functional": "Custom (Libxc / input_dft)"}, "Custom input_dft"),
    ])
    def test_errors_are_reported_never_raised(self, working_dir, structure_file,
                                              pseudo_root, overrides, expected):
        status, files, dropdown = generate(working_dir, "SSSP-lib-pbe-eff-v2",
                                           **overrides)
        assert "color:red" in status and expected in status
        assert files == ["NaCl.cif"]        # nothing was written

    def test_no_working_directory(self, pseudo_root):
        status, files, _d = generate("", "SSSP-lib-pbe-eff-v2")
        assert "No working directory is open" in status
        assert files == []


# --------------------------------------------------------------------------- #
# Tab wiring helpers
# --------------------------------------------------------------------------- #

class TestOnSelectCalculationType:
    def test_pw_type_enables_structure_and_grid(self):
        structure, kx, ky, kz, exe, inp, out = C.on_select_calculation_type("scf")
        assert structure["interactive"] and kx["interactive"]
        assert (ky["interactive"], kz["interactive"]) == (True, True)
        assert exe["value"] == "pw.x"
        assert inp["value"] == "scf.in" and out["value"] == "scf.out"

    def test_bands_keeps_the_structure_but_drops_the_grid(self):
        structure, kx, _ky, _kz, exe, inp, out = C.on_select_calculation_type("bands")
        assert structure["interactive"] and not kx["interactive"]
        assert exe["value"] == "pw.x"
        assert (inp["value"], out["value"]) == ("bands.in", "bands.out")

    def test_post_type_picks_its_own_binary(self):
        structure, kx, _ky, _kz, exe, inp, out = C.on_select_calculation_type("dos")
        assert not structure["interactive"] and not kx["interactive"]
        assert exe["value"] == "dos.x"
        assert (inp["value"], out["value"]) == ("dos.in", "dos.out")

    def test_bands_x_maps_to_the_bands_binary(self):
        *_rest, exe, inp, out = C.on_select_calculation_type("bands.x")
        assert exe["value"] == "bands.x"
        assert (inp["value"], out["value"]) == ("bandsx.in", "bandsx.out")


class TestFileListChange:
    FILES = ["a.cif", "b.vasp", "POSCAR", "nscf.in", "scf.in", "scf.out", "run.pwi"]

    def test_dropdowns_are_filtered_by_kind(self):
        structure, inputs = C.on_working_directory_file_list_change(None, None,
                                                                    self.FILES)
        assert structure["choices"] == ["a.cif", "b.vasp", "POSCAR"]
        assert inputs["choices"] == ["nscf.in", "scf.in", "run.pwi"]

    def test_existing_selections_survive_a_refresh(self):
        structure, inputs = C.on_working_directory_file_list_change(
            "b.vasp", "scf.in", self.FILES)
        assert structure["value"] == "b.vasp"
        assert inputs["value"] == "scf.in"

    def test_stale_selections_fall_back_to_the_first_entry(self):
        structure, inputs = C.on_working_directory_file_list_change(
            "deleted.cif", "deleted.in", self.FILES)
        assert structure["value"] == "a.cif"
        assert inputs["value"] == "nscf.in"

    @pytest.mark.parametrize("files", [[], None])
    def test_empty_directory_clears_both(self, files):
        structure, inputs = C.on_working_directory_file_list_change("a.cif", "scf.in",
                                                                    files)
        assert structure["value"] is None and structure["choices"] == []
        assert inputs["value"] is None and inputs["choices"] == []


class TestRefreshFileList:
    def test_lists_the_directory_sorted(self, working_dir):
        for name in ("scf.out", "a.cif"):
            open(os.path.join(working_dir, name), "w").close()
        assert C.refresh_file_list(working_dir) == ["a.cif", "scf.out"]

    def test_without_a_working_directory(self):
        assert C.refresh_file_list(None) == []


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #

class TestResolveExecutable:
    def test_found_on_path(self, monkeypatch):
        monkeypatch.setattr(C.shutil, "which",
                            lambda name: f"/usr/bin/{name}" if name == "pw.x" else None)
        assert C._resolve_executable("pw.x", "") == "/usr/bin/pw.x"
        assert C._resolve_executable("dos.x", None) is None

    def test_bin_dir_wins_and_is_made_absolute(self, tmp_path):
        exe = tmp_path / "pw.x"
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)
        resolved = C._resolve_executable("pw.x", str(tmp_path))
        assert resolved == str(exe) and os.path.isabs(resolved)

    def test_bin_dir_without_the_binary(self, tmp_path):
        assert C._resolve_executable("pw.x", str(tmp_path)) is None

    def test_non_executable_file_is_rejected(self, tmp_path):
        (tmp_path / "pw.x").write_text("not executable")
        assert C._resolve_executable("pw.x", str(tmp_path)) is None


@pytest.fixture
def qe_bin(tmp_path, monkeypatch):
    """A directory holding fake pw.x/mpirun, with mpirun also on PATH."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("pw.x", "mpirun"):
        exe = bin_dir / name
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)
    monkeypatch.setattr(C.shutil, "which",
                        lambda name: str(bin_dir / name) if (bin_dir / name).exists()
                        else None)
    return str(bin_dir)


class TestPreflightRun:
    def test_accepts_a_runnable_request(self, working_dir, qe_bin):
        open(os.path.join(working_dir, "scf.in"), "w").close()
        exe, err = C.preflight_run(working_dir, "pw.x", "scf.in", qe_bin)
        assert err is None and exe.endswith("pw.x")

    def test_no_working_directory(self, qe_bin):
        exe, err = C.preflight_run("", "pw.x", "scf.in", qe_bin)
        assert exe is None and "No working directory" in err

    def test_no_input_file_selected(self, working_dir, qe_bin):
        exe, err = C.preflight_run(working_dir, "pw.x", None, qe_bin)
        assert exe is None and "select an input file" in err

    def test_input_file_missing_from_disk(self, working_dir, qe_bin):
        exe, err = C.preflight_run(working_dir, "pw.x", "scf.in", qe_bin)
        assert exe is None and "not found" in err

    def test_mpirun_missing(self, working_dir, monkeypatch):
        open(os.path.join(working_dir, "scf.in"), "w").close()
        monkeypatch.setattr(C.shutil, "which", lambda name: None)
        exe, err = C.preflight_run(working_dir, "pw.x", "scf.in", "")
        assert exe is None and "mpirun" in err

    def test_executable_missing_names_where_it_looked(self, working_dir, qe_bin):
        open(os.path.join(working_dir, "scf.in"), "w").close()
        exe, err = C.preflight_run(working_dir, "dos.x", "scf.in", qe_bin)
        assert exe is None and qe_bin in err and "dos.x" in err


class FakeProcess:
    """Stands in for the Popen of an mpirun call."""
    instances = []

    def __init__(self, args, **kwargs):
        FakeProcess.instances.append(self)
        self.command = args
        self.kwargs = kwargs
        self.stdout = iter(["JOB DONE line 1\n", "line 2\n"])
        self.returncode = 0
        self.pid = 12345
        self.waited = False

    def wait(self):
        self.waited = True
        return self.returncode

    def poll(self):
        return self.returncode


@pytest.fixture
def fake_popen(monkeypatch):
    FakeProcess.instances = []
    monkeypatch.setattr(C.subprocess, "Popen", FakeProcess)
    return FakeProcess


class TestRunQeStream:
    def test_streams_then_reports_the_exit_code(self, working_dir, fake_popen):
        yields = list(C.run_qe_stream(working_dir, 2, "/opt/qe/pw.x",
                                      "scf.in", "scf.out"))
        assert [rc for _log, rc in yields] == [None, None, 0]
        assert yields[0][0] == "JOB DONE line 1\n"
        assert yields[-1][0] == "JOB DONE line 1\nline 2\n"

    def test_tees_the_log_into_the_working_directory(self, working_dir, fake_popen):
        list(C.run_qe_stream(working_dir, 1, "/opt/qe/pw.x", "scf.in", "scf.out"))
        assert open(os.path.join(working_dir, "scf.out")).read() == \
            "JOB DONE line 1\nline 2\n"

    def test_builds_an_mpirun_command_in_the_working_directory(self, working_dir,
                                                               fake_popen):
        list(C.run_qe_stream(working_dir, 4, "/opt/qe/pw.x", "scf.in", "scf.out"))
        process = fake_popen.instances[0]
        assert process.command == "mpirun -np 4 /opt/qe/pw.x -in scf.in"
        assert process.kwargs["cwd"] == working_dir
        assert process.kwargs["start_new_session"] is True   # so Stop can kill the group

    def test_names_with_spaces_are_quoted(self, working_dir, fake_popen):
        list(C.run_qe_stream(working_dir, 1, "/opt/my qe/pw.x",
                             "my scf.in", "out.log"))
        assert fake_popen.instances[0].command == \
            "mpirun -np 1 '/opt/my qe/pw.x' -in 'my scf.in'"

    def test_the_process_handle_is_released_when_the_run_ends(self, working_dir,
                                                              fake_popen):
        list(C.run_qe_stream(working_dir, 1, "/opt/qe/pw.x", "scf.in", "scf.out"))
        assert C._current_process is None


class TestOnRunCalculation:
    def test_a_failed_preflight_is_reported_without_running(self, working_dir,
                                                            monkeypatch):
        monkeypatch.setattr(C.shutil, "which", lambda name: None)
        status, log = next(C.on_run_calculation(working_dir, 1, "pw.x", "scf.in",
                                                "scf.out", ""))
        assert "color:red" in status and log == ""

    def test_a_successful_run_reports_the_log_file(self, working_dir, qe_bin,
                                                   fake_popen):
        open(os.path.join(working_dir, "scf.in"), "w").close()
        final = list(C.on_run_calculation(working_dir, 1, "pw.x", "scf.in",
                                          "scf.out", qe_bin))[-1]
        assert "color:green" in final[0] and "scf.out" in final[0]

    def test_a_nonzero_exit_is_reported_as_an_error(self, working_dir, qe_bin,
                                                    monkeypatch, fake_popen):
        open(os.path.join(working_dir, "scf.in"), "w").close()
        monkeypatch.setattr(FakeProcess, "returncode", 2, raising=False)

        class Failing(FakeProcess):
            def __init__(self, args, **kwargs):
                super().__init__(args, **kwargs)
                self.returncode = 2
        monkeypatch.setattr(C.subprocess, "Popen", Failing)
        final = list(C.on_run_calculation(working_dir, 1, "pw.x", "scf.in",
                                          "scf.out", qe_bin))[-1]
        assert "color:red" in final[0] and "error code 2" in final[0]

    def test_a_stopped_run_is_reported_as_stopped(self, working_dir, qe_bin,
                                                  monkeypatch):
        open(os.path.join(working_dir, "scf.in"), "w").close()

        class Killed(FakeProcess):
            def __init__(self, args, **kwargs):
                super().__init__(args, **kwargs)
                self.returncode = -15
        monkeypatch.setattr(C.subprocess, "Popen", Killed)
        final = list(C.on_run_calculation(working_dir, 1, "pw.x", "scf.in",
                                          "scf.out", qe_bin))[-1]
        assert "color:orange" in final[0] and "signal 15" in final[0]

    def test_the_output_name_defaults_to_the_input_name(self, working_dir, qe_bin,
                                                        fake_popen):
        open(os.path.join(working_dir, "relax.in"), "w").close()
        list(C.on_run_calculation(working_dir, 1, "pw.x", "relax.in", "  ", qe_bin))
        assert os.path.exists(os.path.join(working_dir, "relax.out"))


class TestOnStopCalculation:
    def test_nothing_running(self, monkeypatch):
        monkeypatch.setattr(C, "_current_process", None)
        assert "No calculation is currently running" in C.on_stop_calculation()

    def test_an_already_finished_process_is_not_signalled(self, monkeypatch):
        monkeypatch.setattr(C, "_current_process", FakeProcess("cmd"))
        assert "No calculation is currently running" in C.on_stop_calculation()

    def test_a_running_process_is_signalled(self, monkeypatch):
        killed = {}

        class Running(FakeProcess):
            def poll(self):
                return None
        monkeypatch.setattr(C, "_current_process", Running("cmd"))
        monkeypatch.setattr(C.os, "getpgid", lambda pid: pid)
        monkeypatch.setattr(C.os, "killpg",
                            lambda pgid, sig: killed.update(pgid=pgid, sig=sig))
        status = C.on_stop_calculation()
        assert "Stopping calculation" in status
        assert killed == {"pgid": 12345, "sig": C.signal.SIGTERM}

    def test_a_failure_to_signal_is_reported(self, monkeypatch):
        class Running(FakeProcess):
            def poll(self):
                return None

        def boom(*_args):
            raise PermissionError("nope")
        monkeypatch.setattr(C, "_current_process", Running("cmd"))
        monkeypatch.setattr(C.os, "getpgid", boom)
        assert "Could not stop calculation" in C.on_stop_calculation()


class TestWriteRelaxedCif:
    def test_returns_none_for_a_non_relax_input(self, working_dir):
        with open(os.path.join(working_dir, "scf.in"), "w") as fh:
            fh.write("&CONTROL\n calculation = 'scf'\n prefix = 'pwscf'\n/\n")
        assert C.write_relaxed_cif(working_dir, "scf.in", "scf.out") is None

    def test_returns_none_when_the_xml_is_missing(self, working_dir):
        with open(os.path.join(working_dir, "relax.in"), "w") as fh:
            fh.write("&CONTROL\n calculation = 'relax'\n prefix = 'pwscf'\n/\n")
        assert C.write_relaxed_cif(working_dir, "relax.in", "relax.out") is None

    def test_never_raises_on_a_missing_input_file(self, working_dir):
        assert C.write_relaxed_cif(working_dir, "gone.in", "gone.out") is None
