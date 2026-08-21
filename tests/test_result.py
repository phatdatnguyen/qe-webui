"""result.py — output discovery, summary/DOS/band helpers, export.

Parsing a real QE XML needs a finished calculation, so the functions that consume
a parsed run are driven with a stub standing in for PWxml; everything that only
touches the filesystem runs against real files.
"""
import os

import numpy as np
import pytest

import result as R


class StubRun:
    """The slice of PWxml the result helpers actually use."""

    def __init__(self, calculation="scf", final_energy=-1234.5, nkpoints=36, **extra):
        self.parameters = {"control_variables": {"calculation": calculation}}
        self.final_energy = final_energy
        self.actual_kpoints = [[0, 0, 0]] * nkpoints
        self.__dict__.update(extra)


def bundle_for(pwxml, path="/tmp/wd", prefix="pwscf", xml_path="out/pwscf.xml"):
    return {"pwxml": pwxml, "xml_path": xml_path, "prefix": prefix,
            "path": path, "errors": []}


@pytest.fixture
def out_dir(working_dir):
    """<working dir>/out, where QE writes <prefix>.xml and <prefix>.save/."""
    path = os.path.join(working_dir, "out")
    os.makedirs(path)
    return path


def touch(*parts):
    path = os.path.join(*parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()
    return path


# --------------------------------------------------------------------------- #
# Finding output files
# --------------------------------------------------------------------------- #

class TestFindXmlChoices:
    def test_lists_pw_runs_sorted_by_name(self, working_dir, out_dir):
        for name in ("pwscf.xml", "band.xml", "pdos.xml"):
            touch(out_dir, name)
        assert R.find_xml_choices(working_dir) == [
            "out/band.xml", "out/pdos.xml", "out/pwscf.xml"]

    def test_skips_the_projwfc_atomic_proj_file(self, working_dir, out_dir):
        touch(out_dir, "pdos.xml")
        touch(out_dir, "pdos.save", "atomic_proj.xml")
        assert R.find_xml_choices(working_dir) == ["out/pdos.xml"]

    def test_hides_the_save_copy_of_a_run_already_listed(self, working_dir, out_dir):
        touch(out_dir, "pwscf.xml")
        touch(out_dir, "pwscf.save", "data-file-schema.xml")
        assert R.find_xml_choices(working_dir) == ["out/pwscf.xml"]

    def test_keeps_the_save_copy_when_it_is_the_only_one(self, working_dir, out_dir):
        touch(out_dir, "pwscf.save", "data-file-schema.xml")
        assert R.find_xml_choices(working_dir) == \
            ["out/pwscf.save/data-file-schema.xml"]

    def test_finds_xml_outside_outdir_too(self, working_dir, out_dir):
        touch(working_dir, "loose.xml")
        touch(out_dir, "pwscf.xml")
        assert R.find_xml_choices(working_dir) == ["loose.xml", "out/pwscf.xml"]

    @pytest.mark.parametrize("path", [None, ""])
    def test_no_working_directory(self, path):
        assert R.find_xml_choices(path) == []

    def test_directory_without_output(self, working_dir):
        assert R.find_xml_choices(working_dir) == []


class TestDefaultXmlChoice:
    def test_prefers_the_conventional_run(self):
        choices = ["out/band.xml", "out/pwscf.xml"]
        assert R.default_xml_choice(choices) == "out/pwscf.xml"

    def test_otherwise_the_first_by_name(self):
        assert R.default_xml_choice(["out/band.xml", "out/pdos.xml"]) == "out/band.xml"

    def test_nothing_to_choose(self):
        assert R.default_xml_choice([]) is None


class TestPrefixFromXml:
    @pytest.mark.parametrize("xml,expected", [
        ("out/pwscf.xml", "pwscf"),
        ("out/my_run.xml", "my_run"),
        ("out/band.save/data-file-schema.xml", "band"),
        ("weird", R.PREFIX),
    ])
    def test_infers_the_run_prefix(self, xml, expected):
        assert R._prefix_from_xml(xml) == expected


class TestRefreshPickers:
    @pytest.fixture
    def three_runs(self, working_dir, out_dir):
        for name in ("pwscf.xml", "band.xml", "pdos.xml"):
            touch(out_dir, name)
        return working_dir

    def test_the_current_pick_survives_a_file_change(self, three_runs):
        update = R.on_refresh_result_files(three_runs, "out/band.xml")
        assert update["value"] == "out/band.xml"
        assert update["choices"] == ["out/band.xml", "out/pdos.xml", "out/pwscf.xml"]

    def test_a_deleted_pick_falls_back_to_the_default(self, three_runs):
        assert R.on_refresh_result_files(three_runs, "out/gone.xml")["value"] == \
            "out/pwscf.xml"

    def test_no_output_clears_the_picker(self, working_dir):
        update = R.on_refresh_result_files(working_dir, "out/pwscf.xml")
        assert update["choices"] == [] and update["value"] is None

    def test_compare_keeps_only_the_picks_that_still_exist(self, three_runs):
        update = R.on_refresh_compare_files(
            three_runs, ["out/band.xml", "out/gone.xml", "out/pdos.xml"])
        assert update["value"] == ["out/band.xml", "out/pdos.xml"]

    @pytest.mark.parametrize("current", [None, []])
    def test_compare_with_nothing_selected(self, three_runs, current):
        assert R.on_refresh_compare_files(three_runs, current)["value"] == []


class TestFindQeXml:
    def test_prefers_the_conventional_location(self, working_dir, out_dir):
        expected = touch(out_dir, f"{R.PREFIX}.xml")
        touch(working_dir, "other.xml")
        assert R.find_qe_xml(working_dir) == expected

    def test_falls_back_to_the_save_directory(self, working_dir, out_dir):
        expected = touch(out_dir, f"{R.PREFIX}.save", "data-file-schema.xml")
        assert R.find_qe_xml(working_dir) == expected

    def test_falls_back_to_any_xml_deterministically(self, working_dir, out_dir):
        touch(out_dir, "zeta.xml")
        expected = touch(out_dir, "alpha.xml")
        assert R.find_qe_xml(working_dir) == expected

    def test_nothing_found(self, working_dir):
        assert R.find_qe_xml(working_dir) is None


class TestFindDosFiles:
    def test_finds_both_dos_and_pdos(self, working_dir):
        touch(working_dir, "pwscf.dos")
        touch(working_dir, "pwscf.pdos_atm#1(Na)_wfc#1(s)")
        fildos, filpdos = R._find_dos_files(working_dir, "pwscf")
        assert fildos == os.path.join(working_dir, "pwscf.dos")
        assert filpdos == os.path.join(working_dir, "pwscf")

    def test_another_run_prefix_is_not_picked_up(self, working_dir):
        touch(working_dir, "other.dos")
        assert R._find_dos_files(working_dir, "pwscf") == (None, None)

    def test_dos_without_pdos(self, working_dir):
        touch(working_dir, "pwscf.dos")
        fildos, filpdos = R._find_dos_files(working_dir, "pwscf")
        assert fildos and filpdos is None


class TestPickCubeFile:
    def test_prefers_the_cube_of_the_selected_run(self, working_dir):
        touch(working_dir, "aaa.cube")
        expected = touch(working_dir, "pdos.cube")
        assert R.pick_cube_file(working_dir, "out/pdos.xml") == expected

    def test_otherwise_the_first_by_name(self, working_dir):
        expected = touch(working_dir, "aaa.cube")
        touch(working_dir, "zzz.cube")
        assert R.pick_cube_file(working_dir, "out/pwscf.xml") == expected

    def test_no_cube_at_all(self, working_dir):
        assert R.pick_cube_file(working_dir, None) is None

    def test_render_reports_the_missing_cube(self, working_dir):
        html, message = R.on_render_cube(working_dir, None, 0.01)
        assert html is None and "No .cube file found" in message

    def test_render_without_a_working_directory(self):
        html, message = R.on_render_cube("", None, 0.01)
        assert html is None and "No working directory" in message


# --------------------------------------------------------------------------- #
# Parsing guard rails
# --------------------------------------------------------------------------- #

class TestParseQeOutputs:
    def test_no_working_directory(self):
        bundle = R.parse_qe_outputs(None)
        assert bundle["pwxml"] is None
        assert "No working directory" in bundle["errors"][0]

    def test_no_xml_in_the_directory(self, working_dir):
        bundle = R.parse_qe_outputs(working_dir)
        assert bundle["pwxml"] is None
        assert "No Quantum Espresso XML output" in bundle["errors"][0]

    def test_an_unreadable_xml_is_reported_not_raised(self, working_dir, out_dir):
        with open(os.path.join(out_dir, "pwscf.xml"), "w") as fh:
            fh.write("<not-a-qe-file/>")
        bundle = R.parse_qe_outputs(working_dir, "out/pwscf.xml")
        assert bundle["pwxml"] is None
        assert bundle["prefix"] == "pwscf"
        assert bundle["errors"] and "pwscf.xml" in bundle["errors"][0]

    def test_a_selection_that_no_longer_exists_falls_back(self, working_dir, out_dir):
        touch(out_dir, "pwscf.xml")
        bundle = R.parse_qe_outputs(working_dir, "out/deleted.xml")
        assert bundle["xml_path"] == os.path.join(out_dir, "pwscf.xml")


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #

class TestFinalEnergyText:
    def test_a_real_energy_is_formatted(self):
        assert R._final_energy_text(StubRun(final_energy=-7842.123456789)) == \
            "-7842.123457"

    @pytest.mark.parametrize("calculation", ["nscf", "bands"])
    def test_zero_energy_is_flagged_not_printed_as_a_number(self, calculation):
        text = R._final_energy_text(StubRun(calculation=calculation, final_energy=0.0))
        assert "n/a" in text and calculation in text
        assert "0.000000" not in text

    def test_the_short_form_fits_a_table_cell(self):
        text = R._final_energy_text(StubRun(calculation="nscf", final_energy=0.0),
                                    short=True)
        assert text == "n/a (nscf run)"

    def test_an_unreadable_calculation_type_still_reads_sensibly(self):
        run = StubRun(final_energy=0.0)
        run.parameters = {}
        assert R._final_energy_text(run) == \
            "n/a — no total energy in the XML (this run); see the scf run"


class TestRunCalculationType:
    def test_reads_the_control_namelist(self):
        assert R._run_calculation_type(StubRun(calculation="vc-relax")) == "vc-relax"

    def test_missing_parameters_are_not_an_error(self):
        run = StubRun()
        run.parameters = {}
        assert R._run_calculation_type(run) == ""


class TestBuildSummaryDataframe:
    def full_run(self):
        from pymatgen.core import Lattice, Structure
        structure = Structure(Lattice.cubic(5.64), ["Na", "Cl"],
                              [[0, 0, 0], [0.5, 0.5, 0.5]])
        return StubRun(
            calculation="scf", final_energy=-1234.5,
            eigenvalue_band_properties=(5.1, 6.0, 0.9, True),
            efermi=3.25, converged_electronic=True, converged_ionic=False,
            run_type="PBE", final_structure=structure, nionic_steps=1)

    def test_reports_the_key_scalars(self):
        table = R.build_summary_dataframe(bundle_for(self.full_run()))
        values = dict(zip(table["Property"], table["Value"]))
        assert values["Final total energy (eV)"] == "-1234.500000"
        assert values["Band gap"] == "5.100 eV (direct)"
        assert values["Fermi level"] == "3.2500 eV"
        assert values["Converged (electronic)"] == "True"
        assert values["Formula"] == "NaCl"
        assert values["Number of sites"] == "2"

    def test_a_tiny_gap_counts_as_metallic(self):
        run = self.full_run()
        run.eigenvalue_band_properties = (0.001, 0.0, 0.0, False)
        table = R.build_summary_dataframe(bundle_for(run))
        assert dict(zip(table["Property"], table["Value"]))["Band gap"] == \
            "Metallic (no gap)"

    def test_rows_degrade_individually(self):
        class Unreadable(StubRun):
            @property
            def efermi(self):
                raise RuntimeError("field missing from this XML")

        run = self.full_run()
        broken = Unreadable(**{k: v for k, v in run.__dict__.items()
                               if k not in ("parameters", "actual_kpoints")})
        table = R.build_summary_dataframe(bundle_for(broken))
        values = dict(zip(table["Property"], table["Value"]))
        assert values["Fermi level"] == "n/a"
        assert values["Formula"] == "NaCl"              # the other rows survive

    def test_without_a_parsed_run(self):
        table = R.build_summary_dataframe(bundle_for(None))
        assert table["Value"].tolist() == ["No readable QE XML output"]


# --------------------------------------------------------------------------- #
# Band structure guard rails
# --------------------------------------------------------------------------- #

@pytest.fixture
def bands_input(working_dir, primitive_structure, pseudo_root):
    """A real line-mode bands input, plus the number of k-points it expands to."""
    import calculation as C
    from pymatgen.io.espresso.inputs.pwin import PWin
    path = C.generate_pw_input_file(
        working_dir, "bands", primitive_structure, "SSSP-lib-pbe-eff-v2", 50, 400,
        (4, 4, 4), None, False, "pwscf", "bands.in")
    return path, R._kpath_length(PWin.from_file(path).k_points)


class TestKpathLength:
    def test_counts_the_points_qe_will_generate(self):
        assert R._kpath_length(type("K", (), {"weights": [20, 20, 1]})()) == 41

    def test_a_single_vertex(self):
        assert R._kpath_length(type("K", (), {"weights": [1]})()) == 1

    @pytest.mark.parametrize("weights", [[], None])
    def test_no_weights(self, weights):
        assert R._kpath_length(type("K", (), {"weights": weights})()) is None

    def test_matches_the_generated_bands_input(self, bands_input):
        _path, expected = bands_input
        assert expected > 1


class TestFindBandsInput:
    def test_ignores_inputs_without_a_crystal_b_card(self, working_dir):
        with open(os.path.join(working_dir, "scf.in"), "w") as fh:
            fh.write("K_POINTS automatic\n 4 4 4 0 0 0\n")
        assert R._find_bands_input(working_dir) is None

    def test_picks_candidates_in_name_order(self, working_dir):
        for name in ("zeta.in", "alpha.in"):
            with open(os.path.join(working_dir, name), "w") as fh:
                fh.write("K_POINTS crystal_b\n")
        assert R._find_bands_input(working_dir) == \
            os.path.join(working_dir, "alpha.in")

    def test_prefers_the_path_matching_the_runs_kpoint_count(self, working_dir,
                                                             bands_input):
        path, expected = bands_input
        with open(os.path.join(working_dir, "aaa.in"), "w") as fh:
            fh.write("K_POINTS crystal_b\n2\n 0.0 0.0 0.0 5\n 0.5 0.0 0.5 1\n")
        assert R._find_bands_input(working_dir, expected) == path
        assert R._find_bands_input(working_dir) == \
            os.path.join(working_dir, "aaa.in")     # no count given: first by name


class TestBandsInputFor:
    def test_accepts_the_matching_bands_run(self, working_dir, bands_input):
        path, expected = bands_input
        bundle = bundle_for(StubRun(calculation="bands", nkpoints=expected),
                            path=working_dir)
        assert R._bands_input_for(bundle) == (path, None)

    def test_an_scf_run_is_told_to_pick_the_bands_xml(self, working_dir, bands_input):
        _path, expected = bands_input
        bundle = bundle_for(StubRun(calculation="nscf", nkpoints=expected - 1),
                            path=working_dir)
        found, error = R._bands_input_for(bundle)
        assert found is None
        assert "'nscf' run" in error and str(expected) in error
        assert "bands.in" in error

    def test_no_bands_input_in_the_directory(self, working_dir):
        found, error = R._bands_input_for(bundle_for(StubRun(), path=working_dir))
        assert found is None and "K_POINTS crystal_b" in error

    def test_an_unlabelled_path_is_explained(self, working_dir, bands_input):
        path, _expected = bands_input
        text = "\n".join(line.split(" ! ")[0]
                         for line in open(path).read().splitlines())
        open(path, "w").write(text)
        found, error = R._bands_input_for(bundle_for(StubRun(), path=working_dir))
        assert found is None and "no k-point labels" in error

    def test_the_message_names_the_caller(self, working_dir):
        _found, error = R._bands_input_for(bundle_for(StubRun(), path=working_dir),
                                           "Projected bands")
        assert error.startswith("Projected bands")


class TestPlotsWithoutData:
    def test_band_structure_without_a_parsed_run(self):
        figure, message = R.build_band_structure_plot(bundle_for(None))
        assert figure is None and "XML not available" in message

    def test_dos_without_a_parsed_run(self):
        figure, message = R.build_dos_plot(bundle_for(None, xml_path=None))
        assert figure is None and "XML not available" in message

    def test_dos_without_any_dos_file(self, working_dir):
        figure, message = R.build_dos_plot(bundle_for(StubRun(), path=working_dir))
        assert figure is None and "No DOS data found" in message

    def test_convergence_without_a_parsed_run(self):
        figure, message = R.build_convergence_plot(bundle_for(None))
        assert figure is None and "XML not available" in message

    def test_projected_bands_without_a_working_directory(self):
        figure, message = R.on_render_projected_bands("", "out/pwscf.xml")
        assert figure is None and "No working directory" in message


class TestBuildConvergencePlot:
    def test_plots_the_energy_of_every_ionic_step(self):
        steps = [{"total_energy": {"etot": e}} for e in (-10.0, -10.5, -10.6)]
        figure, message = R.build_convergence_plot(bundle_for(StubRun(ionic_steps=steps)))
        assert figure is not None and "3 ionic steps" in message
        assert figure.axes[0].lines[0].get_ydata().tolist() == [-10.0, -10.5, -10.6]

    def test_a_single_point_run_has_nothing_to_plot(self):
        run = StubRun(ionic_steps=[{"total_energy": {"etot": -10.0}}])
        figure, message = R.build_convergence_plot(bundle_for(run))
        assert figure is None and "Single-point run" in message

    def test_steps_without_an_energy_are_skipped(self):
        steps = [{"total_energy": {"etot": -10.0}}, {}, {"total_energy": {}}]
        figure, message = R.build_convergence_plot(bundle_for(StubRun(ionic_steps=steps)))
        assert figure is None and "Single-point run" in message


class TestTrajectory:
    def test_a_single_point_run_has_no_trajectory(self):
        from pymatgen.core import Lattice, Structure
        one = [Structure(Lattice.cubic(5.0), ["Na"], [[0, 0, 0]])]
        html, message = R.build_trajectory_html(bundle_for(StubRun(structures=one)))
        assert html is None and "Single-point run" in message

    def test_an_unparsed_run_has_no_trajectory(self):
        html, message = R.build_trajectory_html(bundle_for(None))
        assert html is None and "Single-point run" in message

    def test_multiple_frames_render_an_iframe(self, tmp_path, monkeypatch):
        from pymatgen.core import Lattice, Structure
        monkeypatch.chdir(tmp_path)
        (tmp_path / "static").mkdir()
        frames = [Structure(Lattice.cubic(5.0), ["Na", "Cl"],
                            [[0, 0, 0], [0.5, 0.5, 0.5 - 0.01 * i]])
                  for i in range(3)]
        html, message = R.build_trajectory_html(bundle_for(StubRun(structures=frames)))
        assert html is not None and "?ts=" in html
        assert "3 ionic steps" in message


# --------------------------------------------------------------------------- #
# Projected DOS grouping
# --------------------------------------------------------------------------- #

class StubOrbital:
    def __init__(self, name):
        self.name = name


class StubDosRun:
    """Two Na sites (s + p) and one Cl site (s), one spin channel."""

    def __init__(self):
        up = 1
        self.atomic_symbols = ["Na", "Na", "Cl"]
        self.pdos = [
            {StubOrbital("s"): {up: np.array([1.0, 1.0])},
             StubOrbital("px"): {up: np.array([2.0, 2.0])}},
            {StubOrbital("s"): {up: np.array([1.0, 1.0])}},
            {StubOrbital("s"): {up: np.array([4.0, 4.0])}},
        ]


class TestProjectedDosSeries:
    def test_grouped_by_element(self):
        groups = R._projected_dos_series(StubDosRun(), "element")
        assert set(groups) == {"Na", "Cl"}
        assert groups["Na"][1].tolist() == [4.0, 4.0]      # 1 + 2 + 1
        assert groups["Cl"][1].tolist() == [4.0, 4.0]

    def test_grouped_by_orbital_letter(self):
        groups = R._projected_dos_series(StubDosRun(), "orbital")
        assert set(groups) == {"s", "p"}
        assert groups["s"][1].tolist() == [6.0, 6.0]       # 1 + 1 + 4
        assert groups["p"][1].tolist() == [2.0, 2.0]

    def test_no_projection_available(self):
        assert R._projected_dos_series(StubRun(pdos=None), "element") == {}


class TestDosMode:
    @pytest.mark.parametrize("label,expected", [
        ("Total + element", "element"),
        ("Total + orbital (s/p/d)", "orbital"),
        (None, "element"),
        ("", "element"),
    ])
    def test_radio_label_maps_to_mode(self, label, expected):
        assert R._dos_mode(label) == expected


# --------------------------------------------------------------------------- #
# Tab-level handlers
# --------------------------------------------------------------------------- #

class TestOnLoadResults:
    def test_returns_every_output_slot_even_with_no_data(self, working_dir):
        outputs = R.on_load_results(working_dir, None, "Total + element")
        assert len(outputs) == 11
        summary, dos_fig, _dos_msg, bs_fig, _bs_msg = outputs[:5]
        assert list(summary.columns) == ["Property", "Value"]
        assert dos_fig is None and bs_fig is None
        assert "color:red" in outputs[-1]

    def test_never_raises_without_a_working_directory(self):
        outputs = R.on_load_results(None, None, "Total + element")
        assert len(outputs) == 11 and "color:red" in outputs[-1]

    def test_the_hint_is_cleared_on_load(self, working_dir):
        assert R.on_load_results(working_dir, None, "Total + element")[9] == ""


class TestOnResultFileListChange:
    def test_clears_the_sections_and_asks_for_a_reload(self, working_dir):
        outputs = R.on_result_file_list_change(working_dir)
        assert len(outputs) == 10
        assert outputs[0].empty
        assert "Load / Refresh Results" in outputs[-1]


class TestOnCompareRuns:
    COLUMNS = ["Run", "Formula", "Energy (eV)", "Band gap", "E_F (eV)", "Run type"]

    def test_no_working_directory(self):
        table, figure, message = R.on_compare_runs(None, ["out/pwscf.xml"])
        assert list(table.columns) == self.COLUMNS and table.empty
        assert figure is None and "No working directory" in message

    @pytest.mark.parametrize("selection", [None, []])
    def test_nothing_selected(self, working_dir, selection):
        table, figure, message = R.on_compare_runs(working_dir, selection)
        assert table.empty and figure is None
        assert "Select one or more XML files" in message

    def test_an_unreadable_run_becomes_a_row_not_an_exception(self, working_dir,
                                                              out_dir):
        with open(os.path.join(out_dir, "pwscf.xml"), "w") as fh:
            fh.write("<not-a-qe-file/>")
        table, figure, message = R.on_compare_runs(working_dir, ["out/pwscf.xml"])
        assert table["Run"].tolist() == ["pwscf"]
        assert table["Energy (eV)"].tolist() == ["unreadable"]
        assert figure is None and "Compared 1 run(s)" in message


class TestExports:
    def test_dos_export_without_dos_files_warns_and_returns_nothing(self, working_dir,
                                                                    out_dir):
        touch(out_dir, "pwscf.xml")
        with pytest.warns(UserWarning, match="No DOS files found"):
            assert R.on_export_dos_csv(working_dir, "out/pwscf.xml",
                                       "Total + element") is None

    def test_band_export_without_a_readable_run_warns(self, working_dir):
        with pytest.warns(UserWarning, match="No eigenvalues found"):
            assert R.on_export_bands_csv(working_dir, None) is None

    def test_exports_never_raise_without_a_working_directory(self):
        with pytest.warns(UserWarning, match="Could not export DOS"):
            assert R.on_export_dos_csv(None, None, "Total + element") is None
        with pytest.warns(UserWarning, match="No eigenvalues found"):
            assert R.on_export_bands_csv(None, None) is None
