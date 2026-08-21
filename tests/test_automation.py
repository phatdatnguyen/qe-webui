"""automation.py — workflow definitions, convergence sweeps, and their guards."""
import os

import pytest

import automation as A
import calculation as C


# --------------------------------------------------------------------------- #
# Workflow definitions
# --------------------------------------------------------------------------- #

class TestWorkflowDefinitions:
    def test_every_stage_is_a_known_calculation_type(self):
        for name, stages in A.WORKFLOWS.items():
            for calc_type, _exe in stages:
                assert calc_type in C.CALC_TYPES, f"{name}: {calc_type}"

    def test_every_stage_runs_the_binary_that_type_needs(self):
        for name, stages in A.WORKFLOWS.items():
            for calc_type, exe in stages:
                assert exe == C.CALC_EXECUTABLE[calc_type], f"{name}: {calc_type}"

    def test_every_stage_has_a_distinct_input_file(self):
        for name, stages in A.WORKFLOWS.items():
            names = [C.default_input_name(calc_type) for calc_type, _e in stages]
            assert len(names) == len(set(names)), f"{name}: {names}"

    def test_post_processing_stages_come_after_a_pw_stage(self):
        for name, stages in A.WORKFLOWS.items():
            assert stages[0][0] in C.PW_TYPES, name

    def test_the_outdir_matches_the_one_calculation_writes(self):
        # automation looks for <wd>/out/<prefix>.xml; calculation writes ./out.
        assert os.path.normpath(C.OUTDIR) == A.OUT_SUBDIR


class TestStructureDropdowns:
    FILES = ["b.cif", "a.cif", "scf.in", "POSCAR"]

    def test_both_dropdowns_offer_the_structures(self):
        workflow, convergence = A.on_file_list_change_structures(None, None, self.FILES)
        assert workflow["choices"] == ["b.cif", "a.cif", "POSCAR"]
        assert convergence["choices"] == workflow["choices"]

    def test_each_dropdown_keeps_its_own_selection(self):
        workflow, convergence = A.on_file_list_change_structures("POSCAR", "a.cif",
                                                                 self.FILES)
        assert workflow["value"] == "POSCAR"
        assert convergence["value"] == "a.cif"

    def test_a_deleted_selection_falls_back_to_the_first(self):
        workflow, convergence = A.on_file_list_change_structures("gone.cif", None,
                                                                 self.FILES)
        assert workflow["value"] == "b.cif" and convergence["value"] == "b.cif"

    @pytest.mark.parametrize("files", [[], None])
    def test_no_structures_clears_both(self, files):
        workflow, convergence = A.on_file_list_change_structures("a.cif", "b.cif",
                                                                 files)
        assert workflow["value"] is None and convergence["value"] is None


# --------------------------------------------------------------------------- #
# Convergence sweeps
# --------------------------------------------------------------------------- #

class TestConvValues:
    def test_inclusive_of_the_stop_value(self):
        assert A._conv_values("ecutwfc", 30, 70, 10) == [30, 40, 50, 60, 70]

    def test_a_fractional_step_keeps_one_point_per_step(self):
        assert A._conv_values("ecutwfc", 30, 31, 0.5) == [30, 30.5, 31]

    def test_whole_numbers_are_not_written_as_floats(self):
        assert A._conv_values("ecutwfc", 30.0, 32.0, 1.0) == [30, 31, 32]

    def test_k_grid_values_are_integers(self):
        assert A._conv_values("k-grid", 2, 8, 2) == [2, 4, 6, 8]

    def test_a_fractional_k_grid_step_is_rounded_to_whole_grids(self):
        # 2, 3.5, 5, 6.5 -> rounded; 6.5 is past the stop value so it is dropped.
        assert A._conv_values("k-grid", 2, 6, 1.5) == [2, 4, 5]

    def test_start_equal_to_stop_is_a_single_point(self):
        assert A._conv_values("ecutwfc", 40, 40, 10) == [40]

    def test_a_tiny_step_cannot_spawn_an_unbounded_sweep(self):
        values = A._conv_values("ecutwfc", 30, 1000, 0.1)
        assert len(values) == A.MAX_CONV_POINTS

    @pytest.mark.parametrize("step", [0, -5])
    def test_a_non_positive_step_is_rejected(self, step):
        with pytest.raises(Exception, match="Step must be greater than 0"):
            A._conv_values("ecutwfc", 30, 70, step)

    def test_a_backwards_range_is_rejected(self):
        with pytest.raises(Exception, match="Stop must be"):
            A._conv_values("ecutwfc", 70, 30, 10)


class TestConvergenceOutputs:
    def test_table_and_plot_from_the_scan(self):
        figure, table = A._convergence_outputs(
            "ecutwfc", [(30, -100.0), (40, -100.5), (50, -100.6)])
        assert table["Value"].tolist() == [30, 40, 50]
        assert table["Total energy (eV)"].tolist() == \
            ["-100.000000", "-100.500000", "-100.600000"]
        assert table["ΔE vs previous (eV)"].tolist() == \
            ["", "-0.500000", "-0.100000"]
        assert figure.axes[0].get_xlabel() == "ecutwfc (Ry)"

    def test_the_k_grid_axis_is_labelled_for_a_grid_scan(self):
        figure, _table = A._convergence_outputs("k-grid", [(2, -1.0), (4, -1.1)])
        assert figure.axes[0].get_xlabel() == "k-grid (n×n×n)"

    def test_a_point_whose_energy_could_not_be_read(self):
        figure, table = A._convergence_outputs(
            "ecutwfc", [(30, -100.0), (40, None), (50, -100.6)])
        assert table["Total energy (eV)"].tolist()[1] == "n/a"
        assert table["ΔE vs previous (eV)"].tolist() == ["", "", "-0.600000"]
        assert figure.axes[0].lines[0].get_xdata().tolist() == [30, 50]

    def test_no_readable_energies_at_all(self):
        figure, table = A._convergence_outputs("ecutwfc", [(30, None)])
        assert figure is None
        assert table["Total energy (eV)"].tolist() == ["n/a"]


# --------------------------------------------------------------------------- #
# Handlers: the guards that run before any QE process is started
# --------------------------------------------------------------------------- #

def first(generator):
    """The first yield of a Gradio streaming handler."""
    return next(generator)


def never_called(*_args, **_kwargs):
    raise AssertionError("a QE process was started when it should not have been")


class TestOnRunWorkflow:
    def args(self, working_dir, **overrides):
        kwargs = dict(working_directory_path=working_dir, workflow_type="SCF → DOS",
                      structure_file="NaCl.cif", pseudo_set="SSSP-lib-pbe-eff-v2",
                      ecutwfc=50, ecutrho=400, kx=4, ky=4, kz=4,
                      functional=C.DEFAULT_FUNCTIONAL, custom_functional="",
                      output_name="pwscf", num_cores=1, qe_bin_dir="")
        kwargs.update(overrides)
        return A.on_run_workflow(*kwargs.values())

    def test_no_working_directory(self):
        status, log = first(self.args(""))
        assert "color:red" in status and "No working directory" in status
        assert log == ""

    def test_no_structure_selected(self, working_dir):
        status, _log = first(self.args(working_dir, structure_file=None))
        assert "select an input structure" in status

    def test_an_invalid_output_name(self, working_dir, structure_file):
        status, _log = first(self.args(working_dir, output_name="../evil"))
        assert "invalid characters" in status

    def test_an_unknown_workflow(self, working_dir, structure_file):
        status, _log = first(self.args(working_dir, workflow_type="nope"))
        assert "Unknown workflow" in status

    def test_a_bad_custom_functional_is_reported_not_raised(
            self, working_dir, structure_file, pseudo_root):
        status, _log = first(self.args(
            working_dir, functional="Custom (Libxc / input_dft)",
            custom_functional=""))
        assert "color:red" in status and "Custom input_dft" in status

    def test_a_band_workflow_checks_the_cell_before_running_anything(
            self, working_dir, structure_file, pseudo_root, monkeypatch):
        # The conventional cell cannot carry a labelled k-path; the workflow must
        # say so up front rather than after the scf stage has already run.
        monkeypatch.setattr(C.subprocess, "Popen", never_called)
        outputs = list(self.args(working_dir, workflow_type="SCF → Band structure"))
        assert "color:red" in outputs[-1][0]
        assert "Save Primitive Cell" in outputs[-1][0]
        assert not os.path.exists(os.path.join(working_dir, "scf.in"))

    def test_a_band_workflow_accepts_the_primitive_cell(
            self, working_dir, primitive_structure_file, pseudo_root, monkeypatch):
        monkeypatch.setattr(C.shutil, "which", lambda name: None)
        outputs = list(self.args(working_dir,
                                 workflow_type="SCF → Band structure",
                                 structure_file=primitive_structure_file))
        # It gets past the cell check and stops at the missing binary instead.
        assert "mpirun" in outputs[-1][0]
        assert os.path.exists(os.path.join(working_dir, "scf.in"))

    def test_a_missing_qe_binary_stops_before_the_first_stage(
            self, working_dir, structure_file, pseudo_root, monkeypatch):
        monkeypatch.setattr(C.shutil, "which", lambda name: None)
        outputs = list(self.args(working_dir))
        assert "color:red" in outputs[-1][0]
        assert "mpirun" in outputs[-1][0]
        # The first stage's input was still generated before the pre-flight failed.
        assert os.path.exists(os.path.join(working_dir, "scf.in"))


class TestOnRunConvergence:
    def args(self, working_dir, **overrides):
        kwargs = dict(working_directory_path=working_dir, structure_file="NaCl.cif",
                      pseudo_set="SSSP-lib-pbe-eff-v2", ecutwfc=50, ecutrho=400,
                      kx=4, ky=4, kz=4, functional=C.DEFAULT_FUNCTIONAL,
                      custom_functional="", output_name="conv", num_cores=1,
                      qe_bin_dir="", param="ecutwfc", start=30, stop=50, step=10)
        kwargs.update(overrides)
        return A.on_run_convergence(*kwargs.values())

    def test_no_working_directory(self):
        status, log, figure, table = first(self.args(""))
        assert "No working directory" in status
        assert log == "" and figure is None and table.empty

    def test_no_structure_selected(self, working_dir):
        status, *_rest = first(self.args(working_dir, structure_file=None))
        assert "select an input structure" in status

    def test_an_invalid_output_name(self, working_dir, structure_file):
        status, *_rest = first(self.args(working_dir, output_name=""))
        assert "Please provide a output name" in status

    def test_a_bad_range_is_reported_not_raised(self, working_dir, structure_file):
        status, _log, figure, table = list(self.args(working_dir, step=0))[-1]
        assert "color:red" in status and "Step must be greater than 0" in status
        assert figure is None and table.empty

    def test_each_point_gets_its_own_prefix_and_input_file(
            self, working_dir, structure_file, pseudo_root, monkeypatch):
        monkeypatch.setattr(C.shutil, "which", lambda name: None)
        list(self.args(working_dir))
        # The sweep stops at the first pre-flight failure, having written point 1.
        written = open(os.path.join(working_dir, "conv_ecutwfc30.in")).read()
        assert "prefix = 'conv_ecutwfc30'" in written
        assert "ecutwfc = 30" in written

    def test_a_low_cutoff_ratio_is_flagged_in_the_status(
            self, working_dir, structure_file, pseudo_root, monkeypatch):
        monkeypatch.setattr(C.shutil, "which", lambda name: None)
        status, *_rest = first(self.args(working_dir, ecutwfc=50, ecutrho=100))
        assert "ecutrho/ecutwfc" in status

    def test_the_k_grid_sweep_scans_the_grid_not_the_cutoff(
            self, working_dir, structure_file, pseudo_root, monkeypatch):
        monkeypatch.setattr(C.shutil, "which", lambda name: None)
        list(self.args(working_dir, param="k-grid", start=2, stop=4, step=2))
        written = open(os.path.join(working_dir, "conv_k2.in")).read()
        assert "2 2 2" in written and "ecutwfc = 50" in written
