"""End-to-end tests that really run Quantum Espresso.

These are the only tests that can catch a generated input QE refuses, or a result
helper that disagrees with a genuine XML — everything else in the suite stubs the
process out. They are opt-in because they cost about a minute:

    pytest -m qe                     # just these
    pytest -m ""                     # everything

They skip themselves unless pw.x/dos.x/mpirun are on PATH and $ESPRESSO_PSEUDO
holds a set covering silicon, so run them from the same shell you start the web UI
from. Each scenario runs its QE steps once (session-scoped fixture) and the tests
then assert against those outputs.
"""
import os
import re
import shutil

import pytest
from pymatgen.core import Lattice, Structure

import calculation as C
import result as R

# Silicon in its standard primitive cell: two atoms, converges in a few seconds,
# and is bands-ready (see calculation.ensure_bands_cell).
SILICON = Structure(Lattice([[0.0, 2.715, 2.715],
                             [2.715, 0.0, 2.715],
                             [2.715, 2.715, 0.0]]),
                    ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])

# Deliberately cheap: these runs are here to exercise the plumbing, not to be
# publication-quality numbers.
ECUTWFC, ECUTRHO, KGRID = 20, 160, (2, 2, 2)

RY_TO_EV = 13.605693122994


def _pseudo_set_covering(structure):
    """First set under $ESPRESSO_PSEUDO with a UPF for every element, else None."""
    if not os.path.isdir(C.PSEUDO_ROOT):
        return None
    for name in C.list_pseudo_sets():
        try:
            C.match_pseudopotentials(structure, C.resolve_pseudo_dir(name))
            return name
        except Exception:
            continue
    return None


MISSING_BINARIES = [name for name in ("mpirun", "pw.x", "dos.x")
                    if shutil.which(name) is None]
PSEUDO_SET = _pseudo_set_covering(SILICON)

pytestmark = [
    pytest.mark.qe,
    pytest.mark.skipif(bool(MISSING_BINARIES),
                       reason=f"not on PATH: {', '.join(MISSING_BINARIES)}"),
    pytest.mark.skipif(PSEUDO_SET is None,
                       reason="no pseudopotential set under $ESPRESSO_PSEUDO covers Si"),
]


def run_qe(working_dir, executable, input_name, cores=1):
    """Run one QE step through the app's own pre-flight + streaming code path."""
    exe, error = C.preflight_run(working_dir, executable, input_name, "")
    assert error is None, error
    log, returncode = "", None
    for log, returncode in C.run_qe_stream(working_dir, cores, exe, input_name,
                                           C.default_output_name(input_name)):
        pass
    assert returncode == 0, \
        f"{executable} on {input_name} exited {returncode}:\n{log[-2000:]}"
    return log


def generate(working_dir, calc_type, prefix, structure=SILICON, kgrid=KGRID):
    return C.generate_pw_input_file(
        working_dir, calc_type, structure, PSEUDO_SET, ECUTWFC, ECUTRHO, kgrid,
        None, False, prefix, C.default_input_name(calc_type))


def energy_from_log(log):
    """The '!    total energy = ... Ry' value pw.x printed, in eV."""
    matches = re.findall(r"^!\s+total energy\s+=\s+(-?\d+\.\d+)\s+Ry", log,
                         re.MULTILINE)
    assert matches, "pw.x printed no total energy"
    return float(matches[-1]) * RY_TO_EV


# --------------------------------------------------------------------------- #
# scf
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def scf_run(tmp_path_factory):
    """A finished scf run; returns (working_dir, log)."""
    working_dir = str(tmp_path_factory.mktemp("qe_scf"))
    generate(working_dir, "scf", "si")
    return working_dir, run_qe(working_dir, "pw.x", "scf.in")


class TestScf:
    def test_qe_accepts_the_generated_input(self, scf_run):
        _working_dir, log = scf_run
        assert "JOB DONE" in log
        assert "Error in routine" not in log

    def test_the_pseudopotentials_were_matched_and_read(self, scf_run):
        _working_dir, log = scf_run
        assert "PseudoPot. #" in log

    def test_the_xml_lands_where_the_result_tab_looks(self, scf_run):
        working_dir, _log = scf_run
        assert os.path.exists(os.path.join(working_dir, "out", "si.xml"))
        assert R.find_qe_xml(working_dir).endswith("si.xml")

    def test_the_picker_offers_the_run_once(self, scf_run):
        working_dir, _log = scf_run
        # out/si.save/data-file-schema.xml is the same run and must not show up.
        assert R.find_xml_choices(working_dir) == ["out/si.xml"]

    def test_the_xml_parses_without_notes(self, scf_run):
        working_dir, _log = scf_run
        bundle = R.parse_qe_outputs(working_dir, "out/si.xml")
        assert bundle["errors"] == []
        assert bundle["prefix"] == "si"

    def test_the_summary_energy_matches_what_pw_x_printed(self, scf_run):
        working_dir, log = scf_run
        bundle = R.parse_qe_outputs(working_dir, "out/si.xml")
        table = R.build_summary_dataframe(bundle)
        values = dict(zip(table["Property"], table["Value"]))
        assert float(values["Final total energy (eV)"]) == \
            pytest.approx(energy_from_log(log), abs=1e-3)

    def test_the_summary_reports_the_cell_and_convergence(self, scf_run):
        working_dir, _log = scf_run
        table = R.build_summary_dataframe(R.parse_qe_outputs(working_dir,
                                                             "out/si.xml"))
        values = dict(zip(table["Property"], table["Value"]))
        assert values["Formula"] == "Si"
        assert values["Number of sites"] == "2"
        assert values["Converged (electronic)"] == "True"
        assert float(values["Fermi level"].split()[0])      # a real number

    def test_a_single_point_run_has_no_trajectory_or_convergence(self, scf_run):
        working_dir, _log = scf_run
        bundle = R.parse_qe_outputs(working_dir, "out/si.xml")
        assert R.build_convergence_plot(bundle)[0] is None
        assert R.build_trajectory_html(bundle)[0] is None

    def test_loading_the_results_reports_success(self, scf_run):
        working_dir, _log = scf_run
        outputs = R.on_load_results(working_dir, "out/si.xml", "Total + element")
        assert "color:green" in outputs[-1]
        assert not outputs[0].empty


# --------------------------------------------------------------------------- #
# bands — the k-path fix, verified against QE itself
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def bands_run(tmp_path_factory):
    """scf followed by a line-mode bands run sharing its prefix."""
    working_dir = str(tmp_path_factory.mktemp("qe_bands"))
    generate(working_dir, "scf", "si_bands")
    run_qe(working_dir, "pw.x", "scf.in")
    generate(working_dir, "bands", "si_bands")
    run_qe(working_dir, "pw.x", "bands.in")
    return working_dir


class TestBands:
    def test_qe_accepts_the_crystal_b_card(self, bands_run):
        log = open(os.path.join(bands_run, "bands.out")).read()
        assert "JOB DONE" in log and "Error in routine" not in log

    def test_qe_generated_exactly_the_k_points_the_card_asked_for(self, bands_run):
        """_kpath_length() predicts QE's expansion — the assumption the Result
        tab uses to pair an XML with its bands input."""
        from pymatgen.io.espresso.inputs.pwin import PWin
        expected = R._kpath_length(
            PWin.from_file(os.path.join(bands_run, "bands.in")).k_points)
        bundle = R.parse_qe_outputs(bands_run, "out/si_bands.xml")
        assert len(bundle["pwxml"].actual_kpoints) == expected

    def test_the_bands_input_is_paired_with_its_run(self, bands_run):
        bundle = R.parse_qe_outputs(bands_run, "out/si_bands.xml")
        path, error = R._bands_input_for(bundle)
        assert error is None
        assert os.path.basename(path) == "bands.in"

    def test_the_band_structure_plots(self, bands_run):
        bundle = R.parse_qe_outputs(bands_run, "out/si_bands.xml")
        figure, message = R.build_band_structure_plot(bundle)
        assert figure is not None, message
        assert "high-symmetry k-path" in message

    def test_the_high_symmetry_labels_survive_into_the_band_structure(self,
                                                                      bands_run):
        bundle = R.parse_qe_outputs(bands_run, "out/si_bands.xml")
        pwxml = bundle["pwxml"]
        if not hasattr(pwxml, "atomic_states"):
            pwxml.atomic_states = None
        band_structure = pwxml.get_band_structure(
            kpoints_filename=os.path.join(bands_run, "bands.in"), line_mode=True)
        labels = {k.label for k in band_structure.kpoints if k.label}
        assert {"\\Gamma", "X", "L"} <= labels

    def test_the_band_edges_land_where_silicons_really_are(self, bands_run):
        """The end-to-end proof that the path is written in the right basis.

        Silicon's valence band maximum is at Gamma and its conduction minimum lies
        ~85% of the way from Gamma to X. Read in the wrong basis those extrema
        would fall on unrelated k-points. Band edges are taken from the occupied
        band count (4 doubly-occupied bands for 8 valence electrons) rather than
        from eigenvalue_band_properties, which mis-reads smeared runs.
        """
        import numpy as np
        from pymatgen.electronic_structure.core import Spin

        pwxml = R.parse_qe_outputs(bands_run, "out/si_bands.xml")["pwxml"]
        energies = pwxml.eigenvalues[Spin.up][:, :, 0]
        kpoints = np.array(pwxml.actual_kpoints)
        n_occupied = 4

        vbm_k = kpoints[np.argmax(energies[:, n_occupied - 1])]
        cbm_k = kpoints[np.argmin(energies[:, n_occupied])]
        assert np.allclose(vbm_k, [0, 0, 0], atol=1e-6), f"VBM at {vbm_k}, not Gamma"

        # X = (0.5, 0, 0.5) in this basis, so the minimum sits on that line.
        assert cbm_k[1] == pytest.approx(0, abs=1e-6)
        assert cbm_k[0] == pytest.approx(cbm_k[2], abs=1e-6)
        assert 0.7 < cbm_k[0] / 0.5 <= 1.0, f"CBM at {cbm_k}, not along Gamma-X"

        gap = energies[:, n_occupied].min() - energies[:, n_occupied - 1].max()
        assert 0.2 < gap < 1.2, f"indirect gap {gap:.3f} eV is not silicon-like"

    def test_the_summary_reports_that_gap(self, bands_run):
        """band_edges() has to reach the same answer as the hand calculation above
        — and not the 0.018 eV that thresholding smeared occupancies gives."""
        import numpy as np
        from pymatgen.electronic_structure.core import Spin

        pwxml = R.parse_qe_outputs(bands_run, "out/si_bands.xml")["pwxml"]
        energies = pwxml.eigenvalues[Spin.up][:, :, 0]
        expected = energies[:, 4].min() - energies[:, 3].max()

        vbm, cbm, is_direct = R.band_edges(pwxml)
        assert cbm - vbm == pytest.approx(expected, abs=1e-9)
        assert not is_direct

        table = R.build_summary_dataframe(R.parse_qe_outputs(bands_run,
                                                             "out/si_bands.xml"))
        value = dict(zip(table["Property"], table["Value"]))["Band gap"]
        assert value == f"{expected:.3f} eV (indirect)"
        assert float(value.split()[0]) > 0.2

    def test_a_bands_run_reports_no_total_energy(self, bands_run):
        bundle = R.parse_qe_outputs(bands_run, "out/si_bands.xml")
        text = R._final_energy_text(bundle["pwxml"])
        assert text.startswith("n/a") and "bands run" in text

    def test_projected_bands_explain_the_missing_projwfc_run(self, bands_run):
        figure, message = R.on_render_projected_bands(bands_run,
                                                      "out/si_bands.xml")
        assert figure is None
        assert "projwfc" in message

    def test_the_eigenvalue_export_writes_a_csv(self, bands_run, tmp_path,
                                                monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "static").mkdir()
        path = R.on_export_bands_csv(bands_run, "out/si_bands.xml")
        assert path and os.path.exists(path)
        header = open(path).readline().strip()
        assert header == "spin,kpoint,band,energy_eV"


# --------------------------------------------------------------------------- #
# nscf + dos.x
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def dos_run(tmp_path_factory):
    """scf -> nscf -> dos.x, the chain the 'SCF → DOS' workflow runs."""
    working_dir = str(tmp_path_factory.mktemp("qe_dos"))
    generate(working_dir, "scf", "si_dos")
    run_qe(working_dir, "pw.x", "scf.in")
    generate(working_dir, "nscf", "si_dos", kgrid=(4, 4, 4))
    run_qe(working_dir, "pw.x", "nscf.in")
    C.write_post_input_file(working_dir, "dos", "si_dos", "dos.in")
    run_qe(working_dir, "dos.x", "dos.in")
    return working_dir


class TestDos:
    def test_dos_x_writes_the_fildos_the_result_tab_looks_for(self, dos_run):
        assert R._find_dos_files(dos_run, "si_dos")[0] == \
            os.path.join(dos_run, "si_dos.dos")

    def test_the_dos_plots(self, dos_run):
        bundle = R.parse_qe_outputs(dos_run, "out/si_dos.xml")
        figure, message = R.build_dos_plot(bundle)
        assert figure is not None, message
        assert "Total DOS" in message

    def test_the_nscf_xml_reports_no_total_energy_instead_of_zero(self, dos_run):
        """The real case behind _final_energy_text: nscf overwrites the scf XML
        under the shared prefix and QE writes etot = 0 into it."""
        bundle = R.parse_qe_outputs(dos_run, "out/si_dos.xml")
        assert R._run_calculation_type(bundle["pwxml"]) == "nscf"
        table = R.build_summary_dataframe(bundle)
        value = dict(zip(table["Property"], table["Value"]))[
            "Final total energy (eV)"]
        assert value.startswith("n/a") and "0.000000" not in value

    def test_the_dos_export_writes_a_csv(self, dos_run, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "static").mkdir()
        path = R.on_export_dos_csv(dos_run, "out/si_dos.xml", "Total + element")
        assert path and os.path.exists(path)
        assert open(path).readline().startswith("energy_eV")


# --------------------------------------------------------------------------- #
# a metal — the other side of the band-gap logic
# --------------------------------------------------------------------------- #

ALUMINIUM = Structure(Lattice([[0.0, 2.025, 2.025],
                               [2.025, 0.0, 2.025],
                               [2.025, 2.025, 0.0]]), ["Al"], [[0, 0, 0]])


@pytest.fixture(scope="session")
def metal_run(tmp_path_factory):
    """An fcc aluminium scf — partially filled bands crossing the Fermi level."""
    if _pseudo_set_covering(ALUMINIUM) is None:
        pytest.skip("no pseudopotential set under $ESPRESSO_PSEUDO covers Al")
    working_dir = str(tmp_path_factory.mktemp("qe_metal"))
    C.generate_pw_input_file(
        working_dir, "scf", ALUMINIUM, _pseudo_set_covering(ALUMINIUM),
        ECUTWFC, ECUTRHO, (4, 4, 4), None, False, "al", "scf.in")
    run_qe(working_dir, "pw.x", "scf.in")
    return working_dir


class TestMetal:
    def test_aluminium_is_reported_as_metallic(self, metal_run):
        pwxml = R.parse_qe_outputs(metal_run, "out/al.xml")["pwxml"]
        assert R._band_gap_text(pwxml) == "Metallic (no gap)"
        assert R._band_gap_text(pwxml, short=True) == "Metallic"

    def test_the_summary_still_reports_a_real_energy_for_it(self, metal_run):
        table = R.build_summary_dataframe(R.parse_qe_outputs(metal_run,
                                                             "out/al.xml"))
        values = dict(zip(table["Property"], table["Value"]))
        assert values["Band gap"] == "Metallic (no gap)"
        assert float(values["Final total energy (eV)"]) < 0

    def test_comparing_a_metal_and_a_semiconductor(self, metal_run, scf_run,
                                                   tmp_path):
        """The Compare table's short forms, on two genuinely different runs."""
        scf_dir, _log = scf_run
        out = tmp_path / "out"
        out.mkdir()
        shutil.copy(os.path.join(metal_run, "out", "al.xml"), out / "al.xml")
        shutil.copy(os.path.join(scf_dir, "out", "si.xml"), out / "si.xml")

        table, _figure, message = R.on_compare_runs(str(tmp_path),
                                                    ["out/si.xml", "out/al.xml"])
        gaps = dict(zip(table["Run"], table["Band gap"]))
        assert gaps["al"] == "Metallic"
        assert gaps["si"].endswith(("(d)", "(i)"))
        assert "Compared 2 run(s)" in message


# --------------------------------------------------------------------------- #
# relax — the run wrapper and the relaxed-structure hand-off
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def relax_run(tmp_path_factory):
    """A relax of a displaced silicon, driven through on_run_calculation()."""
    working_dir = str(tmp_path_factory.mktemp("qe_relax"))
    displaced = SILICON.copy()
    displaced.translate_sites([1], [0.02, 0.01, 0.0])
    generate(working_dir, "relax", "si_relax", structure=displaced)

    statuses = [status for status, _log in C.on_run_calculation(
        working_dir, 1, "pw.x", "relax.in", "relax.out", "")]
    return working_dir, statuses[-1]


class TestRelax:
    def test_the_run_wrapper_reports_success(self, relax_run):
        _working_dir, status = relax_run
        assert "color:green" in status
        assert "relax.out" in status

    def test_the_relaxed_structure_is_saved_as_a_cif(self, relax_run):
        working_dir, status = relax_run
        assert "Relaxed structure saved to relax.cif" in status
        saved = Structure.from_file(os.path.join(working_dir, "relax.cif"))
        assert len(saved) == 2
        assert saved.composition.reduced_formula == "Si"

    def test_the_saved_cell_can_seed_the_next_calculation(self, relax_run):
        working_dir, _status = relax_run
        assert "relax.cif" in C.refresh_file_list(working_dir)
        seeded = Structure.from_file(os.path.join(working_dir, "relax.cif"))
        assert os.path.exists(generate(working_dir, "scf", "si_seeded",
                                       structure=seeded))

    def test_the_ionic_steps_plot_as_a_convergence_curve(self, relax_run):
        working_dir, _status = relax_run
        bundle = R.parse_qe_outputs(working_dir, "out/si_relax.xml")
        figure, message = R.build_convergence_plot(bundle)
        assert figure is not None, message
        assert "ionic steps" in message
        energies = figure.axes[0].lines[0].get_ydata()
        assert len(energies) > 1
        assert energies[-1] <= energies[0] + 1e-6      # the relax went downhill
