"""working_directory.py — the left panel: directory handling, file table, editor."""
import os
import time

import gradio as gr
import pytest

import working_directory as W


def select_event(file_name):
    """The gr.SelectData a Dataframe row click delivers."""
    return gr.SelectData(target=None, data={
        "index": [0, 0], "value": file_name,
        "row_value": [file_name, "Structure file", "Mon Jan  1 00:00:00 2024"]})


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """Run inside a temp cwd so ./data/ is created there and not in the repo."""
    monkeypatch.chdir(tmp_path)
    return tmp_path / "data"


# --------------------------------------------------------------------------- #
# Choosing a working directory
# --------------------------------------------------------------------------- #

class TestGetWorkingDirectories:
    def test_lists_subdirectories_sorted_by_name(self, data_root):
        for name in ("zinc", "Alpha", "beta"):
            (data_root / name).mkdir(parents=True)
        (data_root / "loose.cif").write_text("")
        assert W.get_working_directories() == ["Alpha", "beta", "zinc"]

    def test_creates_the_data_root_when_missing(self, data_root):
        assert W.get_working_directories() == []
        assert data_root.is_dir()

    def test_default_is_the_first_directory(self, data_root):
        (data_root / "NaCl").mkdir(parents=True)
        (data_root / "Ag").mkdir()
        assert W.default_working_directory() == "Ag"

    def test_default_is_none_when_there_are_none(self, data_root):
        assert W.default_working_directory() is None


class TestOnOpenWorkingDirectory:
    def test_creates_the_directory_and_publishes_the_state(self, data_root):
        dropdown, path, files, upload = W.on_open_working_directory("NaCl")
        assert path == os.path.join("./data/", "NaCl")
        assert os.path.isdir(path)
        assert files == []
        assert dropdown["choices"] == ["NaCl"] and dropdown["value"] == "NaCl"
        assert upload["interactive"] is True

    def test_reopening_lists_the_existing_files(self, data_root):
        (data_root / "NaCl").mkdir(parents=True)
        (data_root / "NaCl" / "b.in").write_text("")
        (data_root / "NaCl" / "a.cif").write_text("")
        _dropdown, _path, files, _upload = W.on_open_working_directory("NaCl")
        assert files == ["a.cif", "b.in"]

    @pytest.mark.parametrize("name", [None, "", "   ", "../escape", "sub/dir", ".."])
    def test_rejects_empty_and_traversing_names(self, name, data_root, tmp_path):
        with pytest.warns(UserWarning, match="working directory name"):
            dropdown, path, files, upload = W.on_open_working_directory(name)
        assert (path, files) == (None, None)
        assert dropdown == gr.update() and upload == gr.update()
        assert not (tmp_path / "escape").exists()
        assert not data_root.exists() or list(data_root.iterdir()) == []

    def test_surrounding_whitespace_is_trimmed(self, data_root):
        _dropdown, path, _files, _upload = W.on_open_working_directory("  NaCl  ")
        assert path == os.path.join("./data/", "NaCl")


# --------------------------------------------------------------------------- #
# The file table
# --------------------------------------------------------------------------- #

class TestClassifyFile:
    @pytest.mark.parametrize("name,expected", [
        ("NaCl.cif", "Structure file"),
        ("POSCAR", "Structure file"),
        ("CONTCAR", "Structure file"),
        ("cell.vasp", "Structure file"),
        ("Na.upf", "Pseudopotential"),
        ("Na.UPF", "Pseudopotential"),
        ("scf.in", "Input File - pw.x/pp.x parameters"),
        ("scf.pwi", "Input File - pw.x/pp.x parameters"),
        ("pwscf.xml", "Output File - Summary (QE XML)"),
        ("pwscf.pdos_atm#1(Na)_wfc#1(s)", "Output File - Projected DOS"),
        ("pwscf.projwfc_up", "Output File - Projected DOS"),
        ("pwscf.dos", "Output File - Density of States"),
        ("rho.cube", "Output File - Volumetric data (cube)"),
        ("scf.out", "Output File - Log"),
        ("run.log", "Output File - Log"),
        ("notes.md", "Other File"),
    ])
    def test_labels(self, name, expected):
        assert W._classify_file(name) == expected


class TestOnFileListChange:
    def test_newest_files_come_first(self, working_dir):
        for index, name in enumerate(("old.in", "middle.in", "new.in")):
            path = os.path.join(working_dir, name)
            open(path, "w").close()
            os.utime(path, (time.time() + index, time.time() + index))
        table = W.on_file_list_change(working_dir)
        assert list(table.columns) == ["File", "Type", "Modified"]
        assert table["File"].tolist() == ["new.in", "middle.in", "old.in"]

    def test_each_row_carries_a_type_and_a_readable_timestamp(self, working_dir):
        open(os.path.join(working_dir, "scf.in"), "w").close()
        row = W.on_file_list_change(working_dir).iloc[0]
        assert row["Type"] == "Input File - pw.x/pp.x parameters"
        assert time.strptime(row["Modified"])          # parses as a ctime string

    def test_a_file_removed_mid_scan_is_skipped(self, working_dir, monkeypatch):
        for name in ("keep.in", "vanishing.in"):
            open(os.path.join(working_dir, name), "w").close()
        real_getmtime = os.path.getmtime

        def flaky(path):
            if path.endswith("vanishing.in"):
                raise FileNotFoundError(path)
            return real_getmtime(path)
        monkeypatch.setattr(W.os.path, "getmtime", flaky)
        assert W.on_file_list_change(working_dir)["File"].tolist() == ["keep.in"]

    def test_empty_directory_yields_an_empty_table(self, working_dir):
        table = W.on_file_list_change(working_dir)
        assert table.empty and list(table.columns) == ["File", "Type", "Modified"]


class TestOnSelectFile:
    def test_a_structure_file_enables_both_viewers(self):
        name, structure, text, delete = W.on_select_file(select_event("NaCl.cif"))
        assert name == "NaCl.cif"
        assert structure == "NaCl.cif" and text == "NaCl.cif"   # .cif is both
        assert delete["interactive"] is True

    def test_an_input_file_is_text_only(self):
        _n, structure, text, _d = W.on_select_file(select_event("scf.in"))
        assert structure is None and text == "scf.in"

    def test_a_poscar_is_structure_only(self):
        _n, structure, text, _d = W.on_select_file(select_event("POSCAR"))
        assert structure == "POSCAR" and text is None

    def test_an_unknown_file_enables_neither(self):
        _n, structure, text, delete = W.on_select_file(select_event("pwscf.wfc1"))
        assert structure is None and text is None
        assert delete["interactive"] is True            # deleting is always allowed

    @pytest.mark.parametrize("state,expected", [("scf.in", True), (None, False)])
    def test_viewer_buttons_follow_the_selection_state(self, state, expected):
        assert W.on_selected_structure_file_state_change(state)["interactive"] is expected
        assert W.on_selected_text_file_state_change(state)["interactive"] is expected


# --------------------------------------------------------------------------- #
# File actions
# --------------------------------------------------------------------------- #

class TestFileActions:
    def test_upload_copies_into_the_working_directory(self, working_dir, tmp_path):
        source = tmp_path / "upload.cif"
        source.write_text("data_x\n")
        files = W.on_upload_file(working_dir, str(source))
        assert files == ["upload.cif"]
        assert open(os.path.join(working_dir, "upload.cif")).read() == "data_x\n"

    def test_a_failed_upload_warns_instead_of_raising(self, working_dir, tmp_path):
        with pytest.warns(UserWarning, match="Error uploading file"):
            assert W.on_upload_file(working_dir, str(tmp_path / "missing.cif")) == []

    def test_delete_removes_the_file(self, working_dir):
        open(os.path.join(working_dir, "scf.in"), "w").close()
        open(os.path.join(working_dir, "keep.cif"), "w").close()
        with pytest.warns(UserWarning, match="File deleted successfully"):
            assert W.on_delete_file(working_dir, "scf.in") == ["keep.cif"]

    def test_delete_without_a_selection_is_a_no_op(self, working_dir):
        open(os.path.join(working_dir, "keep.cif"), "w").close()
        assert W.on_delete_file(working_dir, None) == ["keep.cif"]

    def test_deleting_a_missing_file_warns_instead_of_raising(self, working_dir):
        with pytest.warns(UserWarning, match="Error deleting file"):
            assert W.on_delete_file(working_dir, "gone.in") == []

    def test_view_then_save_round_trips(self, working_dir):
        path = os.path.join(working_dir, "scf.in")
        with open(path, "w") as fh:
            fh.write("&CONTROL\n/\n")

        viewer, save_button = W.on_view_text_file(working_dir, "scf.in")
        assert viewer["value"] == "&CONTROL\n/\n"
        assert "scf.in" in viewer["label"] and save_button["interactive"] is True

        with pytest.warns(UserWarning, match="File saved successfully"):
            W.on_save_text_file(working_dir, "scf.in",
                                "&CONTROL\n calculation = 'scf'\n/\n")
        assert "calculation = 'scf'" in open(path).read()

    def test_viewing_without_a_selection_warns_instead_of_raising(self, working_dir):
        with pytest.warns(UserWarning, match="select a text file"):
            viewer, save_button = W.on_view_text_file(working_dir, None)
        assert viewer == gr.update() and save_button == gr.update()

    def test_viewing_a_missing_file_warns_instead_of_raising(self, working_dir):
        with pytest.warns(UserWarning, match="No such file"):
            viewer, _save = W.on_view_text_file(working_dir, "gone.in")
        assert viewer == gr.update()

    def test_saving_without_a_selection_is_a_no_op(self, working_dir):
        open(os.path.join(working_dir, "a.cif"), "w").close()
        with pytest.warns(UserWarning, match="select a text file"):
            assert W.on_save_text_file(working_dir, None, "text") == ["a.cif"]


class TestOnViewStructureFile:
    @pytest.mark.parametrize("name", [None, ""])
    def test_nothing_selected(self, working_dir, name):
        assert W.on_view_structure_file(working_dir, name, 1, 1, 1) is None

    def test_an_unreadable_structure_warns_instead_of_raising(self, working_dir):
        with open(os.path.join(working_dir, "broken.cif"), "w") as fh:
            fh.write("not a cif\n")
        with pytest.warns(UserWarning, match="Invalid CIF"):
            assert W.on_view_structure_file(working_dir, "broken.cif", 1, 1, 1) is None

    def test_renders_an_iframe_with_a_cache_buster(self, working_dir, structure_file,
                                                   monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "static").mkdir()
        html = W.on_view_structure_file(working_dir, structure_file, 2, 1, 1)
        assert html is not None
        assert 'src="/static/input_structure.html?ts=' in html
        assert (tmp_path / "static" / "input_structure.html").exists()
