"""utils.py — the file list every dropdown is built from, and name validation."""
import os

import pytest

from utils import get_files_in_working_directory, sort_by_name, validate_name


class TestSortByName:
    def test_sorts_case_insensitively(self):
        assert sort_by_name(["b.in", "A.cif", "a.out"]) == ["A.cif", "a.out", "b.in"]

    def test_is_stable_for_names_differing_only_in_case(self):
        assert sort_by_name(["Na.upf", "na.upf"]) == ["Na.upf", "na.upf"]

    def test_accepts_any_iterable(self):
        assert sort_by_name(f for f in ["b", "a"]) == ["a", "b"]

    def test_empty(self):
        assert sort_by_name([]) == []


class TestGetFilesInWorkingDirectory:
    def test_returns_names_sorted_by_name(self, working_dir):
        for name in ("scf.out", "Band.in", "alpha.cif", "beta.in"):
            open(os.path.join(working_dir, name), "w").close()
        assert get_files_in_working_directory(working_dir) == [
            "alpha.cif", "Band.in", "beta.in", "scf.out"]

    def test_skips_wsl_zone_identifier_files(self, working_dir):
        open(os.path.join(working_dir, "a.cif"), "w").close()
        open(os.path.join(working_dir, "a.cif:Zone.Identifier"), "w").close()
        assert get_files_in_working_directory(working_dir) == ["a.cif"]

    def test_empty_directory(self, working_dir):
        assert get_files_in_working_directory(working_dir) == []

    @pytest.mark.parametrize("path", [None, "", "/no/such/directory"])
    def test_missing_path_never_lists_the_server_cwd(self, path):
        assert get_files_in_working_directory(path) == []

    def test_a_file_path_is_not_listed_as_a_directory(self, working_dir):
        file_path = os.path.join(working_dir, "a.cif")
        open(file_path, "w").close()
        assert get_files_in_working_directory(file_path) == []


class TestValidateName:
    @pytest.mark.parametrize("name", ["scf.in", "pwscf", "my run_1.in", "a.b.c"])
    def test_accepts_ordinary_names(self, name):
        assert validate_name(name, "input file name") is None

    @pytest.mark.parametrize("name", [None, "", "   "])
    def test_rejects_empty(self, name):
        assert validate_name(name, "output name") == "Please provide a output name."

    @pytest.mark.parametrize("name", [
        "../evil", "sub/dir", "back\\slash", "a:b", "star*", "quest?",
        "quo\"te", "pipe|d", "lt<gt>", "ctrl\x01", ".", "..",
    ])
    def test_rejects_separators_and_control_characters(self, name):
        error = validate_name(name, "working directory name")
        assert error is not None
        assert "working directory name" in error

    def test_surrounding_whitespace_is_ignored(self):
        assert validate_name("  scf.in  ", "input file name") is None
