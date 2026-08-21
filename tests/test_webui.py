"""webui.py — the Blocks actually assemble, and the module contract holds.

Building the app is the cheapest guard against the mistake this layout invites:
an event wired to a component or handler that does not line up. Everything is
built in a temp cwd so ./static and ./data of the real project are untouched.
"""
import inspect
import os
import shutil

import gradio as gr
import pytest

import automation
import calculation
import result
import working_directory

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def app_cwd(tmp_path, monkeypatch):
    """A throwaway copy of the files webui.py reads relative to the cwd."""
    shutil.copy(os.path.join(REPO_ROOT, "styles.css"), tmp_path / "styles.css")
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestModuleContract:
    """webui.py imports these four entry points by name; keep the signatures."""

    def test_working_directory_panel_returns_the_two_shared_states(self):
        with gr.Blocks():
            with gr.Row():
                states = working_directory.working_directory_blocks()
        assert len(states) == 2
        assert all(isinstance(state, gr.State) for state in states)

    @pytest.mark.parametrize("module,name", [
        (calculation, "calculation_tab_content"),
        (automation, "automation_tab_content"),
        (result, "result_tab_content"),
    ])
    def test_each_tab_takes_the_two_states_and_the_status_markdown(self, module, name):
        parameters = list(inspect.signature(getattr(module, name)).parameters)
        assert parameters == ["working_directory_path_state",
                              "working_directory_file_list_state",
                              "status_markdown"]

    @pytest.mark.parametrize("module,name", [
        (calculation, "calculation_tab_content"),
        (automation, "automation_tab_content"),
        (result, "result_tab_content"),
    ])
    def test_each_tab_builds_and_returns_its_tab(self, module, name, app_cwd):
        with gr.Blocks():
            path_state = gr.State()
            file_list_state = gr.State()
            status = gr.Markdown()
            tab = getattr(module, name)(path_state, file_list_state, status)
        assert isinstance(tab, gr.Tab)


class TestAppBuilds:
    def test_the_whole_app_assembles(self, app_cwd):
        import webui        # importing builds the Blocks and mounts them
        assert webui.blocks.fns, "no event handlers were registered"
        assert isinstance(webui.blocks, gr.Blocks)

    def test_the_static_directory_is_prepared_and_cleaned(self, app_cwd):
        static = app_cwd / "static"
        static.mkdir()
        stale = static / "input_structure.html"
        stale.write_text("<html>stale</html>")
        kept = static / "pwscf_dos.csv"
        kept.write_text("energy_eV\n")

        import importlib
        import webui
        importlib.reload(webui)

        assert static.is_dir()
        assert not stale.exists()      # generated viewers are dropped on startup
        assert kept.exists()

    def test_an_available_port_is_found(self, app_cwd):
        import webui
        assert 7860 <= webui.find_available_port() < 7960

    def test_a_busy_port_is_skipped(self, app_cwd):
        import socket

        import webui
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
            taken.bind(("localhost", 0))
            busy = taken.getsockname()[1]
            assert webui.find_available_port(busy) > busy
