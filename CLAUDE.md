# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Gradio + FastAPI web GUI for running [Quantum Espresso](https://www.quantum-espresso.org/) DFT calculations (input generation, local execution, result visualization). It is a QE port of the working VASP app kept in `./vasp-webui/` — that directory is the reference template and mirrors the exact same module/function structure.

## Running & environment

Linux/WSL only (uses `mpirun`, `os.killpg`, POSIX process groups). The venv `qe-env/` is committed with all deps installed.

```bash
source qe-env/bin/activate
python3 webui.py          # serves on 127.0.0.1, first free port from 7860
```

There is no build step, linter config, or test suite. To smoke-test changes without QE binaries:

```bash
./qe-env/bin/python -m py_compile working_directory.py calculation.py result.py webui.py
./qe-env/bin/python -c "import calculation, result, working_directory"
```

Input generation and result-parsing functions can be called directly (e.g. `calculation.on_generate_qe_input(...)`, `result.on_load_results(...)`) with a temp working dir — they never raise into Gradio, so they are safe to exercise in a script.

## Architecture

`webui.py` builds a single `gr.Blocks()` mounted onto FastAPI via `gr.mount_gradio_app`. Layout: a left column (working-directory panel) and a right column with a shared status `Markdown` and a three-tab `Tabs` (Calculation, Automation, Result). It imports exactly these entry functions — **keep these names/signatures**, `webui.py` will not start otherwise:

- `working_directory.py::working_directory_blocks()` — left panel; **returns the two shared `gr.State` objects**
- `calculation.py::calculation_tab_content(path_state, file_list_state, status_markdown)`
- `automation.py::automation_tab_content(path_state, file_list_state, status_markdown)`
- `result.py::result_tab_content(path_state, file_list_state, status_markdown)`

`automation.py` (Workflow + Convergence) reuses non-Gradio cores extracted from `calculation.py`: `generate_pw_input_file()` / `write_post_input_file()` (build one input) and `run_qe_stream()` (a generator that runs one `mpirun` command, tees to the `.out`, and registers the process for Stop) plus `preflight_run()`. The single-shot `on_generate_qe_input` / `on_run_calculation` are thin wrappers over these — change the cores, not both call sites. Workflows chain stages purely through the shared `prefix`/`OUTDIR`; convergence gives each point a distinct prefix (`<base>_<param><value>`) so runs coexist in `./out`.

**State flow is the key to the whole app.** Two `gr.State` objects thread through all three modules:
- `working_directory_path_state` — the active `./data/<name>` directory
- `working_directory_file_list_state` — the file list, which acts as the **event hub**: any handler that writes to it (generate, run, upload, delete) triggers cascading `.change()` handlers that rebuild the file table, refresh the Calculation-tab dropdowns, and invalidate the Result tab. To make something react to file changes, subscribe to this state rather than calling across modules.

Generated 3D viewers (nglview) are written as standalone HTML into `./static/` (cleaned on startup) and embedded via `<iframe src="/static/...?ts=...">`; the `?ts=` cache-buster is required or the iframe shows stale content. Working directories live under `./data/`.

## QE-specific design (how it differs from the VASP reference)

VASP-webui generates inputs from pymatgen's curated `MP*Set` input sets. **QE has no equivalent**, so `calculation.py` generates from templates instead:

- `PW_TYPES` (`scf`, `relax`, `vc-relax`, `nscf`, `bands`) → built from a `Structure` via `pymatgen.io.pwscf.PWInput`. The `bands` type uses `HighSymmKpath` to emit a `K_POINTS crystal_b` line-mode path.
- `POST_TYPES` (`dos`, `projwfc`, `bands.x`, `pp.x`) → written from plain-text namelist templates via `build_post_input()`.
- The QE `prefix` (labelled "Output Name" in the UI) is **user-chosen at generation time** (default `pwscf`), written into each input's `&CONTROL`. The input file name and the run log name track the calculation type (`scf` → `scf.in` / `scf.out`); the prefix does not. `CALC_EXECUTABLE` maps each type to its binary so picking `dos` selects `dos.x`.
- Pseudopotentials live under `PSEUDO_ROOT` = `$ESPRESSO_PSEUDO` (falling back to `~/q-e-pseudo`), whose immediate subdirectories are the selectable *sets*. `list_pseudo_sets()` fills the "Pseudopotential Set" dropdown at UI-build time (so a new set needs an app restart), and `resolve_pseudo_dir()` — called once at the top of `generate_pw_input_file()`, the single choke point for both `calculation.py` and `automation.py` — turns the chosen set name into the directory written as `pseudo_dir`. The dropdowns are `allow_custom_value=True`, so a typed absolute path is passed through as-is. Within the set, a UPF is auto-matched per element by comparing the element symbol to the UPF filename stem.
- The XC functional is chosen from the `FUNCTIONALS` dropdown (GGA/meta-GGA/hybrid/LDA + a Custom free-text `input_dft`). A non-default choice writes `input_dft` into `&SYSTEM` (requires QE built with Libxc for meta-GGA/hybrids); meta-GGA (`META_GGA` set) also lowers `mixing_beta` to 0.3 for SCF stability. Applied before the extra-settings merge, so the extra-settings box overrides it.
- Runs are `mpirun -np N <exe> -in <file>` streamed live (the interpolated exe/input names are `shlex.quote`-d); the user picks which `.in` file and which binary to run.

`result.py` parses QE output with `pymatgen.io.espresso.outputs.PWxml`, which subclasses pymatgen's `Vasprun` and is a near drop-in — so the VASP result logic (summary table, `DosPlotter`/`BSPlotter`, trajectory, convergence) carries over. DOS needs a `dos.x`/`projwfc.x` run; band-structure plots need a line-mode `bands` run plus its `.in` file (parsed for k-labels).

**Cross-module coupling to watch:** `OUTDIR` (`./out`) is shared between `calculation.py` and `result.py`. The prefix is *not* hard-coded — generation bakes the user's choice into `&CONTROL`, and the Result tab lists the `out/*.xml` files, letting the user pick one; `result.py::_prefix_from_xml` then infers that file's prefix and uses it to locate the matching `<prefix>.dos` / `.pdos_*` / `.cube`. So a run's XML, log, and auxiliary files all line up only if the same prefix (Output Name) is used for the pw.x run and its post-processing steps.
