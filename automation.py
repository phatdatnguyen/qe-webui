import os
import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend for the web server
import matplotlib.pyplot as plt
import gradio as gr
from pymatgen.core import Structure
from pymatgen.io.espresso.outputs import PWxml

import calculation as C
from utils import get_files_in_working_directory

# QE writes <prefix>.xml under this subdirectory of the working dir (see OUTDIR).
OUT_SUBDIR = "out"

# Ordered multi-step pipelines. Each stage is (calc_type, executable). Stages
# chain through the shared QE prefix/outdir (nscf/dos/bands reuse the scf charge
# density automatically), so we only need to generate + run them in order.
WORKFLOWS = {
    "Relax → SCF": [("relax", "pw.x"), ("scf", "pw.x")],
    "SCF → DOS": [("scf", "pw.x"), ("nscf", "pw.x"), ("dos", "dos.x")],
    "SCF → PDOS (projwfc)": [("scf", "pw.x"), ("nscf", "pw.x"), ("projwfc", "projwfc.x")],
    "SCF → Band structure": [("scf", "pw.x"), ("bands", "pw.x"), ("bands.x", "bands.x")],
}

# Cap the number of convergence points so a tiny step can't spawn a huge sweep.
MAX_CONV_POINTS = 20


def on_file_list_change_structures(working_directory_file_list):
    """Refresh both structure dropdowns when the working-dir file list changes."""
    files = working_directory_file_list or []
    structure_files = [f for f in files
                       if f.endswith((".cif", ".vasp", "POSCAR", "CONTCAR"))]
    update = gr.update(choices=structure_files,
                       value=structure_files[0] if structure_files else None)
    return update, gr.update(choices=structure_files,
                             value=structure_files[0] if structure_files else None)


# --------------------------------------------------------------------------- #
# Workflow
# --------------------------------------------------------------------------- #

def on_run_workflow(working_directory_path, workflow_type, structure_file, pseudo_set,
                    ecutwfc, ecutrho, kx, ky, kz, functional, custom_functional,
                    output_name, num_cores, qe_bin_dir):
    try:
        if not working_directory_path:
            yield "<p style='color:red'>No working directory is open.</p>", ""
            return
        if not structure_file:
            yield "<p style='color:red'>Please select an input structure file.</p>", ""
            return
        err = C.validate_name(output_name, "output name")
        if err:
            yield f"<p style='color:red'>{err}</p>", ""
            return
        prefix = output_name.strip()

        stages = WORKFLOWS.get(workflow_type)
        if not stages:
            yield f"<p style='color:red'>Unknown workflow: {workflow_type!r}</p>", ""
            return

        structure = Structure.from_file(os.path.join(working_directory_path, structure_file))
        input_dft, is_metagga = C.resolve_functional(functional, custom_functional)

        full_log = ""
        n = len(stages)
        for i, (calc_type, exe_name) in enumerate(stages, 1):
            input_name = C.default_input_name(calc_type)
            out_name = C.default_output_name(input_name)
            full_log += f"\n===== [{i}/{n}] {calc_type} ({exe_name}) =====\n"
            yield f"<p>Workflow '{workflow_type}': generating {input_name} [{i}/{n}]...</p>", full_log

            # Generate this stage's input (shared prefix chains the stages).
            if calc_type in C.POST_TYPES:
                C.write_post_input_file(working_directory_path, calc_type, prefix, input_name)
            else:
                C.generate_pw_input_file(
                    working_directory_path, calc_type, structure, pseudo_set,
                    ecutwfc, ecutrho, (kx, ky, kz), input_dft, is_metagga, prefix, input_name)

            exe, preflight_err = C.preflight_run(working_directory_path, exe_name, input_name, qe_bin_dir)
            if preflight_err:
                yield preflight_err, full_log
                return

            stage_rc, final_log = None, ""
            for log, rc in C.run_qe_stream(working_directory_path, num_cores, exe, input_name, out_name):
                stage_rc, final_log = rc, log
                yield (f"<p>Workflow '{workflow_type}': running {calc_type} "
                       f"({exe_name}) [{i}/{n}]...</p>", full_log + log)
            full_log += final_log

            if stage_rc != 0:
                sig = f" (stopped, signal {-stage_rc})" if stage_rc is not None and stage_rc < 0 else ""
                yield (f"<p style='color:red'>Workflow stopped: {calc_type} ({exe_name}) "
                       f"exited with code {stage_rc}{sig}.</p>", full_log)
                return

            # Save the relaxed structure now, before a later stage (same prefix)
            # overwrites this stage's XML.
            if calc_type in C.RELAX_TYPES:
                cif = C.write_relaxed_cif(working_directory_path, input_name, out_name)
                if cif:
                    full_log += f"\n[Saved relaxed structure: {cif}]\n"

        yield (f"<p style='color:green'>Workflow '{workflow_type}' finished successfully "
               f"({n} stages, prefix '{prefix}').</p>", full_log)

    except Exception as e:
        C._current_process = None
        yield f"<p style='color:red'>Workflow error: {e}</p>", None


# --------------------------------------------------------------------------- #
# Convergence testing
# --------------------------------------------------------------------------- #

def _conv_values(param, start, stop, step):
    """Build the list of parameter values to scan (inclusive of stop, capped)."""
    start, stop, step = float(start), float(stop), float(step)
    if step <= 0:
        raise Exception("Step must be greater than 0.")
    if stop < start:
        raise Exception("Stop must be ≥ start.")
    values, v = [], start
    while v <= stop + 1e-9 and len(values) < MAX_CONV_POINTS:
        if param == "k-grid":
            values.append(int(round(v)))
        else:
            fv = round(v, 6)
            values.append(int(fv) if float(fv).is_integer() else fv)  # 30.0 -> 30
        v += step
    return values


def on_run_convergence(working_directory_path, structure_file, pseudo_set, ecutwfc, ecutrho,
                       kx, ky, kz, functional, custom_functional, output_name,
                       num_cores, qe_bin_dir, param, start, stop, step):
    empty_df = pd.DataFrame(columns=["Value", "Total energy (eV)", "ΔE vs previous (eV)"])
    try:
        if not working_directory_path:
            yield "<p style='color:red'>No working directory is open.</p>", "", None, empty_df
            return
        if not structure_file:
            yield "<p style='color:red'>Please select an input structure file.</p>", "", None, empty_df
            return
        err = C.validate_name(output_name, "output name")
        if err:
            yield f"<p style='color:red'>{err}</p>", "", None, empty_df
            return
        base_prefix = output_name.strip()

        values = _conv_values(param, start, stop, step)
        if not values:
            yield "<p style='color:red'>No values to scan.</p>", "", None, empty_df
            return

        structure = Structure.from_file(os.path.join(working_directory_path, structure_file))
        input_dft, is_metagga = C.resolve_functional(functional, custom_functional)
        ratio = (float(ecutrho) / float(ecutwfc)) if float(ecutwfc) else 4.0

        warn = ""
        if ratio < 4.0:
            warn = " <span style='color:orange'>(note: ecutrho/ecutwfc &lt; 4)</span>"

        full_log, results = "", []
        n = len(values)
        tag_param = "k" if param == "k-grid" else param
        for i, v in enumerate(values, 1):
            tag = f"{tag_param}{v}".replace(" ", "").replace(".", "p")
            run_prefix = f"{base_prefix}_{tag}"
            input_name = f"{run_prefix}.in"
            out_name = f"{run_prefix}.out"

            if param == "ecutwfc":
                this_ecutwfc, this_ecutrho, kgrid = v, ratio * v, (kx, ky, kz)
            else:  # k-grid
                this_ecutwfc, this_ecutrho, kgrid = ecutwfc, ecutrho, (v, v, v)

            full_log += f"\n===== [{i}/{n}] {param} = {v} =====\n"
            yield (f"<p>Convergence: {param} = {v} [{i}/{n}]...{warn}</p>", full_log, gr.update(), gr.update())

            C.generate_pw_input_file(
                working_directory_path, "scf", structure, pseudo_set,
                this_ecutwfc, this_ecutrho, kgrid, input_dft, is_metagga, run_prefix, input_name)

            exe, preflight_err = C.preflight_run(working_directory_path, "pw.x", input_name, qe_bin_dir)
            if preflight_err:
                yield preflight_err, full_log, gr.update(), gr.update()
                return

            stage_rc, final_log = None, ""
            for log, rc in C.run_qe_stream(working_directory_path, num_cores, exe, input_name, out_name):
                stage_rc, final_log = rc, log
                yield (f"<p>Convergence: running {param} = {v} [{i}/{n}]...</p>",
                       full_log + log, gr.update(), gr.update())
            full_log += final_log

            if stage_rc != 0:
                yield (f"<p style='color:red'>Convergence stopped at {param} = {v} "
                       f"(exit code {stage_rc}).</p>", full_log, gr.update(), gr.update())
                return

            energy = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    energy = PWxml(os.path.join(working_directory_path, OUT_SUBDIR,
                                                f"{run_prefix}.xml")).final_energy
            except Exception:
                energy = None
            results.append((v, energy))

        fig, df = _convergence_outputs(param, results)
        yield (f"<p style='color:green'>Convergence scan finished ({n} points).{warn}</p>",
               full_log, fig, df)

    except Exception as e:
        C._current_process = None
        yield f"<p style='color:red'>Convergence error: {e}</p>", "", None, empty_df


def _convergence_outputs(param, results):
    """Build the (figure, dataframe) from a list of (value, energy) pairs."""
    xlabel = "ecutwfc (Ry)" if param == "ecutwfc" else "k-grid (n×n×n)"
    rows, prev = [], None
    for v, e in results:
        de = "" if (e is None or prev is None) else f"{e - prev:.6f}"
        rows.append([v, f"{e:.6f}" if e is not None else "n/a", de])
        if e is not None:
            prev = e
    df = pd.DataFrame(rows, columns=["Value", "Total energy (eV)", "ΔE vs previous (eV)"])

    pts = [(v, e) for v, e in results if e is not None]
    if len(pts) < 1:
        return None, df
    xs, ys = zip(*pts)
    fig, ax = plt.subplots()
    ax.plot(xs, ys, marker="o", markersize=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Total energy (eV)")
    ax.set_title("Convergence of total energy")
    fig.tight_layout()
    return fig, df


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

def automation_tab_content(working_directory_path_state, working_directory_file_list_state, status_markdown):
    num_cores = max(1, (os.cpu_count() or 2) // 2)
    functional_choices = list(C.FUNCTIONALS.keys())

    with gr.Tab("Automation") as automation_tab:
        with gr.Accordion("Workflow (multi-step)", open=True):
            with gr.Row():
                with gr.Column(scale=1):
                    wf_type = gr.Dropdown(choices=list(WORKFLOWS.keys()),
                                          value="SCF → DOS", label="Workflow")
                    wf_structure = gr.Dropdown(choices=[], value=None, label="Input Structure")
                    wf_pseudo_set = gr.Dropdown(
                        choices=C.list_pseudo_sets(), value=C.default_pseudo_set(),
                        label="Pseudopotential Set", allow_custom_value=True,
                        info=f"Subfolders of $ESPRESSO_PSEUDO ({C.PSEUDO_ROOT})")
                    wf_output_name = gr.Textbox(value=C.DEFAULT_PREFIX, label="Output Name (QE prefix)")
                with gr.Column(scale=1):
                    wf_ecutwfc = gr.Slider(minimum=20, maximum=120, step=5, value=50, label="ecutwfc (Ry)")
                    wf_ecutrho = gr.Slider(minimum=80, maximum=960, step=20, value=400, label="ecutrho (Ry)")
                    with gr.Row():
                        wf_kx = gr.Slider(minimum=1, maximum=16, step=1, value=4, label="k x")
                        wf_ky = gr.Slider(minimum=1, maximum=16, step=1, value=4, label="k y")
                        wf_kz = gr.Slider(minimum=1, maximum=16, step=1, value=4, label="k z")
            with gr.Row():
                wf_functional = gr.Dropdown(choices=functional_choices, value=C.DEFAULT_FUNCTIONAL, label="Functional (XC)")
                wf_custom_functional = gr.Textbox(value="", label="Custom input_dft (if Custom)")
            with gr.Row():
                wf_cores = gr.Slider(minimum=1, maximum=num_cores, step=1, value=1, label="Number of Cores")
                wf_bin_dir = gr.Textbox(label="QE binary directory (optional)", placeholder="blank = PATH")
            with gr.Row():
                wf_run_button = gr.Button("Run Workflow", variant="primary")
                wf_stop_button = gr.Button("Stop", variant="stop")
            wf_log = gr.Textbox(label="Workflow Log", lines=12)

        with gr.Accordion("Convergence testing", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    cv_structure = gr.Dropdown(choices=[], value=None, label="Input Structure")
                    cv_pseudo_set = gr.Dropdown(
                        choices=C.list_pseudo_sets(), value=C.default_pseudo_set(),
                        label="Pseudopotential Set", allow_custom_value=True,
                        info=f"Subfolders of $ESPRESSO_PSEUDO ({C.PSEUDO_ROOT})")
                    cv_output_name = gr.Textbox(value="conv", label="Output Name (QE prefix base)")
                with gr.Column(scale=1):
                    cv_ecutwfc = gr.Slider(minimum=20, maximum=120, step=5, value=50, label="ecutwfc (Ry) — base")
                    cv_ecutrho = gr.Slider(minimum=80, maximum=960, step=20, value=400, label="ecutrho (Ry) — base")
                    with gr.Row():
                        cv_kx = gr.Slider(minimum=1, maximum=16, step=1, value=4, label="k x")
                        cv_ky = gr.Slider(minimum=1, maximum=16, step=1, value=4, label="k y")
                        cv_kz = gr.Slider(minimum=1, maximum=16, step=1, value=4, label="k z")
            with gr.Row():
                cv_functional = gr.Dropdown(choices=functional_choices, value=C.DEFAULT_FUNCTIONAL, label="Functional (XC)")
                cv_custom_functional = gr.Textbox(value="", label="Custom input_dft (if Custom)")
            with gr.Row():
                cv_param = gr.Dropdown(choices=["ecutwfc", "k-grid"], value="ecutwfc", label="Parameter to converge")
                cv_start = gr.Number(value=30, label="Start")
                cv_stop = gr.Number(value=70, label="Stop")
                cv_step = gr.Number(value=10, label="Step")
            with gr.Row():
                cv_cores = gr.Slider(minimum=1, maximum=num_cores, step=1, value=1, label="Number of Cores")
                cv_bin_dir = gr.Textbox(label="QE binary directory (optional)", placeholder="blank = PATH")
            with gr.Row():
                cv_run_button = gr.Button("Run Convergence Scan", variant="primary")
                cv_stop_button = gr.Button("Stop", variant="stop")
            cv_log = gr.Textbox(label="Convergence Log", lines=8)
            cv_plot = gr.Plot(label="Energy vs parameter")
            cv_table = gr.Dataframe(headers=["Value", "Total energy (eV)", "ΔE vs previous (eV)"],
                                    wrap=True, interactive=False)

    # Keep both structure dropdowns in sync with the working-dir file list.
    working_directory_file_list_state.change(
        on_file_list_change_structures, [working_directory_file_list_state],
        [wf_structure, cv_structure])

    wf_event = wf_run_button.click(
        on_run_workflow,
        [working_directory_path_state, wf_type, wf_structure, wf_pseudo_set,
         wf_ecutwfc, wf_ecutrho, wf_kx, wf_ky, wf_kz, wf_functional, wf_custom_functional,
         wf_output_name, wf_cores, wf_bin_dir],
        [status_markdown, wf_log])
    wf_event.then(C.refresh_file_list, working_directory_path_state, working_directory_file_list_state)
    wf_stop_event = wf_stop_button.click(C.on_stop_calculation, None, status_markdown, cancels=[wf_event])
    wf_stop_event.then(C.refresh_file_list, working_directory_path_state, working_directory_file_list_state)

    cv_event = cv_run_button.click(
        on_run_convergence,
        [working_directory_path_state, cv_structure, cv_pseudo_set, cv_ecutwfc, cv_ecutrho,
         cv_kx, cv_ky, cv_kz, cv_functional, cv_custom_functional, cv_output_name,
         cv_cores, cv_bin_dir, cv_param, cv_start, cv_stop, cv_step],
        [status_markdown, cv_log, cv_plot, cv_table])
    cv_event.then(C.refresh_file_list, working_directory_path_state, working_directory_file_list_state)
    cv_stop_event = cv_stop_button.click(C.on_stop_calculation, None, status_markdown, cancels=[cv_event])
    cv_stop_event.then(C.refresh_file_list, working_directory_path_state, working_directory_file_list_state)

    return automation_tab
