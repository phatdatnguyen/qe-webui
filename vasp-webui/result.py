import os
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend for the web server
import matplotlib.pyplot as plt
import gradio as gr
import nglview
from pymatgen.io.vasp.outputs import Vasprun, Oszicar, Outcar, Xdatcar, Chgcar
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.electronic_structure.plotter import DosPlotter, BSPlotter, BSDOSPlotter

# Trajectory frames are subsampled to at most this many to keep the
# generated HTML small and the viewer responsive for long MD runs.
MAX_TRAJECTORY_FRAMES = 100


def parse_vasp_outputs(working_directory_path):
    """Parse the main VASP output files once, each guarded independently.

    Returns a bundle dict {vasprun, oszicar, outcar, path, errors}. Any file
    that is missing or unparseable is left as None and noted in errors.
    """
    bundle = {"vasprun": None, "oszicar": None, "outcar": None,
              "path": working_directory_path, "errors": []}

    if not working_directory_path:
        bundle["errors"].append("No working directory is open.")
        return bundle

    vasprun_path = os.path.join(working_directory_path, "vasprun.xml")
    try:
        bundle["vasprun"] = Vasprun(str(vasprun_path), parse_potcar_file=False)
    except Exception as exc:
        bundle["errors"].append(f"vasprun.xml: {exc}")

    oszicar_path = os.path.join(working_directory_path, "OSZICAR")
    try:
        bundle["oszicar"] = Oszicar(str(oszicar_path))
    except Exception as exc:
        bundle["errors"].append(f"OSZICAR: {exc}")

    outcar_path = os.path.join(working_directory_path, "OUTCAR")
    try:
        bundle["outcar"] = Outcar(str(outcar_path))
    except Exception as exc:
        bundle["errors"].append(f"OUTCAR: {exc}")

    return bundle


def build_summary_dataframe(bundle):
    """Build a Property/Value table of key scalars, degrading per-row."""
    vasprun = bundle["vasprun"]
    outcar = bundle["outcar"]
    oszicar = bundle["oszicar"]

    rows = []

    def add(label, fn):
        try:
            rows.append([label, fn()])
        except Exception:
            rows.append([label, "n/a"])

    if vasprun is None:
        # Fall back to OUTCAR/OSZICAR so an unparseable vasprun.xml (e.g. an
        # interrupted run) still yields the scalars those files carry.
        if outcar is None and oszicar is None:
            return pd.DataFrame([["Status", "vasprun.xml not found or unparseable"]],
                                columns=["Property", "Value"])
        rows.append(["Source", "OUTCAR/OSZICAR (vasprun.xml unavailable)"])
        add("Final total energy (eV)",
            lambda: f"{(outcar.final_energy if outcar is not None else oszicar.final_energy):.6f}")
        add("Total magnetization",
            lambda: (f"{outcar.total_mag:.4f} μB"
                     if outcar is not None and outcar.total_mag is not None else "n/a"))
        add("Fermi level",
            lambda: (f"{outcar.efermi:.4f} eV"
                     if outcar is not None and outcar.efermi is not None else "n/a"))
        add("Ionic steps",
            lambda: str(len(oszicar.ionic_steps)) if oszicar is not None else "n/a")
        return pd.DataFrame(rows, columns=["Property", "Value"])

    add("Final total energy (eV)", lambda: f"{vasprun.final_energy:.6f}")

    def band_gap():
        gap, cbm, vbm, is_direct = vasprun.eigenvalue_band_properties
        if gap is None or gap <= 0.01:
            return "Metallic (no gap)"
        return f"{gap:.3f} eV ({'direct' if is_direct else 'indirect'})"
    add("Band gap", band_gap)

    def fermi():
        ef = (outcar.efermi if outcar is not None and outcar.efermi is not None
              else vasprun.efermi)
        return f"{ef:.4f} eV" if ef is not None else "n/a"
    add("Fermi level", fermi)

    add("Converged (electronic)", lambda: str(vasprun.converged_electronic))
    add("Converged (ionic)", lambda: str(vasprun.converged_ionic))

    def magnetization():
        if outcar is not None and outcar.total_mag is not None:
            return f"{outcar.total_mag:.4f} μB"
        return "n/a"
    add("Total magnetization", magnetization)

    add("Run type", lambda: str(vasprun.run_type))
    add("Formula", lambda: vasprun.final_structure.composition.reduced_formula)
    add("Number of sites", lambda: str(len(vasprun.final_structure)))
    add("Ionic steps", lambda: str(len(vasprun.ionic_steps)))

    def wall_time():
        if outcar is None or not outcar.run_stats:
            return "n/a"
        elapsed = outcar.run_stats.get("Elapsed time (sec)")
        cores = outcar.run_stats.get("cores")
        if elapsed is None:
            return "n/a"
        text = f"{elapsed:.1f} s"
        if cores:
            text += f" ({int(cores)} cores)"
        return text
    add("Wall time", wall_time)

    return pd.DataFrame(rows, columns=["Property", "Value"])


def build_dos_plot(bundle):
    """Total + per-element projected DOS. Returns (figure|None, message)."""
    vasprun = bundle["vasprun"]
    if vasprun is None:
        return None, "vasprun.xml not available."
    try:
        dos = vasprun.complete_dos
        plotter = DosPlotter(zero_at_efermi=True, stack=False, sigma=0.05)
        plotter.add_dos("Total", dos)
        try:
            element_dos = {str(el): d for el, d in dos.get_element_dos().items()}
            plotter.add_dos_dict(element_dos)
        except Exception:
            pass  # total DOS still useful if projection is unavailable
        ax = plotter.get_plot()
        return ax.figure, "Total and per-element projected DOS (energy relative to E_F)."
    except Exception as exc:
        return None, f"DOS not available for this run: {exc}"


def build_band_structure_plot(bundle):
    """Band structure, but only for line-mode runs. Returns (figure|None, message)."""
    vasprun = bundle["vasprun"]
    if vasprun is None:
        return None, "vasprun.xml not available."

    kpoints = getattr(vasprun, "kpoints", None)
    is_line_mode = kpoints is not None and getattr(kpoints.style, "name", "") == "Line_mode"
    if not is_line_mode:
        return None, ("Band-structure plot requires a line-mode KPOINTS run; "
                      "this run uses a Γ/Monkhorst grid mesh. See the Summary "
                      "band gap and the DOS instead.")
    try:
        bs = vasprun.get_band_structure(line_mode=True)
        try:
            dos = vasprun.complete_dos
            result = BSDOSPlotter().get_plot(bs, dos)
            ax = result[0] if isinstance(result, (tuple, list)) else result
        except Exception:
            ax = BSPlotter(bs).get_plot()
        return ax.figure, "Band structure along the k-path."
    except Exception as exc:
        return None, f"Could not plot band structure: {exc}"


def build_convergence_plot(bundle):
    """Energy convergence. Returns (figure|None, message).

    Multi-step runs -> energy vs ionic step; single-point -> |dE| vs SCF step.
    """
    oszicar = bundle["oszicar"]
    if oszicar is None:
        return None, "OSZICAR not found — convergence unavailable."

    try:
        ionic_steps = oszicar.ionic_steps or []
        if len(ionic_steps) > 1:
            energies = [step.get("E0", step.get("F")) for step in ionic_steps]
            steps = list(range(1, len(energies) + 1))
            fig, ax = plt.subplots()
            ax.plot(steps, energies, marker="o", markersize=3)
            ax.set_xlabel("Ionic step")
            ax.set_ylabel("Energy E0 (eV)")
            ax.set_title("Ionic (geometry) convergence")
            fig.tight_layout()
            return fig, f"Energy across {len(energies)} ionic steps."

        # Single ionic step -> show SCF convergence of that step.
        if oszicar.electronic_steps:
            scf = oszicar.electronic_steps[0]
            data = [(i, abs(s["dE"])) for i, s in enumerate(scf, 1)
                    if isinstance(s.get("dE"), (int, float)) and s["dE"] != 0]
            if not data:
                return None, "No electronic-step data to plot."
            xs, ys = zip(*data)
            fig, ax = plt.subplots()
            ax.semilogy(xs, ys, marker="o", markersize=3)
            ax.set_xlabel("SCF iteration")
            ax.set_ylabel("|dE| (eV)")
            ax.set_title("Electronic (SCF) convergence")
            fig.tight_layout()
            return fig, f"SCF convergence over {len(xs)} iterations."

        return None, "No convergence data in OSZICAR."
    except Exception as exc:
        return None, f"Could not plot convergence: {exc}"


def build_trajectory_html(bundle):
    """Interactive 3D trajectory viewer for multi-step runs. Returns (html|None, message)."""
    vasprun = bundle["vasprun"]
    structures = None
    if vasprun is not None:
        try:
            structures = vasprun.structures
        except Exception:
            structures = None

    # Fall back to XDATCAR when vasprun.xml is missing/truncated (e.g. an MD run).
    if not structures or len(structures) <= 1:
        try:
            xdatcar_path = os.path.join(bundle["path"], "XDATCAR")
            if os.path.exists(xdatcar_path):
                structures = Xdatcar(xdatcar_path).structures
        except Exception:
            pass

    if not structures or len(structures) <= 1:
        return None, "Single-point run — no trajectory to animate."

    try:
        # Subsample long trajectories so the generated HTML stays small.
        note = ""
        frames = structures
        if len(frames) > MAX_TRAJECTORY_FRAMES:
            stride = len(frames) // MAX_TRAJECTORY_FRAMES + 1
            frames = frames[::stride]
            note = (f" Showing {len(frames)} of {len(structures)} frames "
                    f"(every {stride}th).")

        try:
            atoms_list = [AseAtomsAdaptor.get_atoms(s) for s in frames]
            view = nglview.show_asetraj(atoms_list)
        except Exception:
            # Fall back to the final frame if the ASE animation path fails.
            view = nglview.show_pymatgen(structures[-1])
            note = " Trajectory animation unavailable; showing the final frame."
        view.add_unitcell()

        html_path = "./static/result_trajectory.html"
        if os.path.exists(html_path):
            os.remove(html_path)
        nglview.write_html(html_path, [view])

        timestamp = int(time.time())
        html = (f'<iframe src="/static/result_trajectory.html?ts={timestamp}" '
                f'height="500" width="500" title="Trajectory"></iframe>')
        return html, f"Trajectory over {len(structures)} ionic steps.{note}"
    except Exception as exc:
        return None, f"Could not render trajectory: {exc}"


def on_render_chgcar(working_directory_path, isolevel):
    """Render a CHGCAR charge-density isosurface. Returns (iframe html|None, message).

    Independent of the main Load button because parsing a CHGCAR grid is heavy.
    """
    if not working_directory_path:
        return None, "No working directory is open."
    chgcar_path = os.path.join(working_directory_path, "CHGCAR")
    if not os.path.exists(chgcar_path):
        return None, "CHGCAR not found in this directory."

    try:
        chg = Chgcar.from_file(chgcar_path)

        # Export the total density to a cube file for nglview to load.
        cube_path = "./static/density.cube"
        if os.path.exists(cube_path):
            os.remove(cube_path)
        chg.to_cube(cube_path)

        # Structure + isosurface overlay. add_component/add_surface must happen
        # before write_html so the cube blob is embedded in the standalone HTML.
        view = nglview.show_pymatgen(chg.structure)
        view.add_unitcell()
        component = view.add_component(cube_path)
        component.add_representation("surface", isolevel=float(isolevel),
                                     isolevel_type="sigma", color="blue",
                                     opacity=0.6, wireframe=False)

        html_path = "./static/result_chgcar.html"
        if os.path.exists(html_path):
            os.remove(html_path)
        nglview.write_html(html_path, [view])

        timestamp = int(time.time())
        html = (f'<iframe src="/static/result_chgcar.html?ts={timestamp}" '
                f'height="500" width="500" title="Charge Density"></iframe>')
        grid = "×".join(str(n) for n in chg.dim)
        return html, f"Total charge-density isosurface at {isolevel}σ (grid {grid})."
    except Exception as exc:
        return None, f"Could not render charge density: {exc}"


def on_load_results(working_directory_path):
    """Parse once and fan the results out to every section. Never raises into Gradio."""
    try:
        bundle = parse_vasp_outputs(working_directory_path)

        summary_df = build_summary_dataframe(bundle)
        dos_fig, dos_msg = build_dos_plot(bundle)
        bs_fig, bs_msg = build_band_structure_plot(bundle)
        conv_fig, conv_msg = build_convergence_plot(bundle)
        traj_html, traj_msg = build_trajectory_html(bundle)

        if bundle["vasprun"] is not None:
            status = f"<p style='color:green'>Loaded results from {working_directory_path}</p>"
        elif bundle["outcar"] is not None or bundle["oszicar"] is not None:
            status = ("<p style='color:orange'>Loaded partial results "
                      "(vasprun.xml unavailable — used OUTCAR/OSZICAR).</p>")
        else:
            status = "<p style='color:red'>No readable VASP output files in this directory.</p>"
        if bundle["errors"]:
            joined = "; ".join(bundle["errors"])
            status += f"<p style='color:orange'>Notes: {joined}</p>"

        # Trailing "" clears the "files changed" hint; last value is the shared status.
        return (summary_df, dos_fig, dos_msg, bs_fig, bs_msg,
                conv_fig, conv_msg, traj_html, traj_msg, "", status)
    except Exception as exc:
        empty = pd.DataFrame(columns=["Property", "Value"])
        return (empty, None, "", None, "", None, "", None, "", "",
                f"<p style='color:red'>Error loading results: {exc}</p>")


def on_result_file_list_change(working_directory_path):
    """Clear stale results when files change; prompt the user to reload.

    Kept lightweight on purpose: it does NOT parse (that is the button's job),
    so file operations elsewhere in the app stay responsive.
    """
    empty = pd.DataFrame(columns=["Property", "Value"])
    hint = "<p><em>Files changed — click <b>Load / Refresh Results</b> to update.</em></p>"
    return (empty, None, "", None, "", None, "", None, "", hint)


def result_tab_content(working_directory_path_state, working_directory_file_list_state, status_markdown):
    with gr.Tab("Result") as result_tab:
        with gr.Row():
            load_results_button = gr.Button("Load / Refresh Results", variant="primary")
        result_hint_markdown = gr.Markdown()
        with gr.Accordion("Summary", open=True):
            summary_dataframe = gr.Dataframe(headers=["Property", "Value"], wrap=True, interactive=False)
        with gr.Accordion("Density of States", open=False):
            dos_status_markdown = gr.Markdown()
            dos_plot = gr.Plot(label="Density of States")
        with gr.Accordion("Band Structure", open=False):
            bs_status_markdown = gr.Markdown()
            band_structure_plot = gr.Plot(label="Band Structure")
        with gr.Accordion("Convergence", open=False):
            convergence_status_markdown = gr.Markdown()
            convergence_plot = gr.Plot(label="Convergence")
        with gr.Accordion("Trajectory", open=False):
            trajectory_status_markdown = gr.Markdown()
            trajectory_html = gr.HTML()
        with gr.Accordion("Charge Density (CHGCAR)", open=False):
            chgcar_status_markdown = gr.Markdown()
            chgcar_isolevel_slider = gr.Slider(minimum=0.5, maximum=8.0, step=0.5, value=2.0, label="Isosurface level (σ)")
            render_chgcar_button = gr.Button("Render Charge Density")
            chgcar_html = gr.HTML()

    # Common section outputs shared by both handlers.
    section_outputs = [summary_dataframe, dos_plot, dos_status_markdown,
                       band_structure_plot, bs_status_markdown,
                       convergence_plot, convergence_status_markdown,
                       trajectory_html, trajectory_status_markdown]

    # The button reports to the shared status bar; the passive change-handler
    # only writes the local hint so it never clobbers the Calculation tab's status.
    load_results_button.click(
        on_load_results, [working_directory_path_state],
        section_outputs + [result_hint_markdown, status_markdown])
    working_directory_file_list_state.change(
        on_result_file_list_change, [working_directory_path_state],
        section_outputs + [result_hint_markdown])

    # CHGCAR viewer has its own button (heavy parse, kept off the main Load path).
    render_chgcar_button.click(
        on_render_chgcar, [working_directory_path_state, chgcar_isolevel_slider],
        [chgcar_html, chgcar_status_markdown])

    return result_tab
