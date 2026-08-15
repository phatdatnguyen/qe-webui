import os
import glob
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend for the web server
import matplotlib.pyplot as plt
import gradio as gr
import nglview
from pymatgen.io.espresso.outputs import PWxml
from pymatgen.io.espresso.inputs.pwin import PWin
from pymatgen.electronic_structure.dos import Dos
from pymatgen.electronic_structure.plotter import DosPlotter, BSPlotter
from pymatgen.io.ase import AseAtomsAdaptor

# Trajectory frames are subsampled to at most this many to keep the
# generated HTML small and the viewer responsive for long relaxations.
MAX_TRAJECTORY_FRAMES = 100

# Shared prefix/outdir used by calculation.py when generating inputs.
PREFIX = "pwscf"
OUTDIR = "out"


def find_qe_xml(working_directory_path):
    """Locate the pw.x XML output, trying the standard locations in order."""
    candidates = [
        os.path.join(working_directory_path, OUTDIR, f"{PREFIX}.xml"),
        os.path.join(working_directory_path, f"{PREFIX}.xml"),
        os.path.join(working_directory_path, OUTDIR, f"{PREFIX}.save", "data-file-schema.xml"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Fall back to any .xml under the working dir / outdir (sorted for determinism).
    for pattern in (os.path.join(working_directory_path, OUTDIR, "*.xml"),
                    os.path.join(working_directory_path, "*.xml")):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    return None


def find_xml_choices(working_directory_path):
    """List every .xml under the working dir (relative paths) for the picker,
    with the conventional out/pwscf.xml first."""
    if not working_directory_path:
        return []
    hits = glob.glob(os.path.join(working_directory_path, "**", "*.xml"), recursive=True)
    rels = sorted(os.path.relpath(h, working_directory_path) for h in hits)
    preferred = os.path.join(OUTDIR, f"{PREFIX}.xml")
    if preferred in rels:
        rels.remove(preferred)
        rels.insert(0, preferred)
    return rels


def _prefix_from_xml(xml_path):
    """Infer the QE prefix from an XML path so DOS/cube lookups stay consistent."""
    base = os.path.basename(xml_path)
    if base == "data-file-schema.xml":
        save = os.path.basename(os.path.dirname(xml_path))
        return save[:-5] if save.endswith(".save") else PREFIX
    return base[:-4] if base.endswith(".xml") else PREFIX


def parse_qe_outputs(working_directory_path, xml_rel=None):
    """Parse a single pw.x XML (the one picked in the UI, else the default), guarded."""
    bundle = {"pwxml": None, "xml_path": None, "prefix": PREFIX,
              "path": working_directory_path, "errors": []}

    if not working_directory_path:
        bundle["errors"].append("No working directory is open.")
        return bundle

    xml_path = None
    if xml_rel:
        candidate = os.path.join(working_directory_path, xml_rel)
        if os.path.exists(candidate):
            xml_path = candidate
    if xml_path is None:
        xml_path = find_qe_xml(working_directory_path)
    if xml_path is None:
        bundle["errors"].append("No Quantum Espresso XML output found "
                                f"(looked for {OUTDIR}/{PREFIX}.xml).")
        return bundle

    bundle["xml_path"] = xml_path
    bundle["prefix"] = _prefix_from_xml(xml_path)
    try:
        bundle["pwxml"] = PWxml(xml_path)
    except Exception as exc:
        bundle["errors"].append(f"{os.path.basename(xml_path)}: {exc}")

    return bundle


def build_summary_dataframe(bundle):
    """Build a Property/Value table of key scalars, degrading per-row."""
    pwxml = bundle["pwxml"]
    if pwxml is None:
        return pd.DataFrame([["Status", "No readable QE XML output"]],
                            columns=["Property", "Value"])

    rows = []

    def add(label, fn):
        try:
            rows.append([label, fn()])
        except Exception:
            rows.append([label, "n/a"])

    add("Final total energy (eV)", lambda: f"{pwxml.final_energy:.6f}")

    def band_gap():
        gap, cbm, vbm, is_direct = pwxml.eigenvalue_band_properties
        if gap is None or gap <= 0.01:
            return "Metallic (no gap)"
        return f"{gap:.3f} eV ({'direct' if is_direct else 'indirect'})"
    add("Band gap", band_gap)

    add("Fermi level", lambda: f"{pwxml.efermi:.4f} eV" if pwxml.efermi is not None else "n/a")
    add("Converged (electronic)", lambda: str(pwxml.converged_electronic))
    add("Converged (ionic)", lambda: str(pwxml.converged_ionic))
    add("Run type", lambda: str(pwxml.run_type))
    add("Formula", lambda: pwxml.final_structure.composition.reduced_formula)
    add("Number of sites", lambda: str(len(pwxml.final_structure)))
    add("Ionic steps", lambda: str(pwxml.nionic_steps))

    return pd.DataFrame(rows, columns=["Property", "Value"])


def _find_dos_files(working_directory_path, prefix):
    """Return (fildos_path_or_None, filpdos_prefix_or_None) for the DOS section."""
    fildos = os.path.join(working_directory_path, f"{prefix}.dos")
    fildos = fildos if os.path.exists(fildos) else None

    pdos_hits = glob.glob(os.path.join(working_directory_path, f"{prefix}.pdos_*"))
    filpdos = os.path.join(working_directory_path, prefix) if pdos_hits else None
    return fildos, filpdos


def _projected_dos_series(dos_run, mode):
    """Sum projwfc pdos into labelled {Spin: densities} groups.

    mode="element" groups by element symbol; mode="orbital" groups by orbital
    angular momentum (s/p/d/f). Returns {} if no pdos is available.
    """
    pdos = getattr(dos_run, "pdos", None)
    if not pdos:
        return {}
    symbols = dos_run.atomic_symbols
    groups = {}
    for site_idx, orbital_map in enumerate(pdos):
        element = symbols[site_idx]
        for orbital, spin_map in orbital_map.items():
            label = element if mode == "element" else getattr(orbital, "name", str(orbital))[0]
            for spin, dens in spin_map.items():
                bucket = groups.setdefault(label, {})
                bucket[spin] = bucket.get(spin, np.zeros_like(dens)) + dens
    return groups


def build_dos_plot(bundle, mode="element"):
    """Total DOS plus per-element or per-orbital projected DOS. Returns (figure|None, message).

    Requires a dos.x (fildos) and/or projwfc.x (filpdos) run in the working dir.
    Spin-polarized runs are rendered as up/down automatically by DosPlotter.
    """
    xml_path = bundle["xml_path"]
    if xml_path is None:
        return None, "QE XML not available."

    fildos, filpdos = _find_dos_files(bundle["path"], bundle["prefix"])
    if fildos is None and filpdos is None:
        return None, ("No DOS data found. Run a dos.x (fildos) and/or projwfc.x "
                      "(filpdos) calculation first — see the Calculation tab.")

    try:
        dos_run = PWxml(xml_path,
                        parse_dos=bool(fildos), fildos=fildos,
                        parse_pdos=bool(filpdos), filpdos=filpdos)
        tdos = dos_run.tdos
        if tdos is None:
            return None, "Could not read a total DOS from the DOS files."

        plotter = DosPlotter(zero_at_efermi=True, stack=False, sigma=0.05)
        plotter.add_dos("Total", tdos)

        proj_note = ""
        try:
            groups = _projected_dos_series(dos_run, mode)
            for label, dens in groups.items():
                plotter.add_dos(label, Dos(dos_run.efermi, tdos.energies, dens))
            if groups:
                proj_note = f" ({mode}-projected — needs projwfc.x)"
            elif mode == "orbital":
                proj_note = " (orbital projection needs a projwfc.x run)"
        except Exception:
            pass  # total DOS still useful if projection fails

        ax = plotter.get_plot()
        return ax.figure, f"Total DOS{proj_note} (energy relative to E_F)."
    except Exception as exc:
        return None, f"DOS not available for this run: {exc}"


def _find_bands_input(working_directory_path):
    """Locate the pw.x 'bands' input (K_POINTS crystal_b) needed for k-labels."""
    for path in glob.glob(os.path.join(working_directory_path, "*.in")):
        try:
            with open(path) as fh:
                if "crystal_b" in fh.read().lower():
                    return path
        except Exception:
            continue
    return None


def build_band_structure_plot(bundle):
    """Band structure for line-mode (bands) runs. Returns (figure|None, message)."""
    pwxml = bundle["pwxml"]
    if pwxml is None:
        return None, "QE XML not available."

    kpoints_filename = _find_bands_input(bundle["path"])
    if not kpoints_filename:
        return None, ("Band-structure plot needs the line-mode 'bands' input file "
                      "(K_POINTS crystal_b) in this directory. Generate and run a "
                      "'bands' calculation, then reload.")
    try:
        # The band structure can only be built along symmetry lines if the bands
        # input carries k-point labels. Guide the user if they are missing (older
        # inputs generated before labels were added).
        k_card = PWin.from_file(kpoints_filename).k_points
        if not any((lbl or "").strip() for lbl in (k_card.labels or [])):
            return None, ("The bands input has no k-point labels, so the k-path can't "
                          "be labelled. Re-generate the 'bands' input (this version "
                          "writes labels) and re-run it, then reload.")
        # PWxml.projected_eigenvalues (read inside get_band_structure) references
        # self.atomic_states, which only exists after a projwfc projection parse.
        # Set it to None so a plain (non-projected) band structure builds.
        if not hasattr(pwxml, "atomic_states"):
            pwxml.atomic_states = None
        bs = pwxml.get_band_structure(kpoints_filename=kpoints_filename, line_mode=True)
        ax = BSPlotter(bs).get_plot()
        fig = ax.figure if hasattr(ax, "figure") else ax
        return fig, "Band structure along the high-symmetry k-path."
    except Exception as exc:
        return None, f"Could not plot band structure: {exc}"


def build_convergence_plot(bundle):
    """Energy vs ionic step for relax/vc-relax runs. Returns (figure|None, message)."""
    pwxml = bundle["pwxml"]
    if pwxml is None:
        return None, "QE XML not available."

    try:
        ionic_steps = pwxml.ionic_steps or []
        energies = []
        for step in ionic_steps:
            te = step.get("total_energy") or {}
            energies.append(te.get("etot"))
        energies = [e for e in energies if e is not None]

        if len(energies) > 1:
            steps = list(range(1, len(energies) + 1))
            fig, ax = plt.subplots()
            ax.plot(steps, energies, marker="o", markersize=3)
            ax.set_xlabel("Ionic step")
            ax.set_ylabel("Total energy (eV)")
            ax.set_title("Ionic (geometry) convergence")
            fig.tight_layout()
            return fig, f"Energy across {len(energies)} ionic steps."

        return None, ("Single-point run — no ionic convergence to plot. "
                      "See the Summary for the final energy.")
    except Exception as exc:
        return None, f"Could not plot convergence: {exc}"


def build_trajectory_html(bundle):
    """Interactive 3D trajectory viewer for multi-step runs. Returns (html|None, message)."""
    pwxml = bundle["pwxml"]
    structures = None
    if pwxml is not None:
        try:
            structures = pwxml.structures
        except Exception:
            structures = None

    if not structures or len(structures) <= 1:
        return None, "Single-point run — no trajectory to animate."

    try:
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


def on_render_cube(working_directory_path, isolevel):
    """Render a Gaussian cube isosurface (produced by pp.x). Returns (html|None, message).

    Independent of the main Load button because parsing a volumetric grid is heavy.
    """
    if not working_directory_path:
        return None, "No working directory is open."

    cube_files = glob.glob(os.path.join(working_directory_path, "*.cube"))
    if not cube_files:
        return None, ("No .cube file found. Run pp.x (with output_format=6) to export "
                      "a volumetric cube first — see the Calculation tab.")
    cube_path = cube_files[0]

    try:
        view = nglview.NGLWidget()
        component = view.add_component(cube_path)
        component.add_representation("ball+stick")
        component.add_representation("surface", isolevel=float(isolevel),
                                     isolevel_type="value", color="blue",
                                     opacity=0.6, wireframe=False)

        html_path = "./static/result_cube.html"
        if os.path.exists(html_path):
            os.remove(html_path)
        nglview.write_html(html_path, [view])

        timestamp = int(time.time())
        html = (f'<iframe src="/static/result_cube.html?ts={timestamp}" '
                f'height="500" width="500" title="Volumetric Data"></iframe>')
        return html, f"Isosurface of {os.path.basename(cube_path)} at level {isolevel}."
    except Exception as exc:
        return None, f"Could not render cube: {exc}"


def on_load_results(working_directory_path, selected_xml, dos_mode="element"):
    """Parse the selected XML and fan the results out. Never raises into Gradio."""
    try:
        bundle = parse_qe_outputs(working_directory_path, selected_xml)

        summary_df = build_summary_dataframe(bundle)
        dos_fig, dos_msg = build_dos_plot(bundle, _dos_mode(dos_mode))
        bs_fig, bs_msg = build_band_structure_plot(bundle)
        conv_fig, conv_msg = build_convergence_plot(bundle)
        traj_html, traj_msg = build_trajectory_html(bundle)

        if bundle["pwxml"] is not None:
            status = f"<p style='color:green'>Loaded results from {working_directory_path}</p>"
        else:
            status = "<p style='color:red'>No readable QE output files in this directory.</p>"
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


def _dos_mode(radio_value):
    """Map the DOS-mode radio label to the internal mode string."""
    return "orbital" if radio_value and "orbital" in radio_value.lower() else "element"


def on_render_dos(working_directory_path, selected_xml, dos_mode):
    """Re-render just the DOS when the projection mode changes."""
    try:
        bundle = parse_qe_outputs(working_directory_path, selected_xml)
        return build_dos_plot(bundle, _dos_mode(dos_mode))
    except Exception as exc:
        return None, f"Could not render DOS: {exc}"


def on_compare_runs(working_directory_path, selected_xmls):
    """Compare several runs: a scalar table + overlaid total DOS (when available).

    Returns (dataframe, figure|None, message). Great for e.g. PBE vs SCAN.
    """
    cols = ["Run", "Formula", "Energy (eV)", "Band gap", "E_F (eV)", "Run type"]
    if not working_directory_path:
        return pd.DataFrame(columns=cols), None, "No working directory is open."
    xmls = selected_xmls or []
    if len(xmls) < 1:
        return pd.DataFrame(columns=cols), None, "Select one or more XML files to compare."

    rows = []
    plotter = DosPlotter(zero_at_efermi=True, stack=False, sigma=0.05)
    n_dos = 0
    for rel in xmls:
        label = os.path.splitext(os.path.basename(rel))[0]
        try:
            bundle = parse_qe_outputs(working_directory_path, rel)
            pw = bundle["pwxml"]
            if pw is None:
                rows.append([label, "n/a", "unreadable", "n/a", "n/a", "n/a"])
                continue

            def _gap():
                gap, _c, _v, direct = pw.eigenvalue_band_properties
                return "Metallic" if (gap is None or gap <= 0.01) else \
                    f"{gap:.3f} eV ({'d' if direct else 'i'})"
            rows.append([
                label,
                pw.final_structure.composition.reduced_formula,
                f"{pw.final_energy:.6f}",
                _gap(),
                f"{pw.efermi:.3f}" if pw.efermi is not None else "n/a",
                str(pw.run_type),
            ])

            fildos, filpdos = _find_dos_files(working_directory_path, bundle["prefix"])
            if fildos or filpdos:
                dr = PWxml(bundle["xml_path"], parse_dos=bool(fildos), fildos=fildos,
                           parse_pdos=bool(filpdos), filpdos=filpdos)
                if dr.tdos is not None:
                    plotter.add_dos(label, dr.tdos)
                    n_dos += 1
        except Exception as exc:
            rows.append([label, "n/a", f"error: {exc}", "n/a", "n/a", "n/a"])

    df = pd.DataFrame(rows, columns=cols)
    fig = plotter.get_plot().figure if n_dos else None
    msg = (f"Compared {len(xmls)} run(s); overlaid total DOS for {n_dos}."
           if n_dos else f"Compared {len(xmls)} run(s). (No DOS files found to overlay.)")
    return df, fig, msg


def on_export_dos_csv(working_directory_path, selected_xml, dos_mode):
    """Write the DOS (energy + total + projected series) to a CSV for download."""
    try:
        bundle = parse_qe_outputs(working_directory_path, selected_xml)
        fildos, filpdos = _find_dos_files(working_directory_path, bundle["prefix"])
        if not (fildos or filpdos):
            gr.Warning("No DOS files found to export.")
            return None
        dr = PWxml(bundle["xml_path"], parse_dos=bool(fildos), fildos=fildos,
                   parse_pdos=bool(filpdos), filpdos=filpdos)
        tdos = dr.tdos
        data = {"energy_eV": tdos.energies}
        for spin, dens in tdos.densities.items():
            data[f"total_{'up' if int(spin) > 0 else 'down'}"] = dens
        for label, dens in _projected_dos_series(dr, _dos_mode(dos_mode)).items():
            for spin, arr in dens.items():
                data[f"{label}_{'up' if int(spin) > 0 else 'down'}"] = arr
        out = os.path.join("./static", f"{bundle['prefix']}_dos.csv")
        pd.DataFrame(data).to_csv(out, index=False)
        return out
    except Exception as exc:
        gr.Warning(f"Could not export DOS: {exc}")
        return None


def on_export_bands_csv(working_directory_path, selected_xml):
    """Write band eigenvalues (spin, kpoint, band, energy) to a CSV for download."""
    try:
        bundle = parse_qe_outputs(working_directory_path, selected_xml)
        pw = bundle["pwxml"]
        if pw is None or not getattr(pw, "eigenvalues", None):
            gr.Warning("No eigenvalues found to export.")
            return None
        rows = []
        for spin, arr in pw.eigenvalues.items():
            tag = "up" if int(spin) > 0 else "down"
            nk, nb = arr.shape[0], arr.shape[1]
            for ik in range(nk):
                for ib in range(nb):
                    rows.append([tag, ik, ib, float(arr[ik, ib, 0])])
        out = os.path.join("./static", f"{bundle['prefix']}_bands.csv")
        pd.DataFrame(rows, columns=["spin", "kpoint", "band", "energy_eV"]).to_csv(out, index=False)
        return out
    except Exception as exc:
        gr.Warning(f"Could not export band eigenvalues: {exc}")
        return None


def on_render_projected_bands(working_directory_path, selected_xml):
    """Projected ('fat') band structure. Returns (figure|None, message).

    Needs a projwfc.x run (projected eigenvalues) on the line-mode bands calc.
    Best-effort: degrades with a clear message when the projection files are absent.
    """
    if not working_directory_path:
        return None, "No working directory is open."
    bundle = parse_qe_outputs(working_directory_path, selected_xml)
    if bundle["xml_path"] is None:
        return None, "QE XML not available."
    kpoints_filename = _find_bands_input(working_directory_path)
    if not kpoints_filename:
        return None, ("Projected bands need the line-mode 'bands' input (K_POINTS "
                      "crystal_b) in this directory.")

    # projwfc.x writes the projections to <prefix>.projwfc_up only when its input
    # sets 'filproj'. Pass the path explicitly (FileGuesser doesn't check the
    # working-dir root where projwfc.x writes it).
    filproj = os.path.join(working_directory_path, bundle["prefix"])
    if not os.path.exists(filproj + ".projwfc_up"):
        return None, (f"No projection file '{bundle['prefix']}.projwfc_up' found. "
                      "Re-generate the projwfc input (this version sets 'filproj') and "
                      "re-run projwfc.x on the bands run, then reload.")
    try:
        from pymatgen.electronic_structure.plotter import BSPlotterProjected
        pw = PWxml(bundle["xml_path"], parse_projected_eigen=True, filproj=filproj)
        bs = pw.get_band_structure(kpoints_filename=kpoints_filename, line_mode=True)
        # Projections are keyed by plain element (Na); a structure carrying
        # oxidation states (Na+, from the CIF) would KeyError in the plotter.
        try:
            bs.structure.remove_oxidation_states()
        except Exception:
            pass
        BSPlotterProjected(bs).get_elt_projected_plots()
        fig = plt.gcf()
        # get_elt_projected_plots always lays out a fixed 2x2 grid; drop the empty
        # panels (e.g. only 2 elements) and reflow the rest into a single row.
        try:
            import matplotlib.gridspec as gridspec
            used = [ax for ax in fig.axes if (ax.lines or ax.collections)]
            if used and len(used) != len(fig.axes):
                for ax in list(fig.axes):
                    if ax not in used:
                        fig.delaxes(ax)
                gs = gridspec.GridSpec(1, len(used), figure=fig)
                for i, ax in enumerate(used):
                    ax.set_subplotspec(gs[0, i])
                fig.tight_layout()
        except Exception:
            pass
        return fig, "Element-projected ('fat') band structure."
    except Exception as exc:
        return None, f"Could not plot projected bands: {exc}"


def on_result_file_list_change(working_directory_path):
    """Clear stale results when files change; prompt the user to reload."""
    empty = pd.DataFrame(columns=["Property", "Value"])
    hint = "<p><em>Files changed — click <b>Load / Refresh Results</b> to update.</em></p>"
    return (empty, None, "", None, "", None, "", None, "", hint)


def on_refresh_result_files(working_directory_path):
    """Repopulate the output-file picker with the .xml files in the directory."""
    choices = find_xml_choices(working_directory_path)
    return gr.update(choices=choices, value=choices[0] if choices else None)


def on_refresh_compare_files(working_directory_path):
    """Repopulate the compare multiselect with the .xml files in the directory."""
    return gr.update(choices=find_xml_choices(working_directory_path))


def result_tab_content(working_directory_path_state, working_directory_file_list_state, status_markdown):
    with gr.Tab("Result") as result_tab:
        with gr.Row():
            result_file_dropdown = gr.Dropdown(choices=[], value=None, label="Output File to Visualize (QE XML)")
            load_results_button = gr.Button("Load / Refresh Results", variant="primary")
        result_hint_markdown = gr.Markdown()
        with gr.Accordion("Summary", open=True):
            summary_dataframe = gr.Dataframe(headers=["Property", "Value"], wrap=True, interactive=False)
        with gr.Accordion("Density of States", open=False):
            with gr.Row():
                dos_mode_radio = gr.Radio(
                    choices=["Total + element", "Total + orbital (s/p/d)"],
                    value="Total + element", label="Projection")
                dos_export_button = gr.DownloadButton("Export DOS → CSV")
            dos_status_markdown = gr.Markdown()
            dos_plot = gr.Plot(label="Density of States")
        with gr.Accordion("Band Structure", open=False):
            bs_status_markdown = gr.Markdown()
            band_structure_plot = gr.Plot(label="Band Structure")
            bands_export_button = gr.DownloadButton("Export band eigenvalues → CSV")
        with gr.Accordion("Projected Band Structure (fat bands)", open=False):
            proj_bands_status_markdown = gr.Markdown()
            render_proj_bands_button = gr.Button("Render Projected Bands")
            proj_bands_plot = gr.Plot(label="Projected Band Structure")
        with gr.Accordion("Convergence", open=False):
            convergence_status_markdown = gr.Markdown()
            convergence_plot = gr.Plot(label="Convergence")
        with gr.Accordion("Trajectory", open=False):
            trajectory_status_markdown = gr.Markdown()
            trajectory_html = gr.HTML()
        with gr.Accordion("Volumetric Data (cube)", open=False):
            cube_status_markdown = gr.Markdown()
            cube_isolevel_slider = gr.Slider(minimum=0.0001, maximum=1.0, step=0.0001, value=0.01, label="Isosurface level")
            render_cube_button = gr.Button("Render Cube")
            cube_html = gr.HTML()
        with gr.Accordion("Compare Runs", open=False):
            compare_files_dropdown = gr.Dropdown(choices=[], value=[], multiselect=True,
                                                 label="XML files to compare")
            compare_button = gr.Button("Compare")
            compare_status_markdown = gr.Markdown()
            compare_dataframe = gr.Dataframe(wrap=True, interactive=False)
            compare_plot = gr.Plot(label="Overlaid total DOS")

    # Common section outputs shared by both handlers.
    section_outputs = [summary_dataframe, dos_plot, dos_status_markdown,
                       band_structure_plot, bs_status_markdown,
                       convergence_plot, convergence_status_markdown,
                       trajectory_html, trajectory_status_markdown]

    load_results_button.click(
        on_load_results, [working_directory_path_state, result_file_dropdown, dos_mode_radio],
        section_outputs + [result_hint_markdown, status_markdown])
    working_directory_file_list_state.change(
        on_result_file_list_change, [working_directory_path_state],
        section_outputs + [result_hint_markdown])
    # Keep the output-file pickers in sync as files are generated/produced.
    working_directory_file_list_state.change(
        on_refresh_result_files, [working_directory_path_state], result_file_dropdown)
    working_directory_file_list_state.change(
        on_refresh_compare_files, [working_directory_path_state], compare_files_dropdown)

    # Re-render just the DOS when the projection mode changes.
    dos_mode_radio.change(
        on_render_dos, [working_directory_path_state, result_file_dropdown, dos_mode_radio],
        [dos_plot, dos_status_markdown])

    render_proj_bands_button.click(
        on_render_projected_bands, [working_directory_path_state, result_file_dropdown],
        [proj_bands_plot, proj_bands_status_markdown])

    render_cube_button.click(
        on_render_cube, [working_directory_path_state, cube_isolevel_slider],
        [cube_html, cube_status_markdown])

    compare_button.click(
        on_compare_runs, [working_directory_path_state, compare_files_dropdown],
        [compare_dataframe, compare_plot, compare_status_markdown])

    # Export buttons: the handler returns a file path that the DownloadButton serves.
    dos_export_button.click(
        on_export_dos_csv, [working_directory_path_state, result_file_dropdown, dos_mode_radio],
        dos_export_button)
    bands_export_button.click(
        on_export_bands_csv, [working_directory_path_state, result_file_dropdown],
        bands_export_button)

    return result_tab
