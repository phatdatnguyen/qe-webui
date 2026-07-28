import os
import shutil
import signal
import subprocess
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Incar
from pymatgen.io.vasp.sets import LobsterSet, MITMDSet, MITNEBSet, MITRelaxSet, MP24RelaxSet, MP24StaticSet, MPAbsorptionSet, MPHSEBSSet, MPHSERelaxSet, MPMDSet, MPMetalRelaxSet, MPNMRSet, MPNonSCFSet, MPRelaxSet, MPSOCSet, MPScanRelaxSet, MPScanStaticSet, MPStaticSet, MVLElasticSet, MVLGBSet, MVLGWSet, MVLNPTMDSet, MVLRelax52Set, MVLScanRelaxSet, MVLSlabSet, MatPESStaticSet, NEBSet
import gradio as gr
from utils import get_files_in_working_directory

# Map input-set name -> pymatgen class. Adding a set is now a one-line entry.
INPUT_SETS = {
    "LobsterSet": LobsterSet, "MITMDSet": MITMDSet, "MITNEBSet": MITNEBSet,
    "MITRelaxSet": MITRelaxSet, "MP24RelaxSet": MP24RelaxSet, "MP24StaticSet": MP24StaticSet,
    "MPAbsorptionSet": MPAbsorptionSet, "MPHSEBSSet": MPHSEBSSet, "MPHSERelaxSet": MPHSERelaxSet,
    "MPMDSet": MPMDSet, "MPMetalRelaxSet": MPMetalRelaxSet, "MPNMRSet": MPNMRSet,
    "MPNonSCFSet": MPNonSCFSet, "MPRelaxSet": MPRelaxSet, "MPSOCSet": MPSOCSet,
    "MPScanRelaxSet": MPScanRelaxSet, "MPScanStaticSet": MPScanStaticSet, "MPStaticSet": MPStaticSet,
    "MVLElasticSet": MVLElasticSet, "MVLGBSet": MVLGBSet, "MVLGWSet": MVLGWSet,
    "MVLNPTMDSet": MVLNPTMDSet, "MVLRelax52Set": MVLRelax52Set, "MVLScanRelaxSet": MVLScanRelaxSet,
    "MVLSlabSet": MVLSlabSet, "MatPESStaticSet": MatPESStaticSet, "NEBSet": NEBSet,
}

# These sets build on a previous run: they take their structure from vasprun.xml
# and are constructed without user_kpoints_settings.
PRIOR_RUN_SETS = {"MPHSEBSSet", "MPNonSCFSet", "MPSOCSet"}

# Available VASP binaries and the calculation kinds they are meant for.
VASP_EXECUTABLES = ["vasp_std", "vasp_gam", "vasp_ncl"]

# Handle to the currently running VASP process, so the Stop button can reach it.
_current_process = None


def on_working_directory_file_list_change(working_directory_file_list):
    # Accept structures VASP can start from, not just .cif.
    structure_suffixes = (".cif", "POSCAR", "CONTCAR", ".vasp")
    structure_file_names = [f for f in (working_directory_file_list or [])
                            if f.endswith(structure_suffixes)]
    return gr.update(choices=structure_file_names,
                     value=structure_file_names[0] if structure_file_names else None,
                     interactive=True)


def on_select_calculation_type(calculation_type):
    calculation_type_to_input_sets = {
        "Static": ["MPStaticSet", "MP24StaticSet", "MPScanStaticSet"],
        "Relaxation": ["MPRelaxSet", "MPMetalRelaxSet", "MPScanRelaxSet", "MP24RelaxSet", "MPHSERelaxSet", "MITRelaxSet", "MVLRelax52Set", "MVLScanRelaxSet"],
        "Electronic Structure": ["MPHSEBSSet", "MPNonSCFSet", "MPSOCSet"],
        "Transition States": ["NEBSet", "MITNEBSet"],
        "Molecular Dynamics": ["MPMDSet", "MITMDSet", "MVLNPTMDSet"],
        "Other": ["MPAbsorptionSet", "MVLSlabSet", "MVLGBSet", "MVLGWSet", "LobsterSet", "MPNMRSet", "MVLElasticSet", "MatPESStaticSet"]
    }

    input_sets = calculation_type_to_input_sets.get(calculation_type, [])
    if not input_sets:
        input_sets = [""]

    return gr.update(choices=input_sets, value=input_sets[0] if input_sets else "")


def suggest_executable(input_set):
    """Suggest the correct binary for the chosen set (SOC requires vasp_ncl)."""
    return gr.update(value="vasp_ncl" if input_set == "MPSOCSet" else "vasp_std")


def parse_extra_incar(text):
    """Parse a free-text 'KEY = VALUE' block into a typed INCAR settings dict."""
    if not text or not text.strip():
        return {}
    return dict(Incar.from_str(text))


def on_generate_vasp_inputs(working_directory_path, input_structure_file_name, input_set,
                            precision_dropdown, reciprocal_density, potcar_functional, extra_incar):
    try:
        set_cls = INPUT_SETS.get(input_set)
        if set_cls is None:
            raise Exception(f"Unknown input set: {input_set!r}")

        # Base PREC plus any user-supplied INCAR overrides.
        incar_settings = {"PREC": precision_dropdown}
        incar_settings.update(parse_extra_incar(extra_incar))

        if input_set in PRIOR_RUN_SETS:
            vasprun_path = os.path.join(working_directory_path, "vasprun.xml")
            if not os.path.exists(vasprun_path):
                raise Exception("Cannot find the output files of previous run (vasprun.xml).")
            input_structure = Structure.from_file(vasprun_path)
            vis = set_cls(input_structure,
                          user_incar_settings=incar_settings,
                          user_potcar_functional=potcar_functional)
        else:
            if not input_structure_file_name:
                raise Exception("Please select an input structure file.")
            input_structure = Structure.from_file(os.path.join(working_directory_path, input_structure_file_name))
            vis = set_cls(input_structure,
                          user_incar_settings=incar_settings,
                          user_kpoints_settings={"reciprocal_density": reciprocal_density},
                          user_potcar_functional=potcar_functional)

        vis.write_input(working_directory_path)

        return (f"<p style='color:green'>Input files generated at {working_directory_path}</p>",
                get_files_in_working_directory(working_directory_path))

    except Exception as e:
        return (f"<p style='color:red'>Error generating inputs: {e}</p>",
                get_files_in_working_directory(working_directory_path))


def _resolve_executable(executable_name, vasp_bin_dir):
    """Return the runnable executable path, or None if it can't be found."""
    if vasp_bin_dir and vasp_bin_dir.strip():
        candidate = os.path.join(vasp_bin_dir.strip(), executable_name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
        return None
    return shutil.which(executable_name)


def on_run_calculation(working_directory_path, num_cores_slider, executable_name, vasp_bin_dir):
    global _current_process
    try:
        if not working_directory_path:
            yield "<p style='color:red'>No working directory is open.</p>", ""
            return

        # Pre-flight: required input files must exist.
        required = ["INCAR", "POSCAR", "POTCAR"]
        missing = [f for f in required
                   if not os.path.exists(os.path.join(working_directory_path, f))]
        if missing:
            yield (f"<p style='color:red'>Missing input files: {', '.join(missing)}. "
                   f"Generate inputs first.</p>", "")
            return

        # Pre-flight: mpirun and the chosen VASP binary must be resolvable.
        if shutil.which("mpirun") is None:
            yield "<p style='color:red'>'mpirun' not found on PATH.</p>", ""
            return
        exe = _resolve_executable(executable_name, vasp_bin_dir)
        if exe is None:
            where = f"in '{vasp_bin_dir}'" if vasp_bin_dir and vasp_bin_dir.strip() else "on PATH"
            yield (f"<p style='color:red'>VASP executable '{executable_name}' not found {where}.</p>", "")
            return

        # start_new_session lets us kill the whole process group (mpirun + ranks) on Stop.
        process = subprocess.Popen(
            args=f"mpirun -np {int(num_cores_slider)} {exe}",
            cwd=working_directory_path,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            start_new_session=True,
        )
        _current_process = process

        output_log = ""
        for line in process.stdout:
            output_log += line
            yield "<p>Running VASP calculation...</p>", output_log

        process.wait()
        _current_process = None

        if process.returncode == 0:
            yield "<p style='color:green'>VASP calculation finished successfully!</p>", output_log
        elif process.returncode < 0:
            yield (f"<p style='color:orange'>VASP calculation was stopped "
                   f"(signal {-process.returncode}).</p>", output_log)
        else:
            yield (f"<p style='color:red'>VASP calculation exited with error code "
                   f"{process.returncode}</p>", output_log)

    except Exception as e:
        _current_process = None
        yield f"<p style='color:red'>Error running VASP: {e}</p>", None


def on_stop_calculation():
    global _current_process
    process = _current_process
    if process is not None and process.poll() is None:
        try:
            # Kill the whole process group started with start_new_session.
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception as e:
            return f"<p style='color:red'>Could not stop calculation: {e}</p>"
        return "<p style='color:orange'>Stopping VASP calculation...</p>"
    return "<p>No calculation is currently running.</p>"


def calculation_tab_content(working_directory_path_state, working_directory_file_list_state, status_markdown):
    with gr.Tab("Calculation") as calculation_tab:
        with gr.Accordion("Generate VASP Input Files"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_structure_file_name_dropdown = gr.Dropdown(choices=[], value=None, label="Input Structure")
                    calculation_type_dropdown = gr.Dropdown(choices=["Static", "Relaxation", "Electronic Structure", "Transition States", "Molecular Dynamics", "Other"], value="Static", label="Calculation Type")
                    input_set_dropdown = gr.Dropdown(choices=["MPStaticSet", "MP24StaticSet", "MPScanStaticSet", ""], value="MPStaticSet", label="Input Set")
                with gr.Column(scale=1):
                    precision_dropdown = gr.Dropdown(choices=["Low", "Medium", "Normal", "High", "Accurate"], value="Normal", label="Precision")
                    reciprocal_density_slider = gr.Slider(minimum=20, maximum=400, step=10, value=100, label="Reciprocal Density (Å⁻³)")
                    potcar_functional_dropdown = gr.Dropdown(choices=["PBE_64", "LDA_64"], value="PBE_64", label="POTCAR Functional")
            with gr.Row():
                extra_incar_textbox = gr.Textbox(
                    label="Extra INCAR settings (one KEY = VALUE per line)",
                    placeholder="ENCUT = 520\nISMEAR = 0\nSIGMA = 0.05",
                    lines=4)
            with gr.Row():
                generate_button = gr.Button("Generate Input Files")
        with gr.Accordion("Run Calculation"):
            with gr.Row():
                num_cores = max(1, (os.cpu_count() or 2) // 2)
                num_cores_slider = gr.Slider(minimum=1, maximum=num_cores, step=1, value=1, label="Number of Cores")
                vasp_executable_dropdown = gr.Dropdown(choices=VASP_EXECUTABLES, value="vasp_std", label="VASP Executable")
            with gr.Row():
                vasp_bin_dir_textbox = gr.Textbox(
                    label="VASP binary directory (optional)",
                    placeholder="Leave blank to use PATH, e.g. /opt/vasp/bin")
            with gr.Row():
                run_button = gr.Button("Run", variant="primary")
                stop_button = gr.Button("Stop", variant="stop")
            with gr.Row():
                output_log_textbox = gr.Textbox(label="VASP Output Log", lines=10)

        working_directory_file_list_state.change(on_working_directory_file_list_change, [working_directory_file_list_state], [input_structure_file_name_dropdown])

        calculation_type_dropdown.change(on_select_calculation_type, calculation_type_dropdown, input_set_dropdown)
        input_set_dropdown.change(suggest_executable, input_set_dropdown, vasp_executable_dropdown)
        generate_button.click(on_generate_vasp_inputs, [working_directory_path_state, input_structure_file_name_dropdown, input_set_dropdown, precision_dropdown, reciprocal_density_slider, potcar_functional_dropdown, extra_incar_textbox], [status_markdown, working_directory_file_list_state])

        run_event = run_button.click(on_run_calculation, [working_directory_path_state, num_cores_slider, vasp_executable_dropdown, vasp_bin_dir_textbox], [status_markdown, output_log_textbox])
        stop_button.click(on_stop_calculation, None, status_markdown, cancels=[run_event])

    return calculation_tab
