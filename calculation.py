import os
import re
import shlex
import shutil
import signal
import subprocess
import gradio as gr
from pymatgen.core import Structure
from pymatgen.io.pwscf import PWInput
from pymatgen.symmetry.bandstructure import HighSymmKpath
from utils import get_files_in_working_directory

# Root folder holding the pseudopotential sets, taken from QE's own
# $ESPRESSO_PSEUDO when it is set (see Readme). Each immediate subdirectory is
# one set of .UPF files (e.g. SSSP-lib-pbe-eff-v2, SG14_ONCV) and becomes a
# choice in the "Pseudopotential Set" dropdown.
PSEUDO_ROOT = os.environ.get("ESPRESSO_PSEUDO") or os.path.expanduser("~/q-e-pseudo")

# Dropdown entry meaning "PSEUDO_ROOT itself", offered when .UPF files sit
# directly in the root rather than in a subdirectory.
PSEUDO_ROOT_CHOICE = "."

# QE executables offered in the Run section.
QE_EXECUTABLES = ["pw.x", "dos.x", "projwfc.x", "bands.x", "pp.x"]

# pw.x calculation types generated from a Structure via PWInput.
PW_TYPES = ["scf", "relax", "vc-relax", "nscf", "bands"]

# Calculation types that optimize the geometry — after these we save a .cif of
# the relaxed structure so it can seed further calculations.
RELAX_TYPES = ("relax", "vc-relax")

# Post-processing input types generated from plain-text namelist templates.
POST_TYPES = ["dos", "projwfc", "bands.x", "pp.x"]

# All choices shown in the Calculation Type dropdown.
CALC_TYPES = PW_TYPES + POST_TYPES

# Default input-file names per calculation type (post types get a clean name).
POST_DEFAULT_FILES = {"dos": "dos.in", "projwfc": "projwfc.in",
                      "bands.x": "bandsx.in", "pp.x": "pp.in"}

# Executable each calculation type should run with. pw.x types use pw.x; each
# post-processing type is run by its own binary (dos -> dos.x, projwfc -> projwfc.x).
CALC_EXECUTABLE = {"scf": "pw.x", "relax": "pw.x", "vc-relax": "pw.x",
                   "nscf": "pw.x", "bands": "pw.x", "dos": "dos.x",
                   "projwfc": "projwfc.x", "bands.x": "bands.x", "pp.x": "pp.x"}

# Output directory shared by every run; the prefix (output name) is user-chosen.
OUTDIR = "./out"
DEFAULT_PREFIX = "pwscf"

# Exchange–correlation functional choices. Label -> QE `input_dft` value written
# into &SYSTEM. None keeps QE's default (the functional baked into the UPF
# pseudopotentials); CUSTOM_FUNCTIONAL is a sentinel meaning "use the free-text
# box" (for any Libxc `XC-...` code or name not listed here, e.g. r2SCAN).
CUSTOM_FUNCTIONAL = "__custom__"
DEFAULT_FUNCTIONAL = "Default (from pseudopotentials)"
FUNCTIONALS = {
    DEFAULT_FUNCTIONAL: None,
    "LDA — PZ": "pz",
    "LDA — PW": "pw",
    "GGA — PBE": "pbe",
    "GGA — PBEsol": "pbesol",
    "GGA — revPBE": "revpbe",
    "GGA — BLYP": "blyp",
    "GGA — PW91": "pw91",
    "meta-GGA — SCAN": "scan",
    "meta-GGA — TPSS": "tpss",
    "meta-GGA — M06-L": "m06l",
    "hybrid — PBE0": "pbe0",
    "hybrid — HSE06": "hse",
    "hybrid — B3LYP": "b3lyp",
    "Custom (Libxc / input_dft)": CUSTOM_FUNCTIONAL,
}

# input_dft values that are meta-GGA; used to apply an SCF-friendly mixing_beta.
META_GGA = {"scan", "tpss", "m06l"}

# Characters not allowed in a file name / prefix.
_INVALID_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# Handle to the currently running QE process, so the Stop button can reach it.
_current_process = None


def default_input_name(calc_type):
    """Default input-file name for a calculation type, e.g. scf -> scf.in."""
    return POST_DEFAULT_FILES.get(calc_type, f"{calc_type}.in")


def resolve_functional(choice, custom_text):
    """Resolve the Functional dropdown choice to (input_dft | None, is_metagga).

    ``None`` means "leave QE's default" (from the pseudopotentials). For the
    Custom entry the stripped free-text value is used (error if empty).
    """
    value = FUNCTIONALS.get(choice, None)
    if value is None:
        return None, False
    if value == CUSTOM_FUNCTIONAL:
        value = (custom_text or "").strip()
        if not value:
            raise Exception("Please enter a functional in the Custom input_dft box "
                            "(or pick a functional from the dropdown).")
    low = value.lower()
    is_metagga = low in META_GGA or "scan" in low or "mgga" in low
    return value, is_metagga


def validate_name(name, kind):
    """Return an error string if name is empty/has invalid characters, else None."""
    name = (name or "").strip()
    if not name:
        return f"Please provide a {kind}."
    if _INVALID_NAME.search(name) or name in (".", ".."):
        return (f"The {kind} {name!r} contains invalid characters "
                "(avoid / \\ : * ? \" < > | and control characters).")
    return None


# --------------------------------------------------------------------------- #
# Input-file generation
# --------------------------------------------------------------------------- #

def _base_sections(pseudo_dir, ecutwfc, ecutrho, prefix):
    """Common &CONTROL/&SYSTEM/&ELECTRONS defaults shared by all pw.x runs."""
    control = {
        "prefix": prefix,
        "outdir": OUTDIR,
        "pseudo_dir": pseudo_dir,
        "verbosity": "high",
        "tprnfor": True,
        "tstress": True,
    }
    system = {
        "ecutwfc": float(ecutwfc),
        "ecutrho": float(ecutrho),
        "occupations": "smearing",
        "smearing": "gaussian",
        "degauss": 0.01,
    }
    electrons = {
        "conv_thr": 1.0e-6,
        "mixing_beta": 0.7,
    }
    return control, system, electrons


def _pw_sections(calc_type, pseudo_dir, ecutwfc, ecutrho, prefix):
    """Return (control, system, electrons, ions, cell, kpoints_mode) for a pw.x type."""
    control, system, electrons = _base_sections(pseudo_dir, ecutwfc, ecutrho, prefix)
    ions, cell = {}, {}
    kpoints_mode = "automatic"

    control["calculation"] = calc_type

    if calc_type == "relax":
        control["forc_conv_thr"] = 1.0e-3
        ions = {"ion_dynamics": "bfgs"}
    elif calc_type == "vc-relax":
        control["forc_conv_thr"] = 1.0e-3
        ions = {"ion_dynamics": "bfgs"}
        cell = {"cell_dynamics": "bfgs", "press": 0.0, "press_conv_thr": 0.5}
    elif calc_type == "nscf":
        # Dense mesh + tetrahedra is the standard recipe for a DOS run.
        system["occupations"] = "tetrahedra"
        system.pop("smearing", None)
        system.pop("degauss", None)
        system["nosym"] = True
    elif calc_type == "bands":
        control["calculation"] = "bands"
        kpoints_mode = "crystal_b"

    return control, system, electrons, ions, cell, kpoints_mode


def build_post_input(calc_type, prefix):
    """Plain-text namelist input for a post-processing binary, referencing the
    given prefix/OUTDIR so it acts on the matching pw.x results."""
    if calc_type == "dos":
        return (f"&DOS\n  prefix = '{prefix}'\n  outdir = '{OUTDIR}'\n"
                f"  fildos = '{prefix}.dos'\n  deltae = 0.05\n/\n")
    if calc_type == "projwfc":
        # filproj makes projwfc.x write <prefix>.projwfc_up (needed for projected
        # "fat" band structures); filpdos writes the <prefix>.pdos_* files for DOS.
        return (f"&PROJWFC\n  prefix = '{prefix}'\n  outdir = '{OUTDIR}'\n"
                f"  filpdos = '{prefix}'\n  filproj = '{prefix}'\n"
                f"  degauss = 0.01\n  deltae = 0.05\n/\n")
    if calc_type == "bands.x":
        return (f"&BANDS\n  prefix = '{prefix}'\n  outdir = '{OUTDIR}'\n"
                f"  filband = '{prefix}.bands'\n/\n")
    if calc_type == "pp.x":
        return (f"&INPUTPP\n  prefix = '{prefix}'\n  outdir = '{OUTDIR}'\n"
                f"  plot_num = 0\n  filplot = '{prefix}.pp'\n/\n"
                f"&PLOT\n  iflag = 3\n  output_format = 6\n"
                f"  fileout = '{prefix}.cube'\n/\n")
    raise Exception(f"Unknown post-processing type: {calc_type!r}")


def _contains_upf(path):
    try:
        return any(f.lower().endswith(".upf") for f in os.listdir(path))
    except OSError:
        return False


def list_pseudo_sets():
    """Names of the pseudopotential sets available under PSEUDO_ROOT.

    Every immediate subdirectory counts as one set; PSEUDO_ROOT itself is listed
    first as PSEUDO_ROOT_CHOICE when it holds .UPF files directly. Returns an
    empty list when $ESPRESSO_PSEUDO is unset/missing, which leaves the dropdown
    empty (the user can still type a path into it).
    """
    if not os.path.isdir(PSEUDO_ROOT):
        return []
    sets = sorted(name for name in os.listdir(PSEUDO_ROOT)
                  if os.path.isdir(os.path.join(PSEUDO_ROOT, name)))
    if _contains_upf(PSEUDO_ROOT):
        sets.insert(0, PSEUDO_ROOT_CHOICE)
    return sets


def default_pseudo_set():
    """Initial dropdown value: the first available set, or None if there are none."""
    sets = list_pseudo_sets()
    return sets[0] if sets else None


def resolve_pseudo_dir(pseudo_set):
    """Turn a dropdown choice into the directory that holds the .UPF files.

    A set name resolves under PSEUDO_ROOT; the dropdown also accepts a typed
    absolute (or ~) path, which is used as-is so a folder outside
    $ESPRESSO_PSEUDO still works.
    """
    if pseudo_set is None or not str(pseudo_set).strip():
        raise Exception(
            "No pseudopotential set selected. Set $ESPRESSO_PSEUDO to the folder "
            "containing your pseudopotential sets (see the Readme) and restart "
            "the web UI, or type a directory into the dropdown.")
    choice = os.path.expanduser(str(pseudo_set).strip())
    if choice == PSEUDO_ROOT_CHOICE:
        return PSEUDO_ROOT
    if os.path.isabs(choice):
        return choice
    return os.path.join(PSEUDO_ROOT, choice)


def match_pseudopotentials(structure, pseudo_dir):
    """Map each species in the structure to a UPF file found in pseudo_dir.

    A UPF matches an element if its filename (before the first '.', '_' or '-')
    equals the element symbol, case-insensitively. Raises with a clear message
    listing every unmatched element.
    """
    if not pseudo_dir or not os.path.isdir(pseudo_dir):
        raise Exception(f"Pseudopotential directory not found: {pseudo_dir!r}")

    upf_files = [f for f in os.listdir(pseudo_dir) if f.lower().endswith(".upf")]
    if not upf_files:
        raise Exception(f"No .UPF files found in {pseudo_dir!r}")

    pseudo = {}
    missing = []
    for species in structure.composition:
        symbol = species.symbol
        match = None
        for f in upf_files:
            stem = f.split(".")[0].split("_")[0].split("-")[0]
            if stem.lower() == symbol.lower():
                match = f
                break
        if match is None:
            missing.append(symbol)
        else:
            pseudo[str(species)] = match

    if missing:
        raise Exception("No pseudopotential (.UPF) found for element(s): "
                        f"{', '.join(sorted(set(missing)))} in {pseudo_dir!r}")
    return pseudo


def build_kpath_crystal_b(structure, npoints=20):
    """Build a labelled K_POINTS crystal_b path from the structure's high-symmetry
    path. Returns a list of (kx, ky, kz, count, label).

    ``count`` is the number of points from a vertex to the next. The last vertex of
    each branch gets count 1: for an interior branch end this is QE's discontinuity
    marker (one point, no interpolation to the next branch — pymatgen-io-espresso
    reads it as a break), and for the final vertex QE ignores the count anyway.
    ``label`` is the high-symmetry point name — required for the band-structure plot.
    """
    kpath = HighSymmKpath(structure).kpath
    kpoints = kpath["kpoints"]

    out = []
    for branch in kpath["path"]:
        # Drop vertices whose coordinates repeat the previous one within a branch.
        verts = []
        for label in branch:
            coords = tuple(round(float(c), 8) for c in kpoints[label])
            if not verts or verts[-1][0] != coords:
                verts.append((coords, label))
        for j, (coords, label) in enumerate(verts):
            count = npoints if j < len(verts) - 1 else 1  # branch end / last -> 1
            out.append((coords[0], coords[1], coords[2], count, label))
    return out


def _kpoints_crystal_b_block(kpath):
    """Render a labelled 'K_POINTS crystal_b' card from build_kpath_crystal_b output.

    pymatgen's PWInput cannot write the per-point ``! label`` comments that
    pymatgen-io-espresso needs to build a band structure, so we render the card here.
    """
    lines = ["K_POINTS crystal_b", f"{len(kpath)}"]
    for kx, ky, kz, count, label in kpath:
        line = f"  {kx:.6f} {ky:.6f} {kz:.6f} {int(count)}"
        if label:
            line += f" ! {label}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def parse_extra_settings(text):
    """Parse a 'namelist.key = value' block into {namelist: {key: typed_value}}.

    Example line: ``system.nbnd = 24`` or ``electrons.conv_thr = 1e-8``.
    """
    settings = {}
    if not text or not text.strip():
        return settings

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "!")):
            continue
        if "=" not in line or "." not in line.split("=")[0]:
            raise Exception(f"Cannot parse extra setting {raw!r} "
                            "(expected 'namelist.key = value').")
        lhs, rhs = line.split("=", 1)
        namelist, key = lhs.strip().split(".", 1)
        settings.setdefault(namelist.strip().lower(), {})[key.strip()] = _coerce(rhs.strip())
    return settings


def _coerce(value):
    """Coerce a string into bool/int/float where possible, else strip quotes."""
    low = value.lower()
    if low in (".true.", "true"):
        return True
    if low in (".false.", "false"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("'\"")


def generate_pw_input_file(working_directory_path, calc_type, structure, pseudo_dir,
                           ecutwfc, ecutrho, kgrid, input_dft, is_metagga, prefix,
                           input_file_name, extras=None):
    """Build & write a pw.x input file from an already-loaded Structure.

    Reusable core (no Gradio). ``kgrid`` is a (kx, ky, kz) tuple (ignored for the
    crystal_b bands path); ``extras`` is a parsed {namelist: {key: value}} dict of
    overrides (or None). ``pseudo_dir`` is a set name from the dropdown (resolved
    under PSEUDO_ROOT) or an explicit path. Returns the written path; raises on error.
    """
    pseudo_dir = resolve_pseudo_dir(pseudo_dir)
    pseudo = match_pseudopotentials(structure, pseudo_dir)
    control, system, electrons, ions, cell, kpoints_mode = _pw_sections(
        calc_type, pseudo_dir, ecutwfc, ecutrho, prefix)

    # Functional override + meta-GGA mixing default, applied before the extra
    # overrides so the user can still override either.
    if input_dft is not None:
        system["input_dft"] = input_dft
    if is_metagga:
        electrons["mixing_beta"] = 0.3

    section_map = {"control": control, "system": system, "electrons": electrons,
                   "ions": ions, "cell": cell}
    for nml, kv in (extras or {}).items():
        if nml not in section_map:
            raise Exception(f"Unknown namelist {nml!r} in extra settings "
                            "(use control/system/electrons/ions/cell).")
        section_map[nml].update(kv)

    path = os.path.join(working_directory_path, input_file_name)

    if kpoints_mode == "crystal_b":
        kpath = build_kpath_crystal_b(structure)
        numeric = [[kx, ky, kz, count] for kx, ky, kz, count, _ in kpath]
        pw = PWInput(structure, pseudo=pseudo, control=control, system=system,
                     electrons=electrons, ions=ions, cell=cell,
                     kpoints_mode="crystal_b", kpoints_grid=numeric,
                     format_options={"kpoints_grid_decimals": 6})
        # Splice in a labelled K_POINTS block (PWInput can't write the labels that
        # pymatgen-io-espresso needs to build the band structure).
        text = str(pw)
        i, j = text.index("K_POINTS"), text.index("CELL_PARAMETERS")
        text = text[:i] + _kpoints_crystal_b_block(kpath) + text[j:]
        with open(path, "w") as fh:
            fh.write(text)
        return path

    kx, ky, kz = kgrid
    pw = PWInput(structure, pseudo=pseudo, control=control, system=system,
                 electrons=electrons, ions=ions, cell=cell,
                 kpoints_mode="automatic", kpoints_grid=(int(kx), int(ky), int(kz)))
    pw.write_file(path)
    return path


def write_post_input_file(working_directory_path, calc_type, prefix, input_file_name):
    """Write a plain-text post-processing input (dos/projwfc/bands.x/pp.x). Returns path."""
    path = os.path.join(working_directory_path, input_file_name)
    with open(path, "w") as fh:
        fh.write(build_post_input(calc_type, prefix))
    return path


def on_generate_qe_input(working_directory_path, calc_type, input_structure_file_name,
                         pseudo_set, ecutwfc, ecutrho, kx, ky, kz, extra_settings,
                         input_file_name, output_name, functional, custom_functional):
    """Gradio wrapper: validate, load structure, resolve functional, write the input."""
    try:
        if not working_directory_path:
            raise Exception("No working directory is open.")

        # Validate the user-chosen file name and output name (QE prefix).
        for value, kind in ((input_file_name, "input file name"),
                            (output_name, "output name")):
            err = validate_name(value, kind)
            if err:
                raise Exception(err)
        input_file_name = input_file_name.strip()
        prefix = output_name.strip()

        input_dft = None
        if calc_type in POST_TYPES:
            # Post-processing types are plain-text templates, no structure needed.
            write_post_input_file(working_directory_path, calc_type, prefix, input_file_name)
        else:
            if not input_structure_file_name:
                raise Exception("Please select an input structure file.")
            structure = Structure.from_file(
                os.path.join(working_directory_path, input_structure_file_name))
            input_dft, is_metagga = resolve_functional(functional, custom_functional)
            extras = parse_extra_settings(extra_settings)
            generate_pw_input_file(working_directory_path, calc_type, structure, pseudo_set,
                                   ecutwfc, ecutrho, (kx, ky, kz), input_dft, is_metagga,
                                   prefix, input_file_name, extras)

        functional_note = f" (input_dft = '{input_dft}')" if input_dft else ""
        files = get_files_in_working_directory(working_directory_path)
        # Select the just-generated file in the "Input File to Run" dropdown.
        input_files = [f for f in files if f.endswith((".in", ".pwi"))]
        return (f"<p style='color:green'>Generated {input_file_name} in "
                f"{working_directory_path}{functional_note}</p>",
                files,
                gr.update(choices=input_files, value=input_file_name))

    except Exception as e:
        return (f"<p style='color:red'>Error generating input: {e}</p>",
                get_files_in_working_directory(working_directory_path),
                gr.update())


def on_select_calculation_type(calc_type):
    """Toggle inputs and fill default file names when the calculation type changes.

    The input and output (log) file names track the type; the Output Name (QE
    prefix) is left untouched so the user controls it (defaults to 'pwscf').
    """
    is_pw = calc_type in PW_TYPES
    needs_grid = is_pw and calc_type != "bands"
    input_name = default_input_name(calc_type)
    return (gr.update(interactive=is_pw),                        # structure dropdown
            gr.update(interactive=needs_grid),                   # kx
            gr.update(interactive=needs_grid),                   # ky
            gr.update(interactive=needs_grid),                   # kz
            gr.update(value=CALC_EXECUTABLE.get(calc_type, "pw.x")),  # executable
            gr.update(value=input_name),                         # input file name
            gr.update(value=default_output_name(input_name)))    # output (log) file name


def on_working_directory_file_list_change(current_input_file, working_directory_file_list):
    """Refresh the structure dropdown and the run input-file dropdown.

    The run-input selection is preserved when it still exists (so a refresh after
    a run, or a generate that already selected a file, isn't clobbered); otherwise
    it defaults to the first input file.
    """
    files = working_directory_file_list or []
    structure_files = [f for f in files
                       if f.endswith((".cif", ".vasp", "POSCAR", "CONTCAR"))]
    input_files = [f for f in files if f.endswith((".in", ".pwi"))]
    input_value = (current_input_file if current_input_file in input_files
                   else (input_files[0] if input_files else None))
    return (gr.update(choices=structure_files,
                      value=structure_files[0] if structure_files else None),
            gr.update(choices=input_files, value=input_value))


def default_output_name(input_file_name):
    """Default log-file name for a given input, e.g. scf.in -> scf.out."""
    if not input_file_name:
        return ""
    return os.path.splitext(input_file_name)[0] + ".out"


def on_select_input_file(input_file_name):
    """When the input file changes, suggest a matching output file name."""
    return gr.update(value=default_output_name(input_file_name))


# --------------------------------------------------------------------------- #
# Running calculations
# --------------------------------------------------------------------------- #

def _resolve_executable(executable_name, qe_bin_dir):
    """Return the runnable executable path, or None if it can't be found."""
    if qe_bin_dir and qe_bin_dir.strip():
        # Resolve to an absolute path: the pre-flight check runs in the server's
        # cwd but the command later runs with cwd=working_directory_path.
        candidate = os.path.abspath(os.path.join(qe_bin_dir.strip(), executable_name))
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
        return None
    return shutil.which(executable_name)


def preflight_run(working_directory_path, executable_name, input_file_name, qe_bin_dir):
    """Validate a run request. Returns (exe_path, error_html); exe_path is None
    exactly when error_html is set."""
    if not working_directory_path:
        return None, "<p style='color:red'>No working directory is open.</p>"
    if not input_file_name:
        return None, "<p style='color:red'>Please select an input file to run.</p>"
    if not os.path.exists(os.path.join(working_directory_path, input_file_name)):
        return None, f"<p style='color:red'>Input file '{input_file_name}' not found.</p>"
    if shutil.which("mpirun") is None:
        return None, "<p style='color:red'>'mpirun' not found on PATH.</p>"
    exe = _resolve_executable(executable_name, qe_bin_dir)
    if exe is None:
        where = f"in '{qe_bin_dir}'" if qe_bin_dir and qe_bin_dir.strip() else "on PATH"
        return None, f"<p style='color:red'>QE executable '{executable_name}' not found {where}.</p>"
    return exe, None


def run_qe_stream(working_directory_path, num_cores, exe_path, input_file_name, output_file_name):
    """Run one QE command, streaming stdout. Reusable core (no Gradio).

    Yields (cumulative_log, returncode): returncode is None while running and the
    process exit code on the final yield. Tees stdout to output_file_name and
    registers the process in _current_process so the Stop button can reach it.
    """
    global _current_process
    out_path = os.path.join(working_directory_path, output_file_name)

    # Quote exe/input names: file names may legitimately contain spaces or other
    # shell-special characters. start_new_session lets us kill the whole process
    # group (mpirun + ranks) on Stop.
    command = (f"mpirun -np {int(num_cores)} "
               f"{shlex.quote(exe_path)} -in {shlex.quote(input_file_name)}")
    process = subprocess.Popen(
        args=command, cwd=working_directory_path, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1, universal_newlines=True, start_new_session=True,
    )
    _current_process = process

    output_log = ""
    with open(out_path, "w") as out_file:
        for line in process.stdout:
            output_log += line
            out_file.write(line)
            out_file.flush()
            yield output_log, None

    process.wait()
    _current_process = None
    yield output_log, process.returncode


def write_relaxed_cif(working_directory_path, input_file_name, output_file_name):
    """After a relax/vc-relax pw.x run, save the relaxed structure as <base>.cif
    (base = the output file name without its extension), so it can seed further
    calculations. Returns the .cif filename, or None if not applicable.

    The input file is read for its calculation type and prefix; the relaxed
    structure is taken from that prefix's XML (out/<prefix>.xml).
    """
    try:
        with open(os.path.join(working_directory_path, input_file_name)) as fh:
            text = fh.read()
        m_calc = re.search(r"calculation\s*=\s*['\"]([^'\"]+)['\"]", text, re.I)
        if not m_calc or m_calc.group(1).strip().lower() not in RELAX_TYPES:
            return None
        m_prefix = re.search(r"prefix\s*=\s*['\"]([^'\"]+)['\"]", text, re.I)
        prefix = m_prefix.group(1).strip() if m_prefix else DEFAULT_PREFIX

        xml_path = os.path.join(working_directory_path, "out", f"{prefix}.xml")
        if not os.path.exists(xml_path):
            return None

        import warnings
        from pymatgen.io.espresso.outputs import PWxml
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structure = PWxml(xml_path).final_structure

        base = os.path.splitext((output_file_name or "").strip()
                                or default_output_name(input_file_name))[0]
        cif_name = f"{base}.cif"
        structure.to(filename=os.path.join(working_directory_path, cif_name))
        return cif_name
    except Exception:
        return None


def on_run_calculation(working_directory_path, num_cores_slider, executable_name,
                       input_file_name, output_file_name, qe_bin_dir):
    """Gradio wrapper over run_qe_stream: pre-flight, then stream with status text."""
    global _current_process
    try:
        exe, err = preflight_run(working_directory_path, executable_name, input_file_name, qe_bin_dir)
        if err:
            yield err, ""
            return

        out_name = (output_file_name or "").strip() or default_output_name(input_file_name)
        for output_log, rc in run_qe_stream(working_directory_path, num_cores_slider,
                                            exe, input_file_name, out_name):
            if rc is None:
                yield f"<p>Running {executable_name} on {input_file_name}...</p>", output_log
            elif rc == 0:
                cif = write_relaxed_cif(working_directory_path, input_file_name, out_name)
                extra = f" Relaxed structure saved to {cif}." if cif else ""
                yield (f"<p style='color:green'>{executable_name} finished successfully! "
                       f"(log saved to {out_name}){extra}</p>", output_log)
            elif rc < 0:
                yield (f"<p style='color:orange'>Calculation was stopped "
                       f"(signal {-rc}).</p>", output_log)
            else:
                yield (f"<p style='color:red'>{executable_name} exited with error code "
                       f"{rc}</p>", output_log)

    except Exception as e:
        _current_process = None
        yield f"<p style='color:red'>Error running QE: {e}</p>", None


def on_stop_calculation():
    global _current_process
    process = _current_process
    if process is not None and process.poll() is None:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception as e:
            return f"<p style='color:red'>Could not stop calculation: {e}</p>"
        return "<p style='color:orange'>Stopping calculation...</p>"
    return "<p>No calculation is currently running.</p>"


def refresh_file_list(working_directory_path):
    """Re-scan the working dir so newly written output files show up. Writing to
    working_directory_file_list_state cascades to the file table and dropdowns."""
    if not working_directory_path:
        return []
    return get_files_in_working_directory(working_directory_path)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

def calculation_tab_content(working_directory_path_state, working_directory_file_list_state, status_markdown):
    with gr.Tab("Calculation") as calculation_tab:
        with gr.Accordion("Generate Quantum Espresso Input File"):
            with gr.Row():
                with gr.Column(scale=1):
                    calculation_type_dropdown = gr.Dropdown(choices=CALC_TYPES, value="scf", label="Calculation Type")
                    input_structure_file_name_dropdown = gr.Dropdown(choices=[], value=None, label="Input Structure")
                    pseudo_set_dropdown = gr.Dropdown(
                        choices=list_pseudo_sets(), value=default_pseudo_set(),
                        label="Pseudopotential Set", allow_custom_value=True,
                        info=f"Subfolders of $ESPRESSO_PSEUDO ({PSEUDO_ROOT}); "
                             "you can also type a full path")
                with gr.Column(scale=1):
                    ecutwfc_slider = gr.Slider(minimum=20, maximum=120, step=5, value=50, label="ecutwfc (Ry) — wavefunction cutoff")
                    ecutrho_slider = gr.Slider(minimum=80, maximum=960, step=20, value=400, label="ecutrho (Ry) — charge-density cutoff")
                    with gr.Row():
                        kx_slider = gr.Slider(minimum=1, maximum=16, step=1, value=4, label="k-grid x")
                        ky_slider = gr.Slider(minimum=1, maximum=16, step=1, value=4, label="k-grid y")
                        kz_slider = gr.Slider(minimum=1, maximum=16, step=1, value=4, label="k-grid z")
            with gr.Row():
                functional_dropdown = gr.Dropdown(
                    choices=list(FUNCTIONALS.keys()), value=DEFAULT_FUNCTIONAL,
                    label="Functional (XC)")
                custom_functional_textbox = gr.Textbox(
                    value="", label="Custom input_dft (Libxc code or name)",
                    placeholder="used only when Functional = Custom, e.g. r2scan or XC-000i-000i-000i-000i-263L-267L")
            with gr.Row():
                input_file_name_textbox = gr.Textbox(
                    value=default_input_name("scf"), label="Input File Name")
                output_name_textbox = gr.Textbox(
                    value=DEFAULT_PREFIX,
                    label="Output Name (QE prefix → out/<name>.xml)")
            with gr.Row():
                extra_settings_textbox = gr.Textbox(
                    label="Extra namelist settings (one 'namelist.key = value' per line)",
                    placeholder="system.nbnd = 24\nelectrons.conv_thr = 1e-8\nsystem.nspin = 2",
                    lines=4)
            with gr.Row():
                generate_button = gr.Button("Generate Input File")
        with gr.Accordion("Run Calculation"):
            with gr.Row():
                num_cores = max(1, (os.cpu_count() or 2) // 2)
                num_cores_slider = gr.Slider(minimum=1, maximum=num_cores, step=1, value=1, label="Number of Cores")
                qe_executable_dropdown = gr.Dropdown(choices=QE_EXECUTABLES, value="pw.x", label="QE Executable")
                input_file_dropdown = gr.Dropdown(choices=[], value=None, label="Input File to Run")
            with gr.Row():
                output_file_textbox = gr.Textbox(
                    label="Output File Name", value=default_output_name(default_input_name("scf")),
                    placeholder="e.g. scf.out")
                qe_bin_dir_textbox = gr.Textbox(
                    label="QE binary directory (optional)",
                    placeholder="Leave blank to use PATH, e.g. /opt/qe/bin")
            with gr.Row():
                run_button = gr.Button("Run", variant="primary")
                stop_button = gr.Button("Stop", variant="stop")
            with gr.Row():
                output_log_textbox = gr.Textbox(label="QE Output Log", lines=10)

        working_directory_file_list_state.change(
            on_working_directory_file_list_change,
            [input_file_dropdown, working_directory_file_list_state],
            [input_structure_file_name_dropdown, input_file_dropdown])

        calculation_type_dropdown.change(
            on_select_calculation_type, calculation_type_dropdown,
            [input_structure_file_name_dropdown, kx_slider, ky_slider, kz_slider,
             qe_executable_dropdown, input_file_name_textbox, output_file_textbox])

        input_file_dropdown.change(on_select_input_file, input_file_dropdown, output_file_textbox)

        generate_button.click(
            on_generate_qe_input,
            [working_directory_path_state, calculation_type_dropdown, input_structure_file_name_dropdown,
             pseudo_set_dropdown, ecutwfc_slider, ecutrho_slider, kx_slider, ky_slider, kz_slider,
             extra_settings_textbox, input_file_name_textbox, output_name_textbox,
             functional_dropdown, custom_functional_textbox],
            [status_markdown, working_directory_file_list_state, input_file_dropdown])

        run_event = run_button.click(
            on_run_calculation,
            [working_directory_path_state, num_cores_slider, qe_executable_dropdown,
             input_file_dropdown, output_file_textbox, qe_bin_dir_textbox],
            [status_markdown, output_log_textbox])
        # Refresh the file list once the run finishes so output files appear.
        run_event.then(refresh_file_list, working_directory_path_state, working_directory_file_list_state)

        stop_event = stop_button.click(on_stop_calculation, None, status_markdown, cancels=[run_event])
        # A stopped run may have produced partial output; refresh too.
        stop_event.then(refresh_file_list, working_directory_path_state, working_directory_file_list_state)

    return calculation_tab
