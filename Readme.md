## Introduction
This web UI is for computational calculations with [Quantum Espresso](https://www.quantum-espresso.org/):

* Single-Point Calculation

* Geometry Optimization


## Installation  (Linux only)
- Clone this repo: Open terminal

```
git clone https://github.com/phatdatnguyen/qe-webui
```

- Create and activate virtual environment:

```
cd qe-webui
python3 -m venv qe-env
source qe-env/bin/activate
```

- Install packages:

```
pip install pymatgen
pip install ase
pip install nglview==4.0
pip install gradio
pip install git+https://github.com/Griffin-Group/pymatgen-io-espresso
```


## Pseudopotentials

The web UI looks for pseudopotentials in the folder named by the `$ESPRESSO_PSEUDO`
environment variable — the same variable Quantum Espresso itself uses. Keep each
pseudopotential set (SSSP, ONCV, ...) in its own subfolder of that directory:

```
$ESPRESSO_PSEUDO/
├── SSSP-lib-pbe-eff-v2/
│   ├── Si.pbe-n-rrkjus_psl.1.0.0.UPF
│   └── ...
├── SSSP-lib-pbesol-prec-v2/
│   └── ...
└── SG14_ONCV/
    └── ...
```

The **Pseudopotential Set** dropdown (Calculation and Automation tabs) lists these
subfolders, and the chosen one is written as `pseudo_dir` in the generated input
file. Within a set, the `.UPF` file for each element is matched by filename.

- Set the variable in `~/.bashrc` (create the folder first, then add the line):

```
export ESPRESSO_PSEUDO=$HOME/quantum-espresso/q-e-pseudo
```

- Apply it, then start the web UI **from that same shell** — the dropdown is filled
  when the app starts, so a change to the variable or a newly added set needs a restart:

```
source ~/.bashrc
echo $ESPRESSO_PSEUDO      # check it is set
```

Notes:

- If `$ESPRESSO_PSEUDO` is not set, the UI falls back to `~/q-e-pseudo`.
- If you keep `.UPF` files directly in `$ESPRESSO_PSEUDO` instead of subfolders,
  the dropdown offers `.` for that folder itself.
- The dropdown also accepts a typed-in absolute path, for a set kept elsewhere.


## Start web UI
To start the web UI:

```
source qe-env/bin/activate
python3 webui.py
```