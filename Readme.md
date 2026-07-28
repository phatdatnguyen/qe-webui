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
```


## Start web UI
To start the web UI:

```
source qe-env/bin/activate
python3 webui.py
```