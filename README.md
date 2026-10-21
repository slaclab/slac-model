# slac-model

Accelerator model utilities for SLAC, including PV-equivalent conversions for
Bmad, Impact, and Cheetah, and helpers for building LCLS / FACET models.

## Installation

Python 3.11 or newer is required. Conda (for example, Miniforge) is recommended
for installing the compiled simulation backends.

Create an environment:

```bash
# Only needed when creating a new environment:
conda create -n slac-model -c conda-forge python=3.13 pip
conda activate slac-model
```

Clone this repository and install the package with the backend you need. For
the LCLS Bmad models:

```bash
git clone https://github.com/slaclab/slac-model.git
cd slac-model
conda install -c conda-forge "pytao>=1"
python -m pip install -e ".[bmad]"
```

Installing PyTao through Conda also provides its compiled Tao/Bmad dependencies;
installing the Python packages alone is not a substitute for those libraries.
The `-e` option makes local source changes available without reinstalling.
Omit it for a non-editable installation.

The available extras are defined in [pyproject.toml](./pyproject.toml):

Impact simulations require the Impact-T executable in addition to the `impact`
extra. For the core utilities without optional backends, use
`python -m pip install -e .`.

## LCLS configuration

The LCLS lattice files are maintained separately in
[slaclab/lcls-lattice](https://github.com/slaclab/lcls-lattice).
The CU_HXR helpers require the `LCLS_LATTICE` environment variable to point to
the **root of that repository**, not its `bmad` subdirectory.

For example, clone the lattice into a directory:

```bash
mkdir -p "$HOME/lattices"
git clone https://github.com/slaclab/lcls-lattice.git "$HOME/lattices/lcls-lattice"
export LCLS_LATTICE="$HOME/lattices/lcls-lattice"
```

If you already have a local or site-managed lattice checkout, set
`LCLS_LATTICE` to its absolute path instead. Avoid spaces in the path.

| Environment variable | Required for | Value |
| --- | --- | --- |
| `LCLS_LATTICE` | LCLS CU_HXR model construction | Absolute path to the `lcls-lattice` checkout |


### Construct a CU_HXR model

With the `bmad` extra installed and `LCLS_LATTICE` configured:

```python
from slac_model.models.cu import get_cu_hxr_bmad_model

tao = get_cu_hxr_bmad_model(start_element="OTR2", end_element="END")
```

To construct the lume-bmad virtual accelerator instead:

```python
from slac_model.models.cu import get_cu_hxr_bmad_va

va = get_cu_hxr_bmad_va(start_element="OTR2", end_element="END")
```

## Running tests

From the repository root, with the desired backends and the `dev` extra
installed:

```bash
python -m pytest slac_model/tests
```

The Bmad tests use a bundled test lattice and do not require `LCLS_LATTICE`.
Backend tests may be skipped when their optional Python dependencies are not
installed; use the `all,dev` installation above for all backends.
