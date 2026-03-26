# Installation Guide

## Install with uv

This is the most direct way to set up a local environment for the repo and run the TI2V web demo.

```bash
uv venv .venv --python 3.10
uv sync --python .venv/bin/python --extra dev --extra demo
```

`flash-attn` is not required for the TI2V demo. If you want the acceleration path later, install it separately after the base environment is working:

```bash
uv pip install --python .venv/bin/python --upgrade pip setuptools wheel
uv pip install --python .venv/bin/python flash-attn --no-build-isolation
```

### Launch the TI2V Web Demo

The app defaults to `/mnt/nas10_shared/models/Wan2.2-TI2V-5B`, which matches the shared model location on this machine.

```bash
uv run --python .venv/bin/python python app_ti2v.py --host 0.0.0.0 --port 7860
```

If you want to point to another checkpoint path:

```bash
WAN_TI2V_CKPT_DIR=/path/to/Wan2.2-TI2V-5B \
uv run --python .venv/bin/python python app_ti2v.py --host 0.0.0.0 --port 7860
```

## Install with pip

```bash
pip install .
pip install .[dev]  # Installe aussi les outils de dev
```

## Install with Poetry

Ensure you have [Poetry](https://python-poetry.org/docs/#installation) installed on your system.

To install all dependencies:

```bash
poetry install
```

### Handling `flash-attn` Installation Issues

If `flash-attn` fails due to **PEP 517 build issues**, you can try one of the following fixes.

#### No-Build-Isolation Installation (Recommended)
```bash
poetry run pip install --upgrade pip setuptools wheel
poetry run pip install flash-attn --no-build-isolation
poetry install
```

#### Install from Git (Alternative)
```bash
poetry run pip install git+https://github.com/Dao-AILab/flash-attention.git
```

---

### Running the Model

Once the installation is complete, you can run **Wan2.2** using:

```bash
poetry run python generate.py --task t2v-A14B --size '1280*720' --ckpt_dir ./Wan2.2-T2V-A14B --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

#### Test
```bash
bash tests/test.sh
```

#### Format
```bash
black .
isort .
```
