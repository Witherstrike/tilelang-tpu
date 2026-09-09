# Contributing

Contributions to TileLang-TPU are welcome. Please keep each change focused and include tests and
documentation for any behavior that it adds or changes.

## Reporting bugs

Search the existing issues before opening a new one. A useful bug report includes:

- a minimal reproducer;
- the target chip, programming model, and runtime mode;
- the expected and actual results;
- the relevant compiler or runtime log.

Do not attach proprietary SDK files or raw data that may contain sensitive information.

## Asking questions

Use the project issue tracker for development and usage questions. Include enough target and
environment information for another contributor to reproduce the problem.

## Repository setup

For TPU development, follow the
[TileLang-TPU installation guide](docs/get_started/Installation.md). It uses the vendored TVM
submodule, a Python virtual environment, and `./build_tpu.sh`. The generic `setup.py` path still
targets the upstream GPU package and is not the TPU-only development workflow.

## Tests and formatting

Run `./format.sh` before submitting a change. Start with the tests closest to the modified code,
then run the TPU-only suite documented in the project [README](README.md#开发检查). Hardware tests
must follow the staged CModel-to-PCIe process in [`tpu_demo/README.md`](tpu_demo/README.md); do not
bypass its device lock or safety checks.

## Pull requests

A pull request should explain what changed, why the change is needed, which target combinations it
affects, and how it was verified. Keep unsupported and unverified combinations explicit instead of
silently falling back to another backend.
