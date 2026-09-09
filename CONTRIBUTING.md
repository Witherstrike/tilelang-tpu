# Contributing

Contributions to TileLang-TPU are welcome. Keep each change focused and add
tests and documentation for new or changed behavior.

## Report a bug

Search the issue tracker before opening a new report. Include:

- A minimal reproducer
- The target chip, programming model, and runtime mode
- Expected and actual behavior
- Relevant compiler or runtime logs

## Set up the repository

Follow the [installation guide](docs/get_started/Installation.md) for the TPU
development environment. It uses the bundled TVM submodule, a Python virtual
environment, and `./build_tpu.sh`.

## Run tests and formatting

Start with the tests closest to the modified code, then run the TPU test suite
described in the project [README](README.md#development).

```bash
./format.sh
```

Run hardware cases through the staged CModel and PCIe process in the
[TPU demo guide](tpu_demo/README.md). The PCIe runner serializes device access
and stops at the first error.

## Submit a pull request

The pull request description should explain:

- What changed and why
- Which chip, programming model, and runtime combinations are affected
- Which tests were run
- Any remaining limits

Use an explicit compile-time error for unsupported target combinations. A
backend must not silently fall back to another programming model.
