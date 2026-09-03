# TPU JIT templates

This directory contains checked-in source templates only. A TPU JIT build
creates `kernel.c`, `kernel.cpp`, `kernel.h`, `libkernel.so`, `main.cpp`, and
`main.so` in a private temporary directory owned by its `LibraryGenerator`.
It never writes generated artifacts back here.

The generated host source receives the absolute path to its matching private
`libkernel.so` through the compile-time `TILELANG_PPL_KERNEL_PATH` definition.
There is no process-global `PPL_KERNEL_PATH` fallback for JIT execution.
Consequently, a prebuilt TPU `main.so` cannot be loaded through the generic
cache/database path until TileLang ships a manifest that bundles and validates
its private device library, PPL SDK/runtime identity, chip, programming model,
and runtime configuration.

The artifact names remain in `.gitignore` only to prevent stale files produced
by old/manual workflows from being committed; they are not JIT output paths.
