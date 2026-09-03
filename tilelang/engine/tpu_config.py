# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""TPU target capabilities and compilation configuration.

The public selection is deliberately split into three independent axes:

* chip: the physical TPU generation (``bm1690`` or ``sg2260e``);
* device mode: the device programming model (``tpukernel`` or ``rv``);
* runtime mode: the host execution environment (``cmodel`` or ``pcie``).

``tpu -mcpu=<chip>`` is the canonical TVM Target spelling. The historical
``chip=`` API and the old Target ``-model=`` spelling are accepted at the
boundary and normalized here, so the rest of the pipeline never has to infer a
chip from a PPL directory name.
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple
import re
import warnings


# ``atomic`` is kept solely as an input compatibility alias. It never meant a
# TIR atomic operation: it selected the normal PPL / TPU-Kernel command stream.
TPUDeviceMode = Literal["tpukernel", "rv", "atomic"]
TPUProgrammingModel = Literal["tpukernel", "rv"]
TPURuntimeMode = Literal["pcie", "cmodel"]


@dataclass(frozen=True)
class TPUChipSpec:
    """Stable capability description for one supported physical TPU chip."""

    name: str
    ppl_arch: str
    ppl_compile_definitions: Tuple[str, ...]
    physical_core_count: int
    programming_models: Tuple[TPUProgrammingModel, ...]

    def supports(self, programming_model: str) -> bool:
        return programming_model in self.programming_models


# BM1690 and SG2260E share the TPUv7 local-memory geometry used by the current
# TileLang allocator/codegen. The different SDK architecture and physical core
# count are represented here rather than inferred from a directory layout.
TPU_CHIP_SPECS: Dict[str, TPUChipSpec] = {
    "bm1690": TPUChipSpec(
        name="bm1690",
        ppl_arch="tpub_7_1",
        ppl_compile_definitions=("__tpub_7_1__", "__sg2260__"),
        physical_core_count=8,
        programming_models=("tpukernel",),
    ),
    "sg2260e": TPUChipSpec(
        name="sg2260e",
        ppl_arch="tpub_7_1_e",
        ppl_compile_definitions=("__tpub_7_1_e__", "__sg2260e__"),
        physical_core_count=4,
        programming_models=("tpukernel", "rv"),
    ),
}


def _normalise_chip_name(chip: str) -> str:
    if not isinstance(chip, str):
        raise TypeError(f"TPU chip must be a string, got {type(chip).__name__}")
    normalised = chip.strip().lower()
    if not normalised:
        raise ValueError("TPU chip must not be empty")
    return normalised


def _looks_like_tpu_chip_name(value: str) -> bool:
    """Whether a legacy Target.model value is clearly intended as a TPU SKU.

    Target.model is normally generic workload metadata, so unknown values must
    remain valid.  A value such as ``sg2260erv`` or ``bm1684`` is different: it
    looks like a device selection and falling back to BM1690 would compile for
    the wrong chip.  Reject those values unless they are in the capability
    registry.
    """
    return re.fullmatch(r"(?:bm|sg)\d[\w-]*", value.lower()) is not None


def get_tpu_chip_spec(chip: str) -> TPUChipSpec:
    """Return a supported chip capability record or fail before toolchain use."""
    normalised = _normalise_chip_name(chip)
    try:
        return TPU_CHIP_SPECS[normalised]
    except KeyError as exc:
        supported = ", ".join(TPU_CHIP_SPECS)
        raise ValueError(
            f"Unsupported TPU chip {chip!r}; supported chips: {supported}") from exc


def _normalise_programming_model(device_mode: TPUDeviceMode) -> TPUProgrammingModel:
    if device_mode == "atomic":
        warnings.warn(
            "TPU device_mode='atomic' was a misleading name for the PPL "
            "TPU-Kernel path and is deprecated; use device_mode='tpukernel'.",
            DeprecationWarning,
            stacklevel=3,
        )
        return "tpukernel"
    if device_mode not in ("tpukernel", "rv"):
        raise ValueError(
            f"Unsupported TPU device mode {device_mode!r}; expected 'tpukernel' or 'rv'")
    return device_mode


def get_tpu_target_chip(target: Any) -> Optional[str]:
    """Read the chip selected by a TPU ``Target``.

    ``-mcpu`` is canonical because it is a registered TVM target attribute for
    a concrete processor. ``-model`` is accepted only for compatibility with
    early SG2260E experiments. Supplying both with different values is an
    error rather than an order-dependent choice.
    """
    kind = getattr(getattr(target, "kind", None), "name", None)
    if kind != "tpu":
        return None

    attrs = getattr(target, "attrs", {})
    raw_mcpu = attrs.get("mcpu")
    mcpu_value = str(raw_mcpu).strip() if raw_mcpu is not None else ""
    selected_chip = (
        get_tpu_chip_spec(mcpu_value).name
        if mcpu_value and mcpu_value != "unknown" else None)

    # Target.model is a generic TVM workload annotation. Only interpret it as
    # the early TPU chip spelling when it names a known TPU chip; otherwise it
    # remains independent metadata and must not prevent -mcpu selection.
    raw_model = attrs.get("model")
    model_value = str(raw_model).strip() if raw_model is not None else ""
    model_chip = None
    if model_value and model_value != "unknown":
        try:
            model_chip = get_tpu_chip_spec(model_value).name
        except ValueError as exc:
            # A generic workload model is allowed alongside -mcpu, but a
            # chip-looking value is an attempted device selection.  Do not
            # ignore it merely because a valid -mcpu is also present: that
            # would hide a typo such as ``-model=sg2260erv``.
            if _looks_like_tpu_chip_name(model_value):
                raise ValueError(
                    "Unsupported TPU chip in legacy target model attribute: "
                    f"model={model_value!r}. Use a supported "
                    "tpu -mcpu=<chip> target instead.") from exc
            if selected_chip is None:
                return None

    if selected_chip is not None and model_chip is not None and model_chip != selected_chip:
        raise ValueError(
            "Conflicting TPU chip target attributes: "
            f"mcpu={selected_chip}, model={model_chip}")
    return selected_chip or model_chip


@dataclass(frozen=True)
class TPUCompileConfig:
    """A validated chip, device-programming-model, and runtime selection."""

    chip: str = "bm1690"
    device_mode: TPUDeviceMode = "tpukernel"
    runtime_mode: Optional[TPURuntimeMode] = None

    def __post_init__(self):
        spec = get_tpu_chip_spec(self.chip)
        programming_model = _normalise_programming_model(self.device_mode)
        if not spec.supports(programming_model):
            supported = ", ".join(spec.programming_models)
            raise ValueError(
                f"TPU chip {spec.name!r} does not support device_mode="
                f"{programming_model!r}; supported modes: {supported}")
        selected_runtime = self.runtime_mode
        if selected_runtime is None:
            selected_runtime = (
                "cmodel" if spec.name == "sg2260e" or programming_model == "rv" else "pcie")
        if selected_runtime not in ("pcie", "cmodel"):
            raise ValueError(
                f"Unsupported TPU runtime mode {selected_runtime!r}; expected 'pcie' or 'cmodel'")
        object.__setattr__(self, "chip", spec.name)
        object.__setattr__(self, "device_mode", programming_model)
        object.__setattr__(self, "runtime_mode", selected_runtime)

    @property
    def programming_model(self) -> TPUProgrammingModel:
        """Clearer semantic name for the canonical ``device_mode`` value."""
        return self.device_mode

    @property
    def chip_spec(self) -> TPUChipSpec:
        return get_tpu_chip_spec(self.chip)


def resolve_tpu_compile_config(
    *,
    chip: Optional[str] = None,
    device_mode: TPUDeviceMode = "tpukernel",
    runtime_mode: Optional[TPURuntimeMode] = None,
    mode: Optional[TPURuntimeMode] = None,
    target_chip: Optional[str] = None,
) -> TPUCompileConfig:
    """Resolve legacy API arguments and a TPU Target into one configuration.

    An explicit ``chip=`` and ``tpu -mcpu=`` must agree. SG2260E and RV default
    to CModel so a newly selected target cannot silently attempt board
    initialization; BM1690's historical TPU-Kernel default remains PCIe.
    PCIe still requires the explicit loader safety gate.
    """
    if runtime_mode is not None and mode is not None and runtime_mode != mode:
        raise ValueError(
            f"Conflicting TPU runtime modes: runtime_mode={runtime_mode!r}, mode={mode!r}")

    explicit_chip = get_tpu_chip_spec(chip).name if chip is not None else None
    selected_target_chip = (
        get_tpu_chip_spec(target_chip).name if target_chip is not None else None)
    if explicit_chip is not None and selected_target_chip is not None and \
            explicit_chip != selected_target_chip:
        raise ValueError(
            "Conflicting TPU chip selections: "
            f"chip={explicit_chip!r}, target chip={selected_target_chip!r}")

    selected_chip = explicit_chip or selected_target_chip or "bm1690"
    programming_model = _normalise_programming_model(device_mode)
    selected_runtime = runtime_mode or mode
    if selected_runtime is None:
        # SG2260E is being brought up CModel-first. BM1690 retains the old
        # default for source compatibility, while any RV selection is always
        # CModel-first regardless of chip.
        selected_runtime = (
            "cmodel" if selected_chip == "sg2260e" or programming_model == "rv" else "pcie")
    return TPUCompileConfig(
        chip=selected_chip,
        device_mode=programming_model,
        runtime_mode=selected_runtime,
    )


def bind_tpu_target(target: Any, config: TPUCompileConfig, target_host: Any = None):
    """Return a TPU Target carrying the validated canonical ``-mcpu`` chip.

    This is deliberately kept at the API boundary: target-bound passes and C++
    codegen receive the same chip that PPL toolchain resolution later uses.
    Non-TPU targets are left untouched.
    """
    kind = getattr(getattr(target, "kind", None), "name", None)
    if kind != "tpu":
        return target

    selected_target_chip = get_tpu_target_chip(target)
    if selected_target_chip is not None and selected_target_chip != config.chip:
        raise ValueError(
            "Conflicting TPU chip selections: "
            f"target chip={selected_target_chip!r}, config chip={config.chip!r}")

    # Delay importing Target until this boundary to keep the capability registry
    # usable in lightweight tooling and unit tests.
    from tvm.target import Target

    attrs = {str(name): value for name, value in target.attrs.items()}
    attrs["kind"] = "tpu"
    attrs["mcpu"] = config.chip
    # This is an internal target attribute, not a second public user-facing
    # spelling.  It carries the already validated programming model across the
    # Python/C++ boundary so direct native codegen cannot infer it from extern
    # names or compiler flags.
    raw_programming_model = attrs.get("tpu-programming-model")
    if raw_programming_model is not None:
        target_programming_model = str(raw_programming_model).strip()
        if target_programming_model and target_programming_model != "unknown" and \
                target_programming_model != config.programming_model:
            raise ValueError(
                "Conflicting TPU programming-model selections: "
                f"target={target_programming_model!r}, "
                f"config={config.programming_model!r}")
    attrs["tpu-programming-model"] = config.programming_model
    # Remove only a legacy model-as-chip spelling. A normal Target.model may
    # describe a workload and remains useful metadata for unrelated tooling.
    raw_model = attrs.get("model")
    if raw_model is not None:
        try:
            model_chip = get_tpu_chip_spec(str(raw_model)).name
        except ValueError:
            model_chip = None
        if model_chip == config.chip:
            del attrs["model"]
    return Target(attrs, target_host) if target_host is not None else Target(attrs)
