#!/usr/bin/env python3
"""Serial fail-stop CModel/PCIe validation; each case gets a fresh process.

PCIe requires a successful CModel manifest for the identical source fingerprint.
This script never resets hardware. A timeout permanently poisons this run root.
The caller must establish exclusive device access before a PCIe run.
"""
import argparse
from contextlib import ExitStack, suppress
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def fingerprint():
    h = hashlib.sha256()
    for directory in ('src', 'tilelang', 'testing/python/jit', '3rdparty/tvm/src',
                      '3rdparty/tvm/include', 'tools'):
        for path in sorted((ROOT / directory).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.cc', '.h',
                                                  '.cpp') and not {'__pycache__', '.cycache'}.intersection(path.parts):
                h.update(str(path.relative_to(ROOT)).encode())
                h.update(path.read_bytes())
    return h.hexdigest()


def sdk_identity():
    root = Path(os.environ["PPL_PROJECT_ROOT"]).resolve()
    digest = hashlib.sha256()
    # The used chip headers and device/CModel libraries must travel together.
    paths = sorted((root / "deps/chip/tpub_7_1_e").rglob("*"))
    for path in paths:
        if path.is_file() and (path.suffix == ".h" or ".so" in path.name):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def cases(path):
    # Importing TileLang is safe here, but use a separate listing process so the
    # supervisor never reserves a runtime identity or loads a TPU runtime.
    code = 'import runpy,json; print(json.dumps(runpy.run_path(' + repr(str(path)) + ')["CASES"]))'
    result = subprocess.run([sys.executable, '-c', code],
                            capture_output=True,
                            text=True,
                            check=True)
    return json.loads(result.stdout.splitlines()[-1])


def board_state(smi, device_id, destination):
    command = [str(smi), '--noloop', '--json_format', f'--dev={device_id}']
    with destination.open('w') as output:
        result = run_child(command, output, 15)
    if result.returncode:
        raise RuntimeError(f'Board status command failed: {destination}')
    state = json.loads(destination.read_text())
    chips = [
        chip for card in state.values() if isinstance(card, dict) for chip in card.values()
        if isinstance(chip, dict) and 'status' in chip
    ]
    if not chips or any(
            c['status'] != 'Active' or c.get('tpu_util') != '0%' or c.get('mem_usage') != '0MB'
            for c in chips):
        raise RuntimeError(f'Board is not quiescent: {destination}')


def run_child(command, log, timeout):
    # Do not use subprocess.run(timeout=...) here: its timeout cleanup waits
    # unboundedly for a killed process, which may be stuck in the PCIe driver.
    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        code = child.wait(timeout=timeout)
        return subprocess.CompletedProcess(command, code)
    except BaseException as error:
        error.process_group = child.pid
        with suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        # poll is nonblocking; a D-state process remains recorded/quarantined.
        child.poll()
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runtime', choices=('cmodel', 'pcie'), required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--cmodel-manifest', type=Path)
    p.add_argument('--timeout', type=int, default=180)
    p.add_argument('--smi', type=Path, help='Required for PCIe pre/post-launch idle checks')
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    poison = a.output_dir / 'POISONED'
    if poison.exists():
        raise SystemExit(
            'Run root is poisoned; investigate and confirm device recovery before a new run.')
    scripts = [
        ROOT / 'testing/python/jit' / f
        for f in ('test_tpu_rv_essential_ops.py', 'test_tpu_llama_ops.py')
    ]
    tasks = [(s, c) for s in scripts for c in cases(s)]
    identity = fingerprint()
    sdk_hash = sdk_identity()
    device_id = int(os.environ.get('TILELANG_TPU_DEVICE_ID', '0'))
    if a.runtime == 'pcie':
        if not a.smi or not a.smi.is_file():
            raise SystemExit('PCIe requires --smi pointing to board tpu-smi')
        if not a.cmodel_manifest:
            raise SystemExit('PCIe requires --cmodel-manifest')
        proof = json.loads(a.cmodel_manifest.read_text())
        if proof.get('runtime') != 'cmodel' or proof.get('source_sha256') != identity or proof.get(
                'sdk_sha256') != sdk_hash or proof.get('passed_cases') != [
                    f'{s.stem}/{c}' for s, c in tasks
                ]:
            raise SystemExit('CModel proof incomplete or source fingerprint differs')
    with ExitStack() as stack:
        if a.runtime == 'pcie':
            from tilelang.jit.adapter.tpu_profiling import (_exclusive_pcie_device_lock,
                                                            _quarantine_pcie_device)
            stack.enter_context(_exclusive_pcie_device_lock(device_id))
        manifest = {
            'runtime': a.runtime,
            'source_sha256': identity,
            'sdk_sha256': sdk_hash,
            'passed_cases': [],
            'python': sys.version,
            'sdk': os.environ.get('PPL_PROJECT_ROOT'),
            'results': []
        }
        for script, case in tasks:
            path = a.output_dir / script.stem / case
            path.mkdir(parents=True, exist_ok=True)
            if fingerprint() != identity:
                raise SystemExit('Source changed during validation; start a fresh matrix')
            command = [
                sys.executable,
                str(script), '--case', case, '--runtime', a.runtime, '--output-dir',
                str(path)
            ]
            print(f'{a.runtime}: {script.stem}/{case}', flush=True)
            with (path / 'run.log').open('w') as log:
                try:
                    if a.runtime == 'pcie':
                        board_state(a.smi, device_id, path / 'board-before.json')
                    proc = run_child(command, log, a.timeout)
                    if proc.returncode:
                        raise RuntimeError(f'exit {proc.returncode}')
                    if a.runtime == 'pcie':
                        board_state(a.smi, device_id, path / 'board-after.json')
                except BaseException as e:
                    if a.runtime == 'pcie':
                        _quarantine_pcie_device(
                            device_id,
                            process_group=getattr(e, "process_group", None),
                            reason=f'Llama validation failed: {case}')
                    poison.write_text(
                        json.dumps({
                            'case': case,
                            'command': command,
                            'error': str(e)
                        }))
                    raise SystemExit(
                        f'STOP: {case}: {e}; log: {path}/run.log. No further launch or automatic reset.'
                    ) from e
            manifest['passed_cases'].append(f'{script.stem}/{case}')
            hashes = {
                str(f.relative_to(path)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in path.iterdir()
                if f.suffix in ('.npy', '.c')
            }
            import torch
            tensor_files = list(path.glob('*.pt'))
            output = torch.load(tensor_files[0], map_location='cpu', weights_only=True)['output']
            output_hash = hashlib.sha256(output.contiguous().view(
                torch.uint8).numpy().tobytes()).hexdigest()
            manifest['results'].append({
                'case': case,
                'files': hashes,
                'output_sha256': output_hash
            })
            (a.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
