#!/usr/bin/env python3
"""Run an SG2260E CModel or PCIe validation matrix one process at a time.

This script never resets hardware. A timeout permanently poisons the selected
output directory. Run the complete CModel matrix before starting a PCIe run.
"""
import argparse
from contextlib import ExitStack, suppress
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def cases(path):
    # Importing TileLang is safe here, but use a separate listing process so the
    # supervisor never reserves a runtime identity or loads a TPU runtime.
    code = 'import runpy,json; print(json.dumps(runpy.run_path(' + repr(str(path)) + ')["CASES"]))'
    result = subprocess.run([sys.executable, '-c', code],
                            capture_output=True,
                            text=True,
                            check=True)
    return json.loads(result.stdout.splitlines()[-1])


def board_state(smi, device_id, destination, *, settle=False):
    # Only a successfully exited kernel may need time for utilization sampling
    # to settle. Faults, retained memory, malformed status and command errors
    # fail immediately. Never launch another kernel during this bounded wait.
    command = [str(smi), '--noloop', '--json_format', f'--dev={device_id}']
    idle_samples = 0
    for attempt in range(10 if settle else 1):
        sample = destination.with_name(
            f'{destination.stem}-{attempt}.json') if settle else destination
        with sample.open('w') as output:
            result = run_child(command, output, 15)
        if result.returncode:
            raise RuntimeError(f'Board status command failed: {sample}')
        state = json.loads(sample.read_text())
        chips = [
            chip for card in state.values() if isinstance(card, dict) for chip in card.values()
            if isinstance(chip, dict) and 'status' in chip
        ]
        if not chips or any(c['status'] != 'Active' or c.get('mem_usage') != '0MB' for c in chips):
            raise RuntimeError(f'Board is not quiescent: {sample}')
        for chip in chips:
            util = chip.get('tpu_util', '')
            if not util.endswith('%') or not util[:-1].isdigit() or not 0 <= int(util[:-1]) <= 100:
                raise RuntimeError(f'Board is not quiescent: invalid utilization: {sample}')
        idle_samples = idle_samples + 1 if all(c['tpu_util'] == '0%' for c in chips) else 0
        if idle_samples >= (2 if settle else 1):
            if settle:
                destination.write_text(sample.read_text())
            return
        if settle and attempt < 9:
            time.sleep(1)
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
    p.add_argument('--model', choices=('rv', 'tpukernel'), default='rv')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=180)
    p.add_argument('--smi', type=Path, help='Required for PCIe pre/post-launch idle checks')
    p.add_argument('--case', action='append', dest='selected_cases', metavar='SCRIPT/CASE')
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    poison = a.output_dir / 'POISONED'
    if poison.exists():
        raise SystemExit(
            'Run root is poisoned; investigate and confirm device recovery before a new run.')
    scripts = [ROOT / 'testing/python/jit' / 'test_tpu_llama_ops.py']
    if a.model == 'rv':
        scripts.insert(0, ROOT / 'testing/python/jit' / 'test_tpu_rv_essential_ops.py')
    tasks = [(s, c) for s in scripts for c in cases(s)]
    if a.selected_cases:
        available = {f'{script.stem}/{case}': (script, case) for script, case in tasks}
        unknown = [case for case in a.selected_cases if case not in available]
        if unknown:
            raise SystemExit('Unknown validation case(s): ' + ', '.join(unknown))
        if len(set(a.selected_cases)) != len(a.selected_cases):
            raise SystemExit('Duplicate --case selectors are not allowed')
        tasks = [available[case] for case in a.selected_cases]
    device_id = int(os.environ.get('TILELANG_TPU_DEVICE_ID', '0'))
    if a.runtime == 'pcie' and (not a.smi or not a.smi.is_file()):
        raise SystemExit('PCIe requires --smi pointing to board tpu-smi')
    if a.runtime == 'pcie' and os.environ.get('TILELANG_TPU_ALLOW_PCIE_LOAD') != '1':
        raise SystemExit('PCIe requires TILELANG_TPU_ALLOW_PCIE_LOAD=1 before supervision starts')
    with ExitStack() as stack:
        if a.runtime == 'pcie':
            from tilelang.jit.adapter.tpu_profiling import (_exclusive_pcie_device_lock,
                                                            _quarantine_pcie_device)
            stack.enter_context(_exclusive_pcie_device_lock(device_id))
        summary = {
            'runtime': a.runtime,
            'programming_model': a.model,
            'passed_cases': [],
        }
        for script, case in tasks:
            path = a.output_dir / script.stem / case
            path.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(script), '--case', case, '--runtime', a.runtime, '--output-dir',
                str(path)
            ]
            if script.name == 'test_tpu_llama_ops.py':
                command.extend(['--model', a.model])
            print(f'{a.runtime}: {script.stem}/{case}', flush=True)
            with (path / 'run.log').open('w') as log:
                try:
                    if a.runtime == 'pcie':
                        board_state(a.smi, device_id, path / 'board-before.json')
                    proc = run_child(command, log, a.timeout)
                    if proc.returncode:
                        raise RuntimeError(f'exit {proc.returncode}')
                    if a.runtime == 'pcie':
                        board_state(a.smi, device_id, path / 'board-after.json', settle=True)
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
            summary['passed_cases'].append(f'{script.stem}/{case}')
            (a.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
