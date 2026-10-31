"""Classroom demo: honest saved replay or explicit live classifier inference.

Saved mode is lightweight while training. Live CPU mode does not allocate CUDA
memory. Live GPU mode is refused while the controlled training queue is active.
"""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
QUEUE = ROOT / 'reports/research_review/studies/classifier_source_clean_context_study_v1/queue_status.json'
LABELS = {'OBJECTIVE', 'REQUIREMENT', 'ACTIVITY', 'EXPECTED_RESULT', 'OTHER'}


def training_active():
    try:
        return json.loads(QUEUE.read_text(encoding='utf-8')).get('state') == 'running'
    except FileNotFoundError:
        return False
    except json.JSONDecodeError:
        # The queue may be publishing a status update. Be conservative about CUDA.
        return True


def read_saved(records, path):
    predictions = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(predictions, list) or len(predictions) != len(records):
        raise ValueError('Saved results do not match the example count')
    for index, (item, prediction) in enumerate(zip(records, predictions)):
        if prediction.get('index') != index or any(prediction.get(key) != item[key] for key in ('id', 'item_id') if key in item):
            raise ValueError('Saved result IDs/order differ from this input; use the original input or live mode')
        probabilities = prediction.get('probabilities', {})
        confidence = prediction.get('confidence')
        if (set(probabilities) != LABELS or any(isinstance(v, bool) or not isinstance(v, (int, float))
                or not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities.values())
                or abs(sum(probabilities.values()) - 1) > 1e-5
                or prediction.get('label') != max(probabilities, key=probabilities.get)
                or isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence)
                or abs(confidence - max(probabilities.values())) > 1e-6):
            raise ValueError('Saved results contain invalid probabilities or label mapping')
    return predictions


def live(records, model_dir, cpu=False):
    if training_active() and not cpu:
        raise ValueError('Training is active. Use --live --cpu, or the default saved-results demo.')
    env_keys = ('CUDA_VISIBLE_DEVICES', 'PYTORCH_NVML_BASED_CUDA_CHECK')
    previous = {key: os.environ.get(key) for key in env_keys}
    try:
        if cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            # Avoid the CUDA runtime availability probe even with GPUs hidden.
            # This machine has recorded native NVIDIA DLL crashes during it.
            os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = '1'
        # Import after configuring this demo process, before torch can inspect CUDA.
        from classifier_predict import predict
        return predict(records, model_dir, batch_size=1)
    finally:
        if cpu:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def run_cpu_worker(args, output):
    """Keep a lightweight console parent alive if a native library crashes."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix('.log')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='-1', PYTORCH_NVML_BASED_CUDA_CHECK='1',
               OMP_NUM_THREADS=str(args.cpu_threads), MKL_NUM_THREADS=str(args.cpu_threads),
               PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    # Prefer the isolated CPU build when installed. The training interpreter
    # and its CUDA packages stay untouched.
    cpu_env = ROOT / '.venv/classifier_demo_cpu'
    cpu_python = cpu_env / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    cpu_version = cpu_env / 'Lib/site-packages/torch/version.py'
    executable = sys.executable
    if cpu_python.is_file() and cpu_version.is_file() and '+cpu' in cpu_version.read_text(encoding='utf-8'):
        executable = str(cpu_python)
    command = [executable, '-u', str(Path(__file__).resolve()), '--live', '--cpu',
               '--_cpu-worker', '--input', str(args.input.resolve()),
               '--model-dir', str(args.model_dir.resolve()), '--output', str(output)]
    print('MODE: LIVE INFERENCE (CPU)', flush=True)
    print('PYTHON:', executable, flush=True)
    print('MODEL:', args.model_dir.resolve(), flush=True)
    print('INPUT:', args.input.resolve(), flush=True)
    print('LOG:', log_path, flush=True)
    print('Loading checkpoint and running inference; please wait...', flush=True)
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    with log_path.open('x', encoding='utf-8') as log:
        with subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                              text=True, encoding='utf-8', errors='replace',
                              bufsize=1, creationflags=flags) as process:
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
    if code:
        print(f'Demo process failed: exit {code} (0x{code & 0xffffffff:08X}). '
              f'Inspect log: {log_path}', file=sys.stderr, flush=True)
        return 1
    if not output.is_file():
        print('Demo process exited without a prediction file. Inspect log: ' + str(log_path),
              file=sys.stderr, flush=True)
        return 1
    print('DEMO COMPLETE. Predictions saved to:', output, flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'examples/classifier_input.json')
    parser.add_argument('--saved-results', type=Path, default=ROOT / 'reports/classifier/classifier_demo_predictions.json')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--_cpu-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'models/controlled_study/classifier_stability_v1/seed_123')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
    print('Starting classifier demo...', flush=True)
    from classifier_predict import read_records, write_predictions
    try:
        if args.cpu and not args.live:
            raise ValueError('--cpu is used with --live')
        if not 1 <= args.cpu_threads <= 32:
            raise ValueError('--cpu-threads must be in [1,32]')
        if args._cpu_worker and not (args.live and args.cpu):
            raise ValueError('The internal CPU worker requires --live --cpu')
        records = read_records(args.input)
        if args.live:
            output = args.output or ROOT / 'outputs' / f'classifier_demo_{datetime.now():%Y%m%d_%H%M%S_%f}.json'
            if output.exists():
                raise FileExistsError('Output exists; choose a new output file')
            if args.cpu and not args._cpu_worker:
                return run_cpu_worker(args, output)
            print('Loading tokenizer/model for ' + str(len(records)) + ' input item(s)...', flush=True)
            predictions = live(records, args.model_dir, args.cpu)
            write_predictions(output, predictions)
            print('MODE: LIVE INFERENCE (' + ('CPU' if args.cpu else 'auto device') + ')')
            print('MODEL:', args.model_dir)
            print('OUTPUT:', output)
        else:
            predictions = read_saved(records, args.saved_results)
            print('MODE: SAVED RESULTS — no live model inference')
            print('SOURCE:', args.saved_results)
        for item, prediction in zip(records, predictions):
            print('\nTEXT:', item['text'])
            print('SECTION:', item.get('section', ''))
            print('PREDICTION:', prediction['label'], f"confidence={prediction['confidence']:.4f}")
            print('PROBABILITIES:', json.dumps(prediction['probabilities'], ensure_ascii=False))
        print('\nDemo examples are not an accuracy benchmark. Softmax confidence is not certified probability of correctness.')
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print('Demo failed:', exc, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('Demo cancelled.', file=sys.stderr, flush=True)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
