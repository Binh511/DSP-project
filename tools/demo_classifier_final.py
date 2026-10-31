"""Live CPU demo of the explicitly frozen source-clean classifier release."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.finalize_classifier_release import verify_release


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'examples/classifier_teacher_live.json')
    args = parser.parse_args(argv)
    print('Verifying frozen Classifier v2...', flush=True)
    try:
        manifest = verify_release()
        from classifier_demo import main as run_demo
        print('CLASSIFIER RELEASE:', manifest['release_version'], flush=True)
        print('SELECTED RUN:', manifest['selected_run'], flush=True)
        return run_demo(['--live', '--cpu', '--input', str(args.input.resolve()),
                         '--model-dir', str(ROOT / manifest['model_path'])])
    except (OSError, ValueError, KeyError) as error:
        print('Classifier release verification failed:', error, file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
