"""Research audit CLI. Run --help for commands; no automatic long training."""
import argparse
import sys
from pathlib import Path

from study_data import OUT, registry, annotation_packet, agreement, write_json, validate_assigned_annotations
from study_evaluation import evaluate_models, benchmark_deployed
from study_experiments import create_plan, run_experiment, aggregate
from study_final_test import prepare_final_candidates, screen_final_candidates, freeze_selection, evaluate_final


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('audit', 'annotation-packet', 'final-candidates', 'final-screen', 'experiment-plan', 'aggregate', 'pipeline-prepare', 'pipeline-score', 'final-evaluate', 'benchmark'):
        commands.add_parser(name)
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--models', nargs='*')
    evaluate.add_argument('--dataset', choices=['human_holdout_diagnostic', 'historical_fold0_diagnostic'], default='human_holdout_diagnostic')
    evaluate.add_argument('--batch-size', type=int, default=8)
    raters = commands.add_parser('agreement')
    raters.add_argument('rater_a', type=Path)
    raters.add_argument('rater_b', type=Path)
    raters.add_argument('--output', type=Path, default=OUT / 'annotation/agreement.json')
    train = commands.add_parser('train')
    train.add_argument('experiment')
    train.add_argument('--smoke-steps', type=int)
    freeze = commands.add_parser('freeze-selection')
    freeze.add_argument('model_dir', type=Path)
    freeze.add_argument('validation_metrics', type=Path)
    args = parser.parse_args()
    if args.command == 'audit':
        registry()
    elif args.command == 'annotation-packet':
        annotation_packet()
    elif args.command == 'agreement':
        validate_assigned_annotations(args.rater_a, args.rater_b, OUT / 'annotation/coordinator_key.csv')
        agreement(args.rater_a, args.rater_b, args.output)
    elif args.command == 'evaluate':
        evaluate_models(args.models, args.dataset, args.batch_size)
    elif args.command == 'benchmark':
        benchmark_deployed()
    elif args.command == 'experiment-plan':
        create_plan()
    elif args.command == 'train':
        if args.smoke_steps is not None and args.smoke_steps < 1:
            parser.error('--smoke-steps must be positive')
        name = args.experiment if args.smoke_steps is None else f'{args.experiment}_smoke_{args.smoke_steps}'
        folder = OUT / 'experiments' / name
        existed = folder.exists()
        try:
            run_experiment(args.experiment, args.smoke_steps)
            write_json(folder / 'run_status.json', dict(state='complete', smoke=args.smoke_steps is not None))
        except Exception as exc:
            if not existed and folder.exists():
                write_json(folder / 'failure.json', dict(state='failed', error_type=type(exc).__name__, error=str(exc)))
            raise
    elif args.command == 'aggregate':
        aggregate()
    elif args.command == 'final-candidates':
        prepare_final_candidates()
    elif args.command == 'final-screen':
        screen_final_candidates()
    elif args.command == 'freeze-selection':
        freeze_selection(args.model_dir, args.validation_metrics)
    elif args.command == 'final-evaluate':
        evaluate_final()
    elif args.command == 'pipeline-prepare':
        from study_pipeline import prepare_pipeline
        prepare_pipeline()
    elif args.command == 'pipeline-score':
        from study_pipeline import score_pipeline
        score_pipeline()


if __name__ == '__main__':
    main()
