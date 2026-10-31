"""Freeze/verify the source-clean classifier release using saved validation only.

No training, inference, Torch imports, final-test scoring or production changes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
STUDY_REL = Path('reports/research_review/studies/classifier_source_clean_context_study_v1')
RELEASE_REL = Path('reports/classifier/release_source_clean_v2')
LABELS = ['OBJECTIVE', 'REQUIREMENT', 'ACTIVITY', 'EXPECTED_RESULT', 'OTHER']
SEEDS = (123, 42, 2026)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def safe_file(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError('Artifact path must stay inside the project: ' + str(relative))
    if not path.is_file():
        raise FileNotFoundError('Missing release artifact: ' + str(path))
    return path


def check_bindings(root, bindings):
    for relative, digest in bindings.items():
        if sha(safe_file(root, relative)) != digest:
            raise ValueError('Frozen artifact changed: ' + relative)


def choose_run(results):
    expected = {(variant, seed) for variant in ('baseline', 'context') for seed in SEEDS}
    identities = [(r['variant'], int(r['seed'])) for r in results]
    if len(identities) != 6 or set(identities) != expected:
        raise ValueError('Selection requires exactly three completed seeds per configuration')
    if any(not math.isfinite(r['macro_f1']) or not 0 <= r['macro_f1'] <= 1 for r in results):
        raise ValueError('Invalid validation Macro-F1')
    groups = []
    for variant in ('baseline', 'context'):
        rows = [r for r in results if r['variant'] == variant]
        groups.append(dict(variant=variant, n=3,
                           mean=statistics.mean(r['macro_f1'] for r in rows),
                           sample_std=statistics.stdev(r['macro_f1'] for r in rows)))
    best_group = max(groups, key=lambda g: g['mean'])
    chosen = max((r for r in results if r['variant'] == best_group['variant']),
                 key=lambda r: r['macro_f1'])
    return chosen, groups


def verify_release(root=ROOT):
    folder = root / RELEASE_REL
    manifest = read(folder / 'FINAL_MODEL_MANIFEST.json')
    selection_path = folder / 'final_selection.json'
    if sha(selection_path) != manifest['final_selection_sha256']:
        raise ValueError('Frozen model selection changed')
    selection = read(selection_path)
    if (selection['model_dir'] != manifest['model_path'] or
            selection['model_sha256'] != manifest['model_sha256']):
        raise ValueError('Model identities disagree')
    if manifest.get('production_ensemble_replaced') or manifest.get('independent_final_evaluation_complete'):
        raise ValueError('Unsupported deployment or independent-test claim')
    check_bindings(root, manifest['artifact_sha256'])
    return manifest


def freeze_release(root=ROOT):
    folder = root / RELEASE_REL
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / 'FINAL_MODEL_MANIFEST.json'
    if manifest_path.exists():
        return verify_release(root)
    if (folder / 'final_selection.json').exists():
        raise FileExistsError('Selection already exists without its manifest; inspect before retrying')
    study = root / STUDY_REL
    plan = read(study / 'experiment_plan.json')
    queue = read(study / 'queue_status.json')
    if queue['state'] != 'complete' or len(queue['completed']) != 6:
        raise ValueError('The declared six-run study must be complete')
    audit_path = folder / 'audit_results.json'
    audit = read(audit_path)
    if audit.get('status') != 'PASS' or audit.get('failed_checks'):
        raise ValueError('Independent artifact audit must pass')
    bindings = {}

    def bind(path, expected=None):
        relative = path.relative_to(root).as_posix()
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('Artifact does not match recorded validation: ' + relative)
        bindings[relative] = digest

    for filename, digest in plan['code_sha256'].items():
        bind(root / filename, digest)
    rows = []
    for experiment in plan['experiments']:
        name = experiment['name']
        run = study / 'experiments' / name
        result = read(run / 'validation_metrics.json')
        if (read(run / 'run_status.json').get('state') != 'complete' or
                result.get('evaluation_kind') != 'validation_selection' or result['n'] != 1317 or
                result['config'] != experiment['config']):
            raise ValueError('Incomplete or mismatched validation run: ' + name)
        rows.append(dict(run=name, variant=experiment['variant'], seed=result['config']['seed'],
                         macro_f1=result['macro_f1'], accuracy=result['accuracy']))
        for filename in ['run_status.json', 'validation_metrics.json', 'validation_predictions.csv',
                         'validation_confusion.csv', 'provenance.json']:
            bind(run / filename)
    chosen, groups = choose_run(rows)
    recommendation = read(study / 'selection_recommendation.json')
    if recommendation['recommended_run'] != chosen['run']:
        raise ValueError('Selection differs from the declared study recommendation')
    model_rel = Path('models/controlled_study') / study.name / chosen['run']
    model = root / model_rel
    result_path = study / 'experiments' / chosen['run'] / 'validation_metrics.json'
    result = read(result_path)
    provenance = read(model / 'training_provenance.json')
    config = read(model / 'inference_config.json')
    if (provenance.get('max_steps') is not None or (model / 'invalid_checkpoint.json').exists() or
            provenance['train_count'] != 3942 or provenance['validation_count'] != 1317 or
            provenance['config'] != result['config'] or config.get('labels') != LABELS or
            config.get('ctx_version') != 3 or config.get('other_bias') != 0 or
            config.get('input_protocol') != 'study-v1'):
        raise ValueError('Selected export is not a complete canonical study checkpoint')
    for key in ['train_sha256', 'validation_sha256']:
        if result[key] != provenance[key]:
            raise ValueError('Data binding mismatch: ' + key)
    if any(config.get(key) != value for key, value in result['config'].items()):
        raise ValueError('Inference configuration differs from evaluated configuration')
    saved_manifest = result['validation_inference_manifest']
    for filename, digest in saved_manifest['model_files'].items():
        bind(model / filename, digest)
    for filename, digest in saved_manifest['implementation'].items():
        bind(root / filename, digest)
    if result['model_weight_sha256'] != saved_manifest['model_files']['model.safetensors']:
        raise ValueError('Weight binding mismatch')
    for filename in ['classifier_predict.py', 'classifier_demo.py',
                     'tools/demo_classifier_final.py', 'tools/finalize_classifier_release.py']:
        bind(root / filename)
    prepared = root / 'reports/research_review/studies/classifier_source_clean_v1'
    prep = read(prepared / 'preparation_manifest.json')
    for filename, digest in prep['evidence_files'].items():
        bind(prepared / filename, digest)
    for path in [audit_path, study / 'experiment_plan.json', study / 'summary.json',
                 study / 'selection_recommendation.json', prepared / 'preparation_manifest.json']:
        bind(path)
    created = datetime.now(timezone.utc).isoformat()
    selection = dict(release_version='2.0-source-clean', frozen_utc=created,
        model_dir=model_rel.as_posix(), model_sha256=result['model_weight_sha256'],
        validation_metrics_path=result_path.relative_to(root).as_posix(),
        validation_metrics_sha256=sha(result_path), inference_manifest=saved_manifest,
        train_sha256=result['train_sha256'], validation_sha256=result['validation_sha256'],
        criterion='configuration mean validation Macro-F1 over three seeds, then best exported checkpoint within that configuration',
        checkpoint_selection='Trainer BF16 epoch validation; exported runs compared using saved FP32 predictions',
        selected_run=chosen, configuration_statistics=groups,
        final_test_used=False, production_replaced=False)
    with (folder / 'final_selection.json').open('x', encoding='utf-8') as stream:
        json.dump(selection, stream, indent=2)
    manifest = dict(release_version='2.0-source-clean', created_utc=created,
        status='VALIDATION_SELECTED_CLASSIFIER_RELEASE_V2', model_path=model_rel.as_posix(),
        selected_run=chosen['run'], seed=chosen['seed'], variant=chosen['variant'],
        model_sha256=result['model_weight_sha256'], model_files=saved_manifest['model_files'],
        validation_macro_f1=result['macro_f1'], validation_accuracy=result['accuracy'],
        configuration_statistics=groups, labels=LABELS, input_protocol=config,
        dataset=dict(train_rows=3942, validation_rows=1317, train_documents=147,
                     validation_documents=29, train_sha256=result['train_sha256'],
                     validation_sha256=result['validation_sha256'],
                     full_source_raw_and_normalized_overlap=0,
                     preparation_manifest_sha256=sha(prepared / 'preparation_manifest.json')),
        inference_policy=dict(dtype='float32', prediction='argmax of five softmax probabilities',
                              other_bias=0., confidence_filter=None, confidence_calibrated=False),
        evaluation_runtime=saved_manifest['runtime'], artifact_sha256=bindings,
        final_selection_sha256=sha(folder / 'final_selection.json'),
        independent_final_evaluation_complete=False, human_label_review_complete=False,
        production_ensemble_replaced=False, previous_release_preserved=True,
        limitations=['Validation was used during development; these are not independent-test scores.',
                     'Reference labels are original and not fully independently adjudicated.',
                     'Full-source hash separation does not certify every near duplicate or unknown development exposure.',
                     'No supported superiority claim over production ensemble final_s11 + final_s22 + final_s33.'])
    with manifest_path.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
    return verify_release(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    manifest = verify_release() if args.verify_only else freeze_release()
    print(json.dumps(dict(status=manifest['status'], selected_run=manifest['selected_run'],
                          macro_f1=manifest['validation_macro_f1'], verified=True)))


if __name__ == '__main__':
    main()
