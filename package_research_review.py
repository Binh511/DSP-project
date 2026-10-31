"""Package research evidence and code without model weights/checkpoints."""
from pathlib import Path
import hashlib
import json
import zipfile

from study_data import ROOT, OUT


def main():
    core = ['research_review.py', 'study_data.py', 'study_inputs.py', 'study_evaluation.py',
            'study_experiments.py', 'study_final_test.py', 'study_pipeline.py',
            'classifier_training.py', 'test_classifier_foundations.py', 'test_research_review.py',
            'build_research_review_report.py', 'build_classifier_report.py',
            'train_human_cv.py', 'fair_fold0.py', 'extraction.py', 'planning.py',
            'ensemble_classifier.py', 'sentence_classifier.py',
            'data_segment_cls/segment_classification.csv', 'extra_human.csv',
            'runs/fair_fold0/frozen_split.json', 'runs/fair_fold0/test_predictions.csv',
            'runs/fair_fold0/human_holdout_predictions.csv',
            'runs/human_experiment/cv_fold0_targeted.csv', 'runs/human_experiment/cv_summary_targeted.json']
    files = [ROOT / p for p in core]
    for p in OUT.rglob('*'):
        if p.is_file() and 'checkpoints' not in p.parts and p.suffix.lower() in ('.py', '.json', '.csv', '.md', '.txt', '.log', '.docx'):
            files.append(p)
    files += list((ROOT / 'runs/independent_test').glob('fresh_test_pack_*/*.md'))
    for doc in ('PSAI002', 'PSAI008', 'PSAI011'):
        files += list((ROOT / 'data_cleaned' / doc).glob('*.md'))
    files = sorted(set(files))
    archive = ROOT / 'reports' / 'PSAI_KHAC_PHUC_BAO_CAO.zip'
    manifest = {}
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files:
            relative = p.relative_to(ROOT).as_posix()
            data = p.read_bytes()
            manifest[relative] = hashlib.sha256(data).hexdigest()
            z.writestr(relative, data)
        z.writestr('package_sha256.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert not any(n.endswith(('.safetensors', '.pt', '.bin')) for n in z.namelist())
        for p, digest in manifest.items():
            assert hashlib.sha256(z.read(p)).hexdigest() == digest
    # Separate rater packets do not contain coordinator keys or the other rater.
    for letter in ('A', 'B'):
        destination = ROOT / 'reports' / f'KIEM_DINH_NHAN_{letter}.zip'
        with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(OUT / 'ANNOTATION_GUIDELINE.md', 'HUONG_DAN_GAN_NHAN.md')
            z.write(OUT / f'annotation/rater_{letter}.csv', f'rater_{letter}.csv')
    print(f'{archive}\nFiles: {len(manifest)}; Size: {archive.stat().st_size / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
