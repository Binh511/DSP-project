"""Download a pinned local planner into drive D; never send project text.

This setup command performs network I/O explicitly. Runtime inference is local.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', default='989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
                        help='Explicit Qwen snapshot revision; default is the verified project release')
    args = parser.parse_args()
    cache = ROOT / 'models/project_completion/hf_cache'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(cache)
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    from huggingface_hub import HfApi, snapshot_download
    repo = 'Qwen/Qwen2.5-1.5B-Instruct'
    info = HfApi(token=False).model_info(repo, revision=args.revision)
    print('Downloading local planner:', repo, 'revision:', info.sha, flush=True)
    path = Path(snapshot_download(repo, revision=info.sha, token=False, cache_dir=str(cache / 'hub'),
        allow_patterns=['*.safetensors', '*.safetensors.index.json', 'config.json', 'generation_config.json',
                        'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
                        'special_tokens_map.json', 'README.md', 'LICENSE']))
    files = {}
    for file in path.iterdir():
        if file.is_file():
            digest = hashlib.sha256()
            with file.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            files[file.name] = digest.hexdigest()
    output = ROOT / 'reports/project_completion/local_planner_model.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'repo_id': repo, 'revision': info.sha, 'local_path': str(path.resolve()),
        'created_utc': datetime.now(timezone.utc).isoformat(), 'files_sha256': files,
        'training_performed': False, 'inference_requires_network': False}, indent=2), encoding='utf-8')
    print('Pinned planner ready:', output, flush=True)


if __name__ == '__main__':
    main()
