"""Download only auxiliary query-generation/teacher weights into drive D.

The existing BGE-M3 base and classifier environments are not modified.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    cache = ROOT / 'models/rag/_hf_cache'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(cache)
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(token=False)
    entries = []
    for repo in ['doc2query/msmarco-t5-small-v1', 'cross-encoder/ms-marco-MiniLM-L6-v2']:
        print('Inspecting auxiliary model:', repo, flush=True)
        info = api.model_info(repo)
        names = {item.rfilename for item in info.siblings}
        weights = ['model.safetensors'] if 'model.safetensors' in names else ['pytorch_model.bin']
        if not any(name in names for name in weights):
            raise ValueError('Expected a single-file checkpoint: ' + repo)
        patterns = weights + ['config.json', 'generation_config.json', 'tokenizer.json',
                              'tokenizer_config.json', 'special_tokens_map.json',
                              'spiece.model', 'sentencepiece.bpe.model', 'vocab.txt',
                              'added_tokens.json', 'README.md']
        destination = snapshot_download(repo_id=repo, revision=info.sha, cache_dir=str(cache / 'hub'),
                                        allow_patterns=patterns, token=False)
        entries.append(dict(repo_id=repo, revision=info.sha, local_path=str(Path(destination).resolve()),
                            role='query_generator' if repo.startswith('doc2query/') else 'pseudo_label_teacher'))
        print('Auxiliary model ready:', repo, info.sha, flush=True)
    target = ROOT / 'reports/rag_studies/retrieval_genq_lora_v1/auxiliary_models.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
                                     cache_on_project_drive=True, models=entries), indent=2), encoding='utf-8')
    print('Auxiliary model versions saved:', target, flush=True)


if __name__ == '__main__':
    main()
