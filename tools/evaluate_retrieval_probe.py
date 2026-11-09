"""Probe actual BM25/dense/hybrid runtime rankings against frozen AI passages.

Known source passages are partial AI relevance references, never exhaustive
human qrels. This diagnostic does not measure task quality or a final test.
All modes use identical native BGE chunks, base model and query vectors.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODES = ('bm25', 'dense', 'hybrid')
TOP_KS = (1, 5)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def validate_questions(payload, texts):
    questions = payload.get('questions', [])
    if not questions or len({row['question_id'] for row in questions}) != len(questions):
        raise ValueError('Provide nonempty questions with unique IDs')
    if payload.get('human_ground_truth') is not False or payload.get('viewed_predictions') is not False:
        raise ValueError('This probe requires explicitly AI-authored, source-only blind questions')
    for question in questions:
        text = texts[question['document_id']]
        source_sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if question['source_sha256'] != source_sha or not question['relevant_spans']:
            raise ValueError('Question source hash or relevant passages are missing')
        if question.get('reviewer', {}).get('kind') != 'ai':
            raise ValueError('Reviewer must be recorded as AI')
        query = question['query']
        if not isinstance(query, str) or not query or query.strip() != query:
            raise ValueError('Queries must be nonempty and already whitespace-trimmed')
        for span in question['relevant_spans']:
            start, end = span['char_start'], span['char_end']
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                    or text[start:end] != span['quote'] or span['source_sha256'] != source_sha):
                raise ValueError('Known passage quote, source hash or character span differs')
    return questions


def covered_characters(span, chunks):
    """Count the union of returned character intersections without double count."""
    intervals = sorted((max(span['char_start'], row['char_start']), min(span['char_end'], row['char_end']))
                       for row in chunks
                       if row['char_end'] > span['char_start'] and row['char_start'] < span['char_end'])
    total, covered_until = 0, span['char_start']
    for left, right in intervals:
        total += max(0, right - max(left, covered_until))
        covered_until = max(covered_until, right)
    return total


def contains_span(chunk, span):
    return chunk['char_start'] <= span['char_start'] and chunk['char_end'] >= span['char_end']


def score_question(question, ranking, chunks):
    spans = question['relevant_spans']
    lengths = [span['char_end'] - span['char_start'] for span in spans]
    if [row['rank'] for row in ranking] != list(range(1, len(ranking) + 1)):
        raise ValueError('Question ranking must preserve every contiguous actual runtime rank')
    reference_info = []
    for index, span in enumerate(spans):
        containing = [row['chunk_id'] for row in chunks if contains_span(row, span)]
        ranks = [row['rank'] for row in ranking if contains_span(row, span)]
        all_covered = covered_characters(span, chunks)
        reference_info.append({
            'reference_index': index, 'char_start': span['char_start'], 'char_end': span['char_end'],
            'reference_characters': lengths[index], 'containing_chunk_ids': containing,
            'attainable_as_one_chunk': bool(containing),
            'crosses_chunk_boundary': not containing and all_covered == lengths[index],
            'indexed_reference_character_coverage': all_covered / lengths[index],
            'first_complete_span_rank': min(ranks) if ranks else None,
        })
    first = min((row['first_complete_span_rank'] for row in reference_info
                 if row['first_complete_span_rank'] is not None), default=None)
    metrics = {'known_reference_reciprocal_rank': 1 / first if first else 0.0,
               'known_reference_reciprocal_rank_at_5': 1 / first if first and first <= 5 else 0.0}
    cutoffs = {}
    for k in TOP_KS:
        selected = ranking[:k]
        covered = [covered_characters(span, selected) for span in spans]
        complete = [any(contains_span(row, span) for row in selected) for span in spans]
        cutoffs[str(k)] = {'complete_known_spans': sum(complete), 'known_spans': len(spans),
                           'covered_reference_characters': sum(covered),
                           'reference_characters': sum(lengths)}
        metrics.update({f'complete_known_span_coverage_at_{k}': sum(complete) / len(spans),
                        f'known_reference_character_coverage_at_{k}': sum(covered) / sum(lengths),
                        f'any_complete_known_span_at_{k}': int(any(complete))})
    return {'question_id': question['question_id'], 'document_id': question['document_id'],
            'query': question['query'], 'ranking_count': len(ranking), 'metrics': metrics,
            'cutoff_counts': cutoffs, 'known_references': reference_info,
            'ranking': ranking}


def reconstruct_rankings(result, queries, expected_counts):
    """Recover actual per-query ranks, checking that combined fusion lost none."""
    if result.get('queries') != queries or len(expected_counts) != len(queries):
        raise ValueError('Runtime query ordering differs from prepared queries')
    rankings = [[] for _ in queries]
    for hit in result['hits']:
        for match in hit['query_matches']:
            index = match['query_index']
            if type(index) is not int or not 0 <= index < len(queries) or match['query'] != queries[index]:
                raise ValueError('A returned hit contains an invalid query binding')
            rankings[index].append({**{key: value for key, value in hit.items() if key != 'query_matches'},
                                    'score': match['score'], 'rank': match['rank'], 'query_match': match})
    for index, rows in enumerate(rankings):
        rows.sort(key=lambda row: row['rank'])
        if (len(rows) != expected_counts[index]
                or len({row['chunk_id'] for row in rows}) != len(rows)
                or [row['rank'] for row in rows] != list(range(1, expected_counts[index] + 1))):
            raise ValueError('Combined runtime output truncated, duplicated or omitted per-query ranking entries')
    return rankings


def aggregate_scores(scores):
    if not scores:
        raise ValueError('Cannot summarize zero questions')
    metrics = {key: sum(row['metrics'][key] for row in scores) / len(scores)
               for key in scores[0]['metrics']}
    counts = {}
    for k in TOP_KS:
        count = {key: sum(row['cutoff_counts'][str(k)][key] for row in scores)
                 for key in scores[0]['cutoff_counts'][str(k)]}
        counts[str(k)] = count
        metrics[f'micro_complete_known_span_coverage_at_{k}'] = count['complete_known_spans'] / count['known_spans']
        metrics[f'micro_known_reference_character_coverage_at_{k}'] = count['covered_reference_characters'] / count['reference_characters']
    return {'questions': len(scores), 'mean_metrics': metrics, 'cutoff_counts': counts}


def run(questions_path, source_manifest_path, output, *, device='cuda'):
    from psai_project.execution import offline_environment
    import os
    os.environ.update(offline_environment())
    from psai_retrieval import runtime

    questions_path, source_manifest_path, output = (Path(path).resolve()
        for path in (questions_path, source_manifest_path, output))
    if output.exists():
        raise FileExistsError('Choose a fresh probe directory; preserve prior evidence')
    payload = json.loads(questions_path.read_text('utf-8'))
    manifest = json.loads(source_manifest_path.read_text('utf-8'))
    source_paths = {Path(row['file']).stem: source_manifest_path.parent / row['file'] for row in manifest['sources']}
    texts = {key: path.read_text('utf-8-sig') for key, path in source_paths.items()}
    questions = validate_questions(payload, texts)
    for row in manifest['sources']:
        if sha(source_manifest_path.parent / row['file']) != row['sha256']:
            raise ValueError('Source file no longer matches its original manifest')
    by_document = {}
    for question in questions:
        by_document.setdefault(question['document_id'], []).append(question)
    for rows in by_document.values():
        if runtime._queries([row['query'] for row in rows]) != [row['query'] for row in rows]:
            raise ValueError('Runtime query normalization would merge or alter the frozen questions')
    encoder = runtime._encoder_provenance(runtime.DEFAULT_BASE, None)
    code_paths = ['tools/evaluate_retrieval_probe.py', 'psai_retrieval/runtime.py',
                  'psai_rag/evaluate.py', 'psai_rag/data.py']
    protocol = {
        'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'evaluation_kind': 'external_source_AI_known_passage_retrieval_diagnostic',
        'human_ground_truth': False, 'independent_final_test': False,
        'questions_authored_without_viewing_predictions': payload['viewed_predictions'] is False,
        'questions_path': str(questions_path), 'questions_sha256': sha(questions_path),
        'source_manifest_path': str(source_manifest_path), 'source_manifest_sha256': sha(source_manifest_path),
        'source_files_sha256': {str(path): sha(path) for key, path in source_paths.items() if key in by_document},
        'code_files_sha256': {name: sha(ROOT / name) for name in code_paths},
        'encoder': encoder, 'device': device, 'adapter': None, 'reranker': None,
        'modes': list(MODES), 'metric_cutoffs': list(TOP_KS),
        'chunking': 'same native BGE token chunks shared by all three runtime modes',
        'query_encoding': 'once together with chunks per document; reuse exact prepared vectors for dense/hybrid',
        'ranking': 'retain all eligible candidates; reconstruct individual query rankings from actual query_matches.rank',
        'minimum_dense_score': 0.0,
        'relevance': 'partial preauthored AI reference passages; a complete match means one hit wholly contains a known span',
        'metrics': 'whole-known-span coverage@1/5, union character overlap@1/5, reciprocal rank of first complete known span',
        'unattainable_policy': 'keep boundary-spanning reference in denominator and explicitly expose unattainability in a single chunk',
        'missing_qrels_policy': 'unlabelled passages are unknown relevance, never used to compute Precision or nDCG',
        'no_training': True, 'no_model_downloads': True,
    }
    output.mkdir(parents=True)
    write_new(output / 'protocol.json', protocol)
    protocol_sha = sha(output / 'protocol.json')
    print('Protocol frozen before inference: ' + protocol_sha, flush=True)
    started = time.monotonic()
    documents, scores_by_mode = [], {mode: [] for mode in MODES}
    for document_id, rows in by_document.items():
        queries = [row['query'] for row in rows]
        document_output = output / document_id
        document_output.mkdir()
        print('Indexing real local BGE-M3:', document_id, flush=True)
        sources = runtime._read_sources(source_paths[document_id], 'document')
        before_build = time.monotonic()
        index = runtime._create(sources, mode='hybrid', base_model_dir=runtime.DEFAULT_BASE,
                                device=device, prepared_queries=queries)
        build_seconds = time.monotonic() - before_build
        if index._prepared is None or index._prepared[0] != queries or index.manifest['encoder'] != encoder:
            raise ValueError('Index did not prepare the requested queries using the frozen base encoder')
        if index.manifest['implementation'] != runtime._implementation():
            raise ValueError('Runtime implementation changed during build')
        runtime._validate_chunks(index.chunks, sources)
        write_new(document_output / 'index_manifest.json', index.manifest)
        write_new(document_output / 'chunks.json', index.chunks)
        dense_scores = index._prepared[1] @ index.vectors.T
        lexical_scores = [index.lexical.scores(query) for query in queries]
        retrieval_files, doc_scores, search_seconds = {}, {}, {}
        for mode in MODES:
            expected_counts = []
            for qi in range(len(queries)):
                lexical_eligible = {i for i, score in enumerate(lexical_scores[qi]) if score > 0}
                dense_eligible = {i for i, score in enumerate(dense_scores[qi]) if float(score) >= 0.0}
                expected_counts.append(len(lexical_eligible if mode == 'bm25' else dense_eligible
                                           if mode == 'dense' else lexical_eligible | dense_eligible))
            before_search = time.monotonic()
            result = index.search(queries, top_k=max(5, len(index.chunks)), mode=mode,
                                  base_model_dir=runtime.DEFAULT_BASE, device=device)
            search_seconds[mode] = time.monotonic() - before_search
            if result['scope']['candidate_count'] != len(index.chunks):
                raise ValueError('Runtime search scope changed between modes')
            rankings = reconstruct_rankings(result, queries, expected_counts)
            runtime._validate_chunks(result['hits'], sources)
            name = 'retrieval_' + mode + '.json'
            write_new(document_output / name, result)
            retrieval_files[mode] = {'path': document_id + '/' + name, 'sha256': sha(document_output / name)}
            doc_scores[mode] = [score_question(question, ranking, index.chunks)
                                for question, ranking in zip(rows, rankings)]
            scores_by_mode[mode].extend(doc_scores[mode])
        write_new(document_output / 'scores.json', doc_scores)
        documents.append({'document_id': document_id, 'source_sha256': sources[0]['source_sha256'],
                          'questions': len(rows), 'chunks': len(index.chunks),
                          'build_including_model_load_hashes_encoding_seconds': build_seconds,
                          'search_with_prepared_vectors_seconds': search_seconds,
                          'encoding_runtime': index.manifest['encoding_runtime'],
                          'retrieval_artifacts': retrieval_files,
                          'full_per_query_rankings_verified': True,
                          'quote_and_source_grounding_verified': True})
        print('Completed:', document_id, 'chunks=', len(index.chunks), flush=True)
    integrity = {'questions_unchanged': sha(questions_path) == protocol['questions_sha256'],
                 'source_manifest_unchanged': sha(source_manifest_path) == protocol['source_manifest_sha256'],
                 'protocol_unchanged': sha(output / 'protocol.json') == protocol_sha,
                 'sources_unchanged': all(sha(path) == expected for path, expected in protocol['source_files_sha256'].items()),
                 'code_unchanged': all(sha(ROOT / path) == expected for path, expected in protocol['code_files_sha256'].items())}
    model_files = encoder['base']['files_sha256']
    integrity['model_unchanged'] = all(sha(Path(encoder['base']['path']) / name) == expected for name, expected in model_files.items())
    if not all(integrity.values()):
        raise ValueError('Frozen input, code or model changed during diagnostic: ' + repr(integrity))
    result = {
        'status': 'diagnostic_completed', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': protocol_sha, 'human_ground_truth': False, 'independent_final_test': False,
        'partial_AI_relevance_references': True, 'question_count': len(questions), 'documents': documents,
        'summary_by_mode': {mode: aggregate_scores(scores) for mode, scores in scores_by_mode.items()},
        'integrity': integrity, 'elapsed_wall_seconds': time.monotonic() - started,
        'limits': [
            'Six AI-authored source-only questions on three public technical specifications are not a representative human benchmark.',
            'Known passages are partial relevance references, not exhaustive qrels; scores establish coverage of only those passages.',
            'Whole-span scoring includes chunk-boundary misses, disclosed separately without changing labels or chunking.',
            'Character overlap can combine partial spans across hits and does not prove semantic relevance or answer completeness.',
            'BM25 uses the same native-BGE chunks for this controlled comparison; normal Fast uses character windows.',
            'Retrieval quality is distinct from task extraction, final plan quality, and the role of document RAG in the planner.',
            'No LoRA/reranker comparison, parameter tuning, training, dependency installation or model download was performed.',
            'Public source exposure during model pretraining is unknown.',
            'Prepared-vector search times exclude encoder setup and query encoding, which are counted in index build time.',
        ],
    }
    write_new(output / 'diagnostic.json', result)
    return {'status': result['status'], 'output': str(output), 'summary_by_mode': result['summary_by_mode'],
            'elapsed_wall_seconds': result['elapsed_wall_seconds'], 'integrity': integrity}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args(argv)
    result = run(args.questions, args.source_manifest, args.output, device=args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
