"""Check probe metric meaning and rank/source integrity without model imports."""
import copy
import hashlib
import unittest

from tools.evaluate_retrieval_probe import (
    aggregate_scores, covered_characters, reconstruct_rankings, score_question, validate_questions,
)


class ExternalRetrievalProbeTests(unittest.TestCase):
    def question(self, spans=None):
        return {'question_id': 'q1', 'document_id': 'doc', 'query': 'question',
                'relevant_spans': spans or [{'char_start': 10, 'char_end': 30}]}

    def chunk(self, identifier, start, end, rank=None):
        row = {'chunk_id': identifier, 'char_start': start, 'char_end': end}
        if rank is not None:
            row['rank'] = rank
        return row

    def test_overlap_is_union_and_never_counts_overlaps_twice(self):
        span = {'char_start': 10, 'char_end': 30}
        chunks = [self.chunk('a', 0, 18), self.chunk('b', 15, 25), self.chunk('c', 24, 80)]
        self.assertEqual(covered_characters(span, chunks), 20)
        self.assertEqual(covered_characters(span, chunks[:2]), 15)
        self.assertEqual(covered_characters(span, [self.chunk('d', 30, 50)]), 0)

    def test_cross_boundary_reference_is_unattainable_without_hiding_its_miss(self):
        chunks = [self.chunk('a', 0, 20), self.chunk('b', 20, 40)]
        ranking = [{**row, 'rank': i} for i, row in enumerate(chunks, 1)]
        result = score_question(self.question(), ranking, chunks)
        self.assertTrue(result['known_references'][0]['crosses_chunk_boundary'])
        self.assertFalse(result['known_references'][0]['attainable_as_one_chunk'])
        self.assertEqual(result['metrics']['complete_known_span_coverage_at_5'], 0)
        self.assertEqual(result['metrics']['known_reference_character_coverage_at_5'], 1)
        self.assertEqual(result['metrics']['known_reference_reciprocal_rank'], 0)

    def test_mrr_uses_first_complete_reference_and_does_not_require_every_reference(self):
        spans = [{'char_start': 10, 'char_end': 20}, {'char_start': 40, 'char_end': 50}]
        chunks = [self.chunk('a', 80, 90), self.chunk('b', 8, 25), self.chunk('c', 39, 55)]
        ranking = [{**row, 'rank': i} for i, row in enumerate(chunks, 1)]
        result = score_question(self.question(spans), ranking, chunks)
        self.assertEqual(result['metrics']['known_reference_reciprocal_rank'], .5)
        self.assertEqual(result['metrics']['any_complete_known_span_at_1'], 0)
        self.assertEqual(result['metrics']['complete_known_span_coverage_at_5'], 1)

    def test_real_query_ranks_survive_different_combined_hit_order(self):
        queries = ['question 1', 'question 2']
        result = {'queries': queries, 'hits': [
            {'chunk_id': 'b', 'query_matches': [{'query_index': 0, 'query': queries[0], 'rank': 2, 'score': 3},
                                                {'query_index': 1, 'query': queries[1], 'rank': 1, 'score': 4}]},
            {'chunk_id': 'a', 'query_matches': [{'query_index': 0, 'query': queries[0], 'rank': 1, 'score': 5}]},
        ]}
        ranks = reconstruct_rankings(result, queries, [2, 1])
        self.assertEqual([row['chunk_id'] for row in ranks[0]], ['a', 'b'])
        self.assertEqual(ranks[1][0]['score'], 4)
        with self.assertRaises(ValueError):
            reconstruct_rankings(result, queries, [2, 2])
        bad = copy.deepcopy(result)
        bad['hits'][0]['query_matches'][0]['rank'] = 3
        with self.assertRaises(ValueError):
            reconstruct_rankings(bad, queries, [2, 1])

    def test_source_quote_hash_and_blind_ai_provenance_are_required(self):
        text = 'Known source text'
        source_sha = hashlib.sha256(text.encode()).hexdigest()
        question = self.question([{'char_start': 0, 'char_end': 5, 'quote': 'Known', 'source_sha256': source_sha}])
        question.update(source_sha256=source_sha, reviewer={'kind': 'ai'})
        payload = {'human_ground_truth': False, 'viewed_predictions': False, 'questions': [question]}
        self.assertEqual(validate_questions(payload, {'doc': text}), [question])
        tampered = copy.deepcopy(payload)
        tampered['questions'][0]['relevant_spans'][0]['quote'] = 'False'
        with self.assertRaises(ValueError):
            validate_questions(tampered, {'doc': text})
        payload['viewed_predictions'] = True
        with self.assertRaises(ValueError):
            validate_questions(payload, {'doc': text})

    def test_summary_distinguishes_mean_per_question_and_micro_span_coverage(self):
        first = score_question(self.question(), [self.chunk('a', 0, 40, 1)], [self.chunk('a', 0, 40)])
        second = score_question(self.question([{'char_start': 60, 'char_end': 70},
                                               {'char_start': 80, 'char_end': 90}]), [], [])
        result = aggregate_scores([first, second])
        self.assertEqual(result['mean_metrics']['complete_known_span_coverage_at_1'], .5)
        self.assertAlmostEqual(result['mean_metrics']['micro_complete_known_span_coverage_at_1'], 1 / 3)


if __name__ == '__main__':
    unittest.main()
