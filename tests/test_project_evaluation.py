"""Evaluation must preserve missing human work and separate review dimensions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from extraction import extract_document
from planning import generate_plan
from psai_project.evaluation import (build_evaluation_bundle, evaluate_bundle,
                                     reference_task_sha256, snapshot_run)


TEXT = ('# Proposal\nWe will design the equipment schema.\n'
        'We will implement the inventory API.\nWe will create a backup script.\n')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


class ProjectEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.make_run('first')

    def make_run(self, name, text=TEXT, dependency=True):
        folder = self.root / name
        folder.mkdir()
        record = extract_document(name, text, use_model=False)
        plan = generate_plan(record, source_text=text)
        if dependency and len(plan['steps']) > 1:
            plan['dependency_candidates'] = [{'before_task_id': 'step:1', 'after_task_id': 'step:2',
                'status': 'suggested', 'needs_review': True}]
        digest = hashlib.sha256(text.encode()).hexdigest()
        (folder / 'source.txt').write_text(text, encoding='utf-8', newline='')
        write_json(folder / 'analysis.json', {'document_id': name, 'extraction': record,
            'plan': plan, 'retrieval': {}, 'metadata': {'source_text_sha256': digest}})
        write_json(folder / 'run_manifest.json', {'source_text_sha256': digest})
        return folder

    def reviewed(self, run_dirs=None, kind='case_study', reviewer_kind='ai', independent=False):
        packet = build_evaluation_bundle(run_dirs or [self.run], dataset_kind=kind)
        packet['reviewers'] = [{'reviewer_id': 'R1', 'kind': reviewer_kind, 'independent': independent}]
        actor = {'reviewer_id': 'R1', 'rationale': 'Explicit synthetic test decision, not a real review.'}
        for doc in packet['documents']:
            tasks = doc['prediction_tasks']
            doc['gold_tasks'] = [{'task_id': 'G' + str(index), 'description': task['description'],
                'source_spans': deepcopy(task['source_spans']), 'supported': True, **actor}
                for index, task in enumerate(tasks[:2], 1)]
            for task in doc['gold_tasks']:
                task['task_sha256'] = reference_task_sha256(task)
            doc['matches'] = ([{'prediction_task_id': tasks[0]['task_id'],
                'prediction_task_sha256': tasks[0]['task_sha256'], 'gold_task_id': 'G1',
                'gold_task_sha256': doc['gold_tasks'][0]['task_sha256'],
                'equivalent': True, **actor}] if tasks else [])
            for key in ('reference_review', 'matching_review'):
                doc[key] = {'status': 'complete', **actor}
            for index, review in enumerate(doc['task_reviews']):
                review.update({'evidence_correct': index == 0, **actor})
            for review in doc['dependency_reviews']:
                review.update({'correct': False, **actor})
        return packet

    def test_packet_never_generates_reference_or_semantic_labels(self):
        packet = build_evaluation_bundle([self.run])
        doc = packet['documents'][0]
        self.assertEqual(doc['gold_tasks'], [])
        self.assertEqual(doc['matches'], [])
        self.assertEqual(packet['reviewers'], [])
        self.assertTrue(all(item['evidence_correct'] is None for item in doc['task_reviews']))
        result = evaluate_bundle(packet)
        self.assertEqual(result['status'], 'incomplete')
        self.assertIsNone(result['metrics'])
        self.assertIsNone(result['documents'][0]['metrics'])
        self.assertFalse(result['final_test_gate_passed'])

    def test_one_to_one_tp_fp_fn_and_separate_evidence_dependency_rates(self):
        packet = self.reviewed()
        before = deepcopy(packet)
        result = evaluate_bundle(packet)
        self.assertEqual(packet, before)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['review_scope'], 'case_study_ai_reviewed')
        task = result['metrics']['task_matching']['micro']
        self.assertEqual((task['tp'], task['fp'], task['fn']), (1, 2, 1))
        self.assertAlmostEqual(task['precision'], 1 / 3)
        self.assertAlmostEqual(task['recall'], 1 / 2)
        self.assertAlmostEqual(task['f1'], .4)
        self.assertAlmostEqual(result['metrics']['evidence_correctness']['micro']['rate'], 1 / 3)
        self.assertEqual(result['metrics']['dependency_correctness']['micro']['rate'], 0)
        self.assertFalse(result['independence_verified_by_software'])
        self.assertEqual(result['retrieval_relevance']['status'], 'not_evaluated')

    def test_duplicate_prediction_or_reference_match_cannot_inflate_tp(self):
        for change in ('same_gold', 'same_prediction', 'same_pair'):
            with self.subTest(change=change):
                packet = self.reviewed()
                doc = packet['documents'][0]
                extra = deepcopy(doc['matches'][0])
                if change == 'same_gold':
                    task = doc['prediction_tasks'][1]
                    extra.update(prediction_task_id=task['task_id'], prediction_task_sha256=task['task_sha256'])
                elif change == 'same_prediction':
                    extra['gold_task_id'] = 'G2'
                    extra['gold_task_sha256'] = doc['gold_tasks'][1]['task_sha256']
                doc['matches'].append(extra)
                with self.assertRaisesRegex(ValueError, 'one-to-one|Duplicate semantic'):
                    evaluate_bundle(packet)

    def test_rejected_equivalence_is_not_tp_despite_shared_words_or_source(self):
        packet = self.reviewed()
        packet['documents'][0]['matches'][0]['equivalent'] = False
        result = evaluate_bundle(packet)['metrics']['task_matching']['micro']
        self.assertEqual((result['tp'], result['fp'], result['fn']), (0, 3, 2))
        self.assertEqual(result['f1'], 0)

    def test_every_saved_binding_is_required_and_rechecked(self):
        for key in ('source_sha256', 'analysis_sha256', 'run_manifest_sha256', 'manifest_sha256'):
            for remove in (True, False):
                with self.subTest(key=key, remove=remove):
                    packet = self.reviewed()
                    if remove:
                        packet['documents'][0]['binding'].pop(key)
                    else:
                        packet['documents'][0]['binding'][key] = '0' * 64
                    with self.assertRaisesRegex(ValueError, 'binding'):
                        evaluate_bundle(packet)

    def test_equivalent_json_with_changed_bytes_invalidates_bound_review(self):
        packet = self.reviewed()
        path = self.run / 'analysis.json'
        path.write_bytes(path.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'binding'):
            evaluate_bundle(packet)

    def test_source_and_manifest_tampering_rejected(self):
        for name in ('source.txt', 'run_manifest.json'):
            with self.subTest(name=name):
                packet = self.reviewed()
                path = self.run / name
                original = path.read_bytes()
                if name == 'source.txt':
                    path.write_bytes(original + b'changed')
                else:
                    write_json(path, {'source_text_sha256': '0' * 64})
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    evaluate_bundle(packet)
                path.write_bytes(original)

    def test_manifest_fallback_keeps_absent_original_explicit(self):
        (self.run / 'run_manifest.json').rename(self.run / 'manifest.json')
        snapshot = snapshot_run(self.run)
        self.assertIsNone(snapshot['binding']['run_manifest_sha256'])
        self.assertEqual(snapshot['manifest_files'], ['manifest.json'])
        self.assertIsNotNone(snapshot['binding']['manifest_sha256'])

    def test_prediction_and_review_target_hashes_are_bound(self):
        for target in ('prediction', 'match', 'task_review', 'dependency_review'):
            with self.subTest(target=target):
                packet = self.reviewed()
                doc = packet['documents'][0]
                if target == 'prediction':
                    doc['prediction_tasks'][0]['description'] = 'Replacement task'
                elif target == 'match':
                    doc['matches'][0]['prediction_task_sha256'] = '0' * 64
                elif target == 'task_review':
                    doc['task_reviews'][0]['task_sha256'] = '0' * 64
                else:
                    doc['dependency_reviews'][0]['dependency_sha256'] = '0' * 64
                with self.assertRaisesRegex(ValueError, 'hash mismatch|binding'):
                    evaluate_bundle(packet)

    def test_missing_or_pending_reviews_suppress_numeric_aggregate(self):
        for field in ('task_reviews', 'dependency_reviews', 'reference_review', 'matching_review', 'matches', 'gold_tasks'):
            with self.subTest(field=field):
                packet = self.reviewed()
                doc = packet['documents'][0]
                if field in ('task_reviews', 'dependency_reviews'):
                    doc[field] = []
                elif field in ('reference_review', 'matching_review'):
                    doc[field]['status'] = 'pending'
                elif field == 'matches':
                    doc[field][0]['equivalent'] = None
                else:
                    doc[field][0]['supported'] = None
                result = evaluate_bundle(packet)
                self.assertEqual(result['status'], 'incomplete')
                self.assertIsNone(result['metrics'])
                self.assertIsNone(result['documents'][0]['metrics'])

    def test_unsupported_or_wrong_span_reference_is_rejected(self):
        for update in ('unsupported', 'span', 'hash', 'missing_spans'):
            with self.subTest(update=update):
                packet = self.reviewed()
                gold = packet['documents'][0]['gold_tasks'][0]
                if update == 'unsupported':
                    gold['supported'] = False
                elif update == 'missing_spans':
                    gold['source_spans'] = []
                elif update == 'span':
                    gold['source_spans'][0]['quote'] = 'Text that is absent'
                else:
                    gold['source_spans'][0]['source_sha256'] = '0' * 64
                with self.assertRaisesRegex(ValueError, 'Unsupported|span|hash'):
                    evaluate_bundle(packet)

    def test_empty_document_metrics_are_null_where_denominators_zero(self):
        empty = self.make_run('empty', '# Background\nAn overview only.\n', dependency=False)
        result = evaluate_bundle(self.reviewed([empty]))
        self.assertEqual(result['status'], 'complete')
        task = result['metrics']['task_matching']['micro']
        self.assertIsNone(task['precision'])
        self.assertIsNone(task['recall'])
        self.assertIsNone(task['f1'])
        self.assertEqual(set(task['undefined']), {'precision', 'recall', 'f1'})
        self.assertIsNone(result['metrics']['dependency_correctness']['micro']['rate'])
        self.assertEqual(result['metrics']['dependency_correctness']['micro']['status'], 'undefined_no_items')

    def test_micro_and_macro_exclude_undefined_per_document_explicitly(self):
        empty = self.make_run('empty', '# Background\nAn overview only.\n', dependency=False)
        result = evaluate_bundle(self.reviewed([self.run, empty]))
        self.assertEqual(result['status'], 'complete')
        macro = result['metrics']['task_matching']['macro']
        self.assertAlmostEqual(macro['f1'], .4)
        self.assertEqual(macro['document_count'], 2)
        self.assertEqual(macro['defined_document_counts']['f1'], 1)
        self.assertEqual(result['metrics']['task_matching']['micro']['fp'], 2)

    def test_human_or_final_scope_never_accepts_ai_review_as_human(self):
        for kind in ('human_review', 'independent_final_test'):
            packet = self.reviewed(kind=kind)
            result = evaluate_bundle(packet)
            self.assertEqual(result['status'], 'incomplete')
            self.assertIsNone(result['metrics'])
            self.assertIsNone(result['documents'][0]['metrics'])
            self.assertFalse(result['final_test_gate_passed'])

    def test_final_test_needs_explicit_strict_gate_and_independent_human_declarations(self):
        second = self.make_run('second', TEXT.replace('# Proposal', '# Another proposal'))
        packet = self.reviewed([self.run, second], kind='independent_final_test', reviewer_kind='human', independent=True)
        self.assertIsNone(evaluate_bundle(packet)['metrics'])
        packet['final_test_gate'] = {'sources_independent': True, 'selection_frozen_before_review': True,
            'no_test_used_for_selection': True, 'approved': True, 'reviewer_id': 'R1',
            'rationale': 'Synthetic unit-test declaration only.'}
        result = evaluate_bundle(packet)
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(result['final_test_gate_passed'])
        self.assertEqual(result['identity_verification'], 'self_declared_not_authenticated')
        for key in ('sources_independent', 'selection_frozen_before_review', 'no_test_used_for_selection', 'approved'):
            modified = deepcopy(packet)
            modified['final_test_gate'][key] = 1
            self.assertIsNone(evaluate_bundle(modified)['metrics'])
        packet['reviewers'][0]['independent'] = False
        self.assertIsNone(evaluate_bundle(packet)['metrics'])

    def test_known_exposure_or_single_source_cannot_be_approved_into_final_test(self):
        for metadata in ({}, {'independent_source': False}, {'independent': False},
                         {'source_exposure': 'known_development'},
                         {'source_exposure': {'status': 'known_development_document', 'independent': False}},
                         {'synthetic_fixture': True}):
            with self.subTest(metadata=metadata):
                path = self.run / 'analysis.json'
                original = path.read_bytes()
                analysis = json.loads(original)
                analysis['metadata'].update(metadata)
                write_json(path, analysis)
                packet = self.reviewed(kind='independent_final_test', reviewer_kind='human', independent=True)
                packet['final_test_gate'] = {'sources_independent': True, 'selection_frozen_before_review': True,
                    'no_test_used_for_selection': True, 'approved': True, 'reviewer_id': 'R1',
                    'rationale': 'Synthetic unit-test declaration, must not override exposure.'}
                result = evaluate_bundle(packet)
                self.assertEqual(result['status'], 'incomplete')
                self.assertIsNone(result['metrics'])
                self.assertFalse(result['final_test_gate_passed'])
                if metadata:
                    self.assertTrue(any('known_development_or_synthetic' in item for item in result['missing_reviews']))
                self.assertTrue(any('multiple_distinct_sources' in item for item in result['missing_reviews']))
                path.write_bytes(original)

    def test_legacy_exposure_rejects_final_gate_even_with_multiple_sources_and_approvals(self):
        second = self.make_run('second', TEXT.replace('# Proposal', '# Another proposal'))
        for metadata in ({'independent': False},
                         {'source_exposure': {'status': 'known_development_document', 'independent': False}},
                         {'source_exposure': {'status': 'known_development_document'}},
                         {'source_exposure': {'independent': False}}):
            with self.subTest(metadata=metadata):
                path = self.run / 'analysis.json'
                original = path.read_bytes()
                analysis = json.loads(original)
                analysis['metadata'].update(metadata)
                write_json(path, analysis)
                packet = self.reviewed([self.run, second], kind='independent_final_test',
                                       reviewer_kind='human', independent=True)
                packet['final_test_gate'] = {'sources_independent': True, 'selection_frozen_before_review': True,
                    'no_test_used_for_selection': True, 'approved': True, 'reviewer_id': 'R1',
                    'rationale': 'Synthetic declaration cannot override known development exposure.'}
                result = evaluate_bundle(packet)
                self.assertFalse(result['final_test_gate_passed'])
                self.assertIsNone(result['metrics'])
                self.assertTrue(any('known_development_or_synthetic' in item for item in result['missing_reviews']))
                self.assertFalse(any('multiple_distinct_sources' in item for item in result['missing_reviews']))
                path.write_bytes(original)

    def test_relative_run_path_requires_explicit_base(self):
        packet = self.reviewed()
        packet['documents'][0]['run_dir'] = 'first'
        with self.assertRaisesRegex(ValueError, 'base_dir'):
            evaluate_bundle(packet)
        self.assertEqual(evaluate_bundle(packet, base_dir=self.root)['status'], 'complete')

    def test_unknown_reviewers_and_duplicate_review_decisions_rejected(self):
        for change in ('reviewer', 'duplicate_task', 'duplicate_dependency', 'truthy_decision'):
            with self.subTest(change=change):
                packet = self.reviewed()
                doc = packet['documents'][0]
                if change == 'reviewer':
                    doc['matches'][0]['reviewer_id'] = 'unknown'
                elif change == 'duplicate_task':
                    doc['task_reviews'].append(deepcopy(doc['task_reviews'][0]))
                elif change == 'duplicate_dependency':
                    doc['dependency_reviews'].append(deepcopy(doc['dependency_reviews'][0]))
                else:
                    doc['task_reviews'][0]['evidence_correct'] = 'true'
                with self.assertRaises(ValueError):
                    evaluate_bundle(packet)

    def test_reference_fingerprint_does_not_depend_on_review_rationale_or_run_mode(self):
        packet = self.reviewed()
        result = evaluate_bundle(packet)
        packet['documents'][0]['gold_tasks'][0]['rationale'] = 'Another explicit review rationale.'
        self.assertEqual(result['documents'][0]['reference_sha256'],
                         evaluate_bundle(packet)['documents'][0]['reference_sha256'])

    def test_reference_edit_or_rebinding_does_not_migrate_an_existing_semantic_match(self):
        for rebind in (False, True):
            with self.subTest(rebind=rebind):
                packet = self.reviewed()
                gold = packet['documents'][0]['gold_tasks'][0]
                gold['description'] = 'A new reference interpretation of the same source.'
                if rebind:
                    gold['task_sha256'] = reference_task_sha256(gold)
                with self.assertRaisesRegex(ValueError, 'Reference task hash|reference hash'):
                    evaluate_bundle(packet)

    def test_saved_plan_wording_is_revalidated_even_after_rebinding_bytes(self):
        path = self.run / 'analysis.json'
        analysis = json.loads(path.read_text(encoding='utf-8'))
        analysis['plan']['steps'][0]['description'] = 'Invented unsupported implementation work'
        write_json(path, analysis)
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            snapshot_run(self.run)

    def test_rules_run_rejects_forged_or_unverified_retrieval_before_review(self):
        path = self.run / 'analysis.json'
        original = path.read_bytes()
        analysis = json.loads(original)
        source_hash = analysis['extraction']['source_sha256']
        quote = 'We will design the equipment schema.'
        start = TEXT.index(quote)
        hit = {'evidence_id': 'first:R1', 'snapshot_id': source_hash, 'source_sha256': source_hash,
               'quote': quote, 'char_start': start, 'char_end': start + len(quote), 'verified': True}
        for update in ({'quote': 'Forged quote'}, {'source_sha256': '0' * 64},
                       {'snapshot_id': 'missing-origin'}, {'verified': False}, {'verified': 1},
                       {'char_start': start + 1}):
            with self.subTest(update=update):
                current = deepcopy(analysis)
                current['retrieval'] = {'hits': [{**hit, **update}]}
                write_json(path, current)
                with self.assertRaisesRegex(ValueError, 'Retrieval hit|Source span/hash'):
                    snapshot_run(self.run)
        analysis['retrieval'] = {'hits': [hit]}
        write_json(path, analysis)
        self.assertEqual(snapshot_run(self.run)['document_id'], 'first')
        path.write_bytes(original)

    def test_retrieval_snapshot_identity_can_distinguish_same_content_origins(self):
        path = self.run / 'analysis.json'
        analysis = json.loads(path.read_bytes())
        digest = analysis['extraction']['source_sha256']
        quote = 'We will design the equipment schema.'
        start = TEXT.index(quote)
        # Two origins can share text bytes and still have distinct snapshot IDs.
        analysis['retrieval'] = {'sources': {
            'origin-a': {'artifact': 'source.txt', 'source_sha256': digest},
            'origin-b': {'artifact': 'source.txt', 'source_sha256': digest}},
            'hits': [{'snapshot_id': 'origin-b', 'source_sha256': digest, 'verified': True,
                      'quote': quote, 'char_start': start, 'char_end': start + len(quote)}]}
        write_json(path, analysis)
        self.assertEqual(snapshot_run(self.run)['document_id'], 'first')


if __name__ == '__main__':
    unittest.main()
