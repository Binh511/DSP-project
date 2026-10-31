"""CPU pipeline integrity regressions; no checkpoints or dataset evaluations."""
from __future__ import annotations

from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import extraction
import planning
import run_extraction


class SourceContextTests(unittest.TestCase):
    def test_skipped_heading_levels_replace_peers_and_keep_offsets(self):
        text = ("# Root\r\n### Alpha\r\nWe will build project alpha.\r\n"
                "### Beta ###\r\nWe will build project beta.\r\n"
                "## Final\r\nWe will build project final.")
        items = list(extraction.sentences_with_context(text))
        self.assertEqual([x['section'] for x in items], ['Root > Alpha', 'Root > Beta', 'Root > Final'])
        for item in items:
            self.assertEqual(text[item['start']:item['end']], item['text'])

    def test_fences_require_matching_marker_and_length(self):
        text = ("# Work\n~~~~python\nWe will deploy fake content.\n```\n"
                "We will deploy more fake content.\n~~~\n~~~~\n"
                "We will implement the real dashboard.")
        items = list(extraction.sentences_with_context(text))
        self.assertEqual([x['text'] for x in items], ['We will implement the real dashboard.'])

    def test_classifier_cannot_silently_drop_or_corrupt_sentence_outputs(self):
        text = 'We will implement the project dashboard.'
        for predictions in ([], [{'label': 'ACTIVITY', 'score': float('nan')}],
                            [{'label': 'FAKE', 'score': .9}], [{'label': 'ACTIVITY', 'score': 1.2}]):
            classifier = SimpleNamespace(classify=lambda items: predictions)
            with self.subTest(predictions=predictions), patch.dict('sys.modules', {'ensemble_classifier': classifier}):
                with self.assertRaises(ValueError):
                    extraction.extract_document('D1', text)

    def test_low_confidence_deliverable_cannot_bypass_plan_threshold(self):
        text = 'We will implement the project dashboard.'
        classifier = SimpleNamespace(classify=lambda items: [{'label': 'ACTIVITY', 'score': .2}])
        with patch.dict('sys.modules', {'ensemble_classifier': classifier}):
            record = extraction.extract_document('D1', text)
        self.assertEqual(record['fields']['deliverables'][0]['label_score'], .2)
        self.assertEqual(planning.generate_plan(record, source_text=text)['steps'], [])

    def test_failed_source_offset_is_never_published_as_verified_evidence(self):
        item = dict(text='We will implement the project.', section='Scope', prev='', pos=0.,
                    start=0, end=99, next_text='')
        with patch.object(extraction, 'sentences_with_context', return_value=iter([item])):
            with self.assertRaises(ValueError):
                extraction.extract_document('D1', 'Other source.', use_model=False)


class GroundedPlanningTests(unittest.TestCase):
    def setUp(self):
        self.text = '# Roadmap\nWe will implement the project dashboard.'
        self.record = extraction.extract_document('D1', self.text, use_model=False)

    def test_plan_preserves_source_wording_and_reports_verification_level(self):
        plan = planning.generate_plan(self.record, source_text=self.text)
        self.assertEqual(plan['grounding_validation'], 'source_hash_and_spans')
        self.assertEqual([x['description'] for x in plan['steps']], ['We will implement the project dashboard.'])
        legacy = planning.generate_plan(self.record)
        self.assertEqual(legacy['grounding_validation'], 'record_consistency_only')

    def test_modified_source_or_stale_hash_cannot_ground_plan(self):
        with self.assertRaises(ValueError):
            planning.generate_plan(self.record, source_text=self.text + '\nchanged')

    def test_verified_flag_does_not_authorize_fabricated_value_or_conflicting_id(self):
        record = deepcopy(self.record)
        record['fields']['activities'][0]['value'] = 'We will deploy an unsupported payment system.'
        with self.assertRaises(ValueError):
            planning.generate_plan(record)
        record = deepcopy(self.record)
        entry = record['fields']['deliverables'][0]
        entry['quote'] = entry['value'] = 'We will build something different.'
        entry['char_end'] = entry['char_start'] + len(entry['quote'])
        with self.assertRaises(ValueError):
            planning.generate_plan(record)

    def test_citation_id_alone_does_not_validate_added_plan_wording(self):
        plan = planning.generate_plan(self.record, source_text=self.text)
        evidence = {e['evidence_id']: e for es in self.record['fields'].values() for e in es}
        plan['steps'][0]['description'] += ' Budget: $100,000.'
        with self.assertRaises(ValueError):
            planning.validate_plan(plan, set(evidence), evidence_by_id=evidence)

    def test_table_value_is_only_a_normalization_of_source_quote(self):
        text = '# Roadmap\n| Implement project dashboard | Week 3 |'
        record = extraction.extract_document('D1', text, use_model=False)
        plan = planning.generate_plan(record, source_text=text)
        self.assertEqual(plan['steps'][0]['description'], 'Implement project dashboard Week 3')

    def test_completed_and_past_work_is_not_reissued_as_proposed_task(self):
        for sentence in ['Implement the project dashboard (completed)',
                         'We tested the project dashboard in week 3.']:
            text = '# Roadmap\n' + sentence
            record = extraction.extract_document('D1', text, use_model=False)
            self.assertEqual(planning.generate_plan(record, source_text=text)['steps'], [])


class ExtractionRunnerTests(unittest.TestCase):
    def make_sources(self, root):
        for name in ('D1', 'D2'):
            folder = root / 'cleaned' / name
            folder.mkdir(parents=True)
            (folder / 'source.md').write_text('We will implement the project dashboard.', encoding='utf-8')
        return root / 'cleaned'

    def test_module_import_does_not_write_or_run_inference(self):
        with patch.object(Path, 'open', side_effect=AssertionError('import wrote output')), \
             patch.object(Path, 'mkdir', side_effect=AssertionError('import created a folder')), \
             patch.object(extraction, 'extract_document', side_effect=AssertionError('import ran inference')):
            importlib.reload(run_extraction)

    def test_failure_leaves_previous_output_untouched_and_removes_temporary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cleaned = self.make_sources(root)
            out = root / 'results.jsonl'
            out.write_text('old verified artifact', encoding='utf-8')
            real_extract = extraction.extract_document
            def fail_second(doc, text, **kwargs):
                if doc == 'D2':
                    raise ValueError('simulated inference failure')
                return real_extract(doc, text, **kwargs)
            with patch.object(extraction, 'extract_document', side_effect=fail_second):
                with self.assertRaisesRegex(RuntimeError, 'D2'):
                    run_extraction.run(cleaned_dir=cleaned, output_path=out, use_model=False, overwrite=True)
            self.assertEqual(out.read_text(encoding='utf-8'), 'old verified artifact')
            self.assertEqual(list(root.glob('*.tmp')), [])

    def test_success_is_complete_and_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cleaned = self.make_sources(root)
            out = root / 'results.jsonl'
            result = run_extraction.run(cleaned_dir=cleaned, output_path=out, use_model=False)
            rows = [json.loads(x) for x in out.read_text(encoding='utf-8').splitlines()]
            self.assertEqual([x['document_id'] for x in rows], ['D1', 'D2'])
            self.assertEqual(result['documents'], 2)
            with self.assertRaises(FileExistsError):
                run_extraction.run(cleaned_dir=cleaned, output_path=out, use_model=False)

    def test_ambiguous_document_source_fails_before_output_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cleaned = self.make_sources(root)
            (cleaned / 'D1' / 'second.txt').write_text('Ambiguous source', encoding='utf-8')
            out = root / 'results.jsonl'
            with self.assertRaises(ValueError):
                run_extraction.run(cleaned_dir=cleaned, output_path=out, use_model=False)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
