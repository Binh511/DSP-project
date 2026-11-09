"""Exercise the actual editable review workflow without model inference."""
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from extraction import extract_document, sentences_with_context
from planning import generate_plan
from psai_project.evaluation import reference_task_sha256
from tools import review_project_quality as cli

TEXT = '# Proposal\nWe will implement the inventory API.\n'


class ReviewCommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'run'
        self.run.mkdir()
        record = extract_document('demo', TEXT, use_model=False)
        plan = generate_plan(record, source_text=TEXT)
        metadata = {'source_text_sha256': hashlib.sha256(TEXT.encode()).hexdigest(),
                    'source_exposure': 'known_development'}
        (self.run / 'source.txt').write_text(TEXT, encoding='utf-8', newline='')
        (self.run / 'analysis.json').write_text(json.dumps({'document_id': 'demo', 'extraction': record,
            'plan': plan, 'retrieval': {}, 'metadata': metadata}), encoding='utf-8')
        (self.run / 'manifest.json').write_text(json.dumps(metadata), encoding='utf-8')
        self.packet = self.root / 'packet'
        with patch.object(cli, 'ROOT', self.root):
            cli.prepare([self.run], self.packet)

    def csv_rows(self, name):
        with (self.packet / name).open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))

    def replace_csv(self, name, rows):
        path = self.packet / name
        with path.open(encoding='utf-8-sig', newline='') as stream:
            columns = next(csv.reader(stream))
        with path.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def fill_synthetic_decisions(self):
        bundle = cli.read(self.packet / 'review.json')
        document = bundle['documents'][0]
        actor = {'reviewer_id': 'fixture-AI', 'rationale': 'Synthetic test decision only.'}
        self.replace_csv('reviewers.csv', [{'reviewer_id': 'fixture-AI', 'kind': 'ai', 'independent': 'false'}])
        prediction = document['prediction_tasks'][0]
        span = prediction['source_spans'][0]
        gold = {'task_id': 'G001', 'description': prediction['description'], 'source_spans': [span]}
        self.replace_csv('gold_tasks.csv', [{'case_id': document['case_id'], 'task_id': 'G001',
            'description': gold['description'], 'char_start': span['char_start'], 'char_end': span['char_end'],
            'quote': span['quote'], 'supported': 'true', **actor}])
        self.replace_csv('task_matches.csv', [{'case_id': document['case_id'],
            'prediction_task_id': prediction['task_id'], 'prediction_task_sha256': prediction['task_sha256'],
            'gold_task_id': 'G001', 'gold_task_sha256': reference_task_sha256(gold),
            'equivalent': 'true', **actor}])
        reviews = self.csv_rows('task_reviews.csv')
        for row in reviews:
            row.update({'evidence_correct': 'true', **actor})
        self.replace_csv('task_reviews.csv', reviews)
        rows = self.csv_rows('document_reviews.csv')
        for row in rows:
            for kind in ('reference', 'matching'):
                row.update({kind + '_status': 'complete', kind + '_reviewer_id': actor['reviewer_id'],
                            kind + '_rationale': actor['rationale']})
        self.replace_csv('document_reviews.csv', rows)

    def test_packet_remains_unscored_and_copy_is_portable(self):
        score = cli.score(self.packet / 'review.json')
        self.assertEqual(score['status'], 'incomplete')
        self.assertIsNone(score['metrics'])
        self.assertEqual(self.csv_rows('gold_tasks.csv'), [])
        screening = cli.read(self.packet / 'source_screening.json')
        self.assertEqual(screening['documents'][0]['status'], 'known_development_source')
        with self.assertRaises(FileExistsError):
            cli.prepare([self.run], self.packet)

    def test_csv_semantic_decisions_round_trip_to_diagnostic_score(self):
        self.fill_synthetic_decisions()
        result = cli.import_csv(self.packet / 'review.json', self.packet / 'reviewed.json')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['metrics']['task_matching']['micro']['tp'], 1)
        self.assertIn('ai', result['review_scope'])
        self.assertFalse(result['final_test_gate_passed'])
        self.assertTrue((self.packet / 'reviewed.gold_catalog.csv').is_file())
        with self.assertRaisesRegex(ValueError, 'two distinct saved'):
            cli.compare([self.packet / 'reviewed.json', self.packet / 'reviewed.json'])

    def test_import_cannot_reuse_complete_document_review_when_csv_row_removed(self):
        self.fill_synthetic_decisions()
        cli.import_csv(self.packet / 'review.json', self.packet / 'reviewed.json')
        self.replace_csv('document_reviews.csv', [])
        with self.assertRaisesRegex(ValueError, 'cover each case'):
            cli.import_csv(self.packet / 'reviewed.json', self.packet / 'second.json')

    def test_changed_source_and_changed_gold_match_are_rejected(self):
        self.fill_synthetic_decisions()
        rows = self.csv_rows('gold_tasks.csv')
        rows[0]['description'] = 'Changed meaning after semantic decision.'
        self.replace_csv('gold_tasks.csv', rows)
        with self.assertRaisesRegex(ValueError, 'reference hash mismatch'):
            cli.import_csv(self.packet / 'review.json', self.packet / 'reviewed.json')
        bundle = cli.read(self.packet / 'review.json')
        (self.packet / bundle['documents'][0]['run_dir'] / 'source.txt').write_text('changed', encoding='utf-8')
        with self.assertRaises(ValueError):
            cli.score(self.packet / 'review.json')

    def test_bad_boolean_or_unknown_document_is_rejected(self):
        self.assertIsNone(cli.boolean(''))
        with self.assertRaises(ValueError):
            cli.boolean('yes')
        rows = self.csv_rows('task_reviews.csv')
        rows[0]['case_id'] = 'unknown'
        self.replace_csv('task_reviews.csv', rows)
        with self.assertRaisesRegex(ValueError, 'Unknown case'):
            cli.import_csv(self.packet / 'review.json', self.packet / 'reviewed.json')

    def test_current_sentence_splitter_preserves_section_symbols(self):
        text = '# Scope\nWe will implement the languages (§5.1) and tests (§5.2).\n'
        items = list(sentences_with_context(text))
        matching = [item for item in items if '§5.1' in item['text']]
        self.assertEqual(len(matching), 1)
        for item in items:
            self.assertEqual(text[item['start']:item['end']], item['text'])

    def test_csv_source_text_round_trip_never_becomes_a_spreadsheet_formula(self):
        for text in ['=HYPERLINK("somewhere")', '  - source bullet', '@source', '+123', 'PSAI_TEXT:literal', "'quoted"]:
            with self.subTest(text=text):
                encoded = cli.csv_cell(text)
                self.assertFalse(encoded.lstrip().startswith(('=', '+', '-', '@')))
                self.assertEqual(cli.decoded_cell(encoded), text)

    def test_final_gate_suppresses_even_completed_retrieval_judgments(self):
        analysis = cli.read(self.run / 'analysis.json')
        start, quote = TEXT.index('We will'), TEXT[TEXT.index('We will'):].strip()
        analysis['retrieval'] = {'hits': [{'quote': quote, 'char_start': start,
            'char_end': start + len(quote), 'verified': True, 'role': 'document',
            'source_sha256': hashlib.sha256(TEXT.encode()).hexdigest()}]}
        (self.run / 'analysis.json').write_text(json.dumps(analysis), encoding='utf-8')
        self.packet = self.root / 'retrieval_packet'
        with patch.object(cli, 'ROOT', self.root):
            cli.prepare([self.run], self.packet, dataset_kind='independent_final_test')
        self.fill_synthetic_decisions()
        self.replace_csv('reviewers.csv', [{'reviewer_id': 'fixture-AI', 'kind': 'human', 'independent': 'true'}])
        rows = self.csv_rows('retrieval_reviews.csv')
        rows[0].update({'relevant': 'true', 'source_support_correct': 'true',
                       'reviewer_id': 'fixture-AI', 'rationale': 'Synthetic gate test only.'})
        self.replace_csv('retrieval_reviews.csv', rows)
        with patch.object(cli, 'ROOT', self.root):
            result = cli.import_csv(self.packet / 'review.json', self.packet / 'reviewed.json')
        self.assertEqual(result['status'], 'incomplete')
        self.assertFalse(result['final_test_gate_passed'])
        self.assertIsNone(result['metrics'])
        self.assertIsNone(result['retrieval_relevance']['metrics'])


if __name__ == '__main__':
    unittest.main()
