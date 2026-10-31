"""Synthetic CPU checks for review imports; no inference or dataset changes."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import classifier_adjudication as adjudication


def write_csv(path, rows, columns):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class AdjudicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.packet = self.root / 'human_review'
        self.packet.mkdir()
        analysis = self.root / 'analysis'
        analysis.mkdir()
        self.rows, keys = [], []
        for index in range(91):
            training = index < 57
            row = dict(item_id=f'item-{index:03}',
                       review_scope='training_ai_proposal' if training else 'validation_disagreement',
                       document_id=f'D{index:03}', text=f'Item {index}: build the project.',
                       section='Scope', prev_text='Earlier sentence.', next_text='Later sentence.')
            row['content_sha256'] = adjudication.content_sha256(row)
            row.update({field: '' for field in adjudication.ANSWERS})
            self.rows.append(row)
            key = {field: row[field] for field in adjudication.IMMUTABLE}
            key.update(reference_label='REQUIREMENT' if training else 'EXPECTED_RESULT',
                       original_label='REQUIREMENT' if training else '',
                       proposed_label='ACTIVITY' if training else '',
                       source_line=f'{index + 1}.0' if training else '')
            keys.append(key)
        write_csv(self.packet / 'PHAN_XU_91_CAU.csv', self.rows, adjudication.PUBLIC_COLUMNS)
        write_csv(self.packet / 'coordinator_key.csv', keys, tuple(keys[0]))
        source_hashes = {}
        for name in ('seed_2026_regressions.csv', 'ai_57_effective_proposals.csv'):
            training = name == 'ai_57_effective_proposals.csv'
            source_rows = []
            for row in keys[:57] if training else keys[57:]:
                source = {field: row[field] for field in adjudication.IMMUTABLE if field != 'review_scope'}
                if training:
                    source.update(original_label=row['original_label'], proposed_label=row['proposed_label'],
                                  source_line=row['source_line'])
                else:
                    source['true_label'] = row['reference_label']
                source_rows.append(source)
            write_csv(analysis / name, source_rows, tuple(source_rows[0]))
            source_hashes[name] = adjudication.digest((analysis / name).read_bytes())
        payload = [{field: row[field] for field in (*adjudication.IMMUTABLE, 'content_sha256')} for row in self.rows]
        manifest = dict(items=91, training_items=57, validation_items=34,
                        packet_sha256=adjudication.digest(json.dumps(payload, ensure_ascii=False).replace('<', r'\u003c').encode()),
                        source_sha256=source_hashes, human_completed=False, labels_applied=False)
        (self.packet / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        self.submission = self.root / 'export.csv'
        self.write_submission()

    def write_submission(self, rows=None):
        write_csv(self.submission, self.rows if rows is None else rows, adjudication.PUBLIC_COLUMNS)

    def fill(self, index, annotation):
        self.rows[index].update(annotation=annotation, reviewer='Person A',
                                reason='Main function checked against the stored context.', source_checked='unchecked')

    def validate(self, complete=False):
        return adjudication.validate_submission(self.submission, packet_dir=self.packet, complete=complete)

    def test_blank_sheet_is_pending_and_never_certified_complete(self):
        result = self.validate()
        self.assertEqual(result['summary']['training']['pending'], 57)
        self.assertEqual(result['summary']['validation']['pending'], 34)
        self.assertEqual(result['proposed_overlay'], [])
        self.assertFalse(result['summary']['all_annotations_present'])
        self.assertFalse(result['summary']['reviewer_identity_independently_verified'])
        self.assertFalse(result['summary']['final_test_approved'])

    def test_reordered_export_matches_by_ids_and_counts_decision_outcomes(self):
        self.fill(0, 'ACTIVITY')
        self.fill(1, 'REQUIREMENT')
        self.fill(2, 'OBJECTIVE')
        self.fill(57, 'EXPECTED_RESULT')
        self.fill(58, 'OTHER')
        self.write_submission(list(reversed(self.rows)))
        result = self.validate()
        training, validation = result['summary']['training'], result['summary']['validation']
        self.assertEqual([training[k] for k in ('completed', 'accepted_AI', 'rejected_AI', 'original_retained', 'alternative')], [3, 1, 2, 1, 1])
        self.assertEqual([validation[k] for k in ('completed', 'accepted_AI', 'rejected_AI', 'original_retained', 'alternative')], [2, 0, 0, 1, 1])
        self.assertEqual(len(result['proposed_overlay']), 3)
        self.assertTrue(all(row['overlay_status'] == 'PROPOSED_NOT_APPLIED' for row in result['proposed_overlay']))
        self.assertEqual(result['proposed_overlay'][0]['source_line'], '1')

    def test_context_tampering_rejected_even_when_submitted_hash_is_unchanged(self):
        old_hash = self.rows[0]['content_sha256']
        self.rows[0]['prev_text'] = 'New misleading context.'
        self.assertEqual(self.rows[0]['content_sha256'], old_hash)
        self.write_submission()
        with self.assertRaisesRegex(ValueError, 'Immutable context'):
            self.validate()

    def test_context_tampering_rejected_even_with_recomputed_submitter_hash(self):
        self.rows[0]['text'] = 'Changed source sentence.'
        self.rows[0]['content_sha256'] = adjudication.content_sha256(self.rows[0])
        self.write_submission()
        with self.assertRaises(ValueError):
            self.validate()

    def test_claimed_hash_is_recomputed_not_trusted(self):
        self.rows[0]['content_sha256'] = '0' * 64
        self.write_submission()
        with self.assertRaisesRegex(ValueError, 'content hash'):
            self.validate()

    def test_invalid_label_rejected_in_partial_mode(self):
        self.fill(0, 'REQUIREMENTS')
        self.write_submission()
        with self.assertRaisesRegex(ValueError, 'Invalid annotation'):
            self.validate()

    def test_completed_label_requires_reviewer_and_reason(self):
        self.fill(0, 'ACTIVITY')
        for field, value in [('reviewer', ' '), ('reason', ''), ('reviewer', 'ChatGPT')]:
            with self.subTest(field=field, value=value):
                self.fill(0, 'ACTIVITY')
                self.rows[0][field] = value
                self.write_submission()
                with self.assertRaisesRegex(ValueError, 'reviewer and reason'):
                    self.validate()

    def test_source_check_requires_allowed_choice_and_page_if_yes(self):
        self.fill(0, 'ACTIVITY')
        for choice in ['maybe', 'yes']:
            with self.subTest(choice=choice):
                self.rows[0]['source_checked'] = choice
                self.write_submission()
                with self.assertRaises(ValueError):
                    self.validate()
        self.rows[0]['source_page'] = 'Original PDF, page 3'
        self.write_submission()
        result = self.validate()
        self.assertEqual(result['summary']['training']['source_checked_yes'], 1)
        self.assertFalse(result['summary']['reviewer_identity_independently_verified'])

    def test_complete_mode_requires_all_91_annotations(self):
        with self.assertRaisesRegex(ValueError, 'undecided'):
            self.validate(complete=True)
        for index in range(91):
            self.fill(index, 'ACTIVITY' if index < 57 else 'EXPECTED_RESULT')
        self.write_submission()
        result = self.validate(complete=True)
        self.assertTrue(result['summary']['all_annotations_present'])
        self.assertEqual(len(result['proposed_overlay']), 57)
        self.assertFalse(result['summary']['labels_applied'])

    def test_missing_added_and_duplicate_ids_are_rejected(self):
        variants = [self.rows[:-1], [*self.rows, self.rows[0]],
                    [*self.rows[:-1], {**self.rows[-1], 'item_id': 'unknown'}]]
        for rows in variants:
            with self.subTest(length=len(rows)):
                self.write_submission(rows)
                with self.assertRaises(ValueError):
                    self.validate()

    def test_import_preserves_exact_submission_and_all_source_manifests(self):
        gate = self.root / 'review_gate.json'
        gate.write_text('{"approved_for_final_test": false}', encoding='utf-8')
        frozen = self.root / 'frozen.csv'
        frozen.write_bytes(b'original frozen dataset\n')
        files = [*self.packet.iterdir(), gate, frozen]
        before = {path: path.read_bytes() for path in files}
        self.fill(0, 'ACTIVITY')
        self.write_submission(list(reversed(self.rows)))
        raw = self.submission.read_bytes()
        destination = self.root / 'imported'
        summary = adjudication.import_adjudication(self.submission, destination, packet_dir=self.packet)
        self.assertEqual((destination / 'submitted.csv').read_bytes(), raw)
        self.assertEqual(summary['submission_sha256'], adjudication.digest(raw))
        self.assertEqual({path: path.read_bytes() for path in files}, before)
        self.assertFalse(json.loads((self.packet / 'manifest.json').read_text())['human_completed'])
        self.assertTrue((destination / 'training_decisions.csv').is_file())
        self.assertTrue((destination / 'validation_decisions.csv').is_file())
        self.assertTrue((destination / 'VALIDATION_REPORT.md').is_file())
        original_output = {path.name: path.read_bytes() for path in destination.iterdir()}
        with self.assertRaises(FileExistsError):
            adjudication.import_adjudication(self.submission, destination, packet_dir=self.packet)
        self.assertEqual({path.name: path.read_bytes() for path in destination.iterdir()}, original_output)

    def test_modified_analysis_evidence_invalidates_packet(self):
        (self.root / 'analysis' / 'ai_57_effective_proposals.csv').write_text('different analysis')
        with self.assertRaisesRegex(ValueError, 'analysis source changed'):
            self.validate()

    def test_valid_but_tampered_coordinator_labels_rejected_by_analysis_binding(self):
        path = self.packet / 'coordinator_key.csv'
        with path.open(encoding='utf-8-sig', newline='') as stream:
            original = list(csv.DictReader(stream))
        variants = [(0, {'proposed_label': 'OBJECTIVE'}),
                    (0, {'reference_label': 'OTHER', 'original_label': 'OTHER'}),
                    (57, {'reference_label': 'REQUIREMENT'})]
        for index, changes in variants:
            with self.subTest(index=index, changes=changes):
                rows = [dict(row) for row in original]
                rows[index].update(changes)
                write_csv(path, rows, tuple(rows[0]))
                with self.assertRaisesRegex(ValueError, 'labels differ from analysis source'):
                    self.validate()


if __name__ == '__main__':
    unittest.main(verbosity=2)
