"""Exact-neighbor context evidence checks; synthetic sources, no models."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import classifier_context_review as review
from classifier_context_v2 import parse_source, find_matches


def write_rows(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class NeighborEvidenceTests(unittest.TestCase):
    def select(self, text, previous='Before B.', following='After B.', target='Repeat target.'):
        parsed = parse_source(text)
        return review.neighbor_supported_section(parsed, dict(prev_text=previous, next_text=following), find_matches(parsed, target))

    def test_two_exact_adjacent_neighbors_disambiguate_repeat_paths(self):
        result = self.select('# A\nBefore A. Repeat target. After A.\n# B\nBefore B. Repeat target. After B.')
        self.assertEqual(result['candidate_section'], 'B')
        self.assertEqual(len(result['supported_matches']), 1)
        self.assertFalse(result['human_verified'])
        self.assertFalse(result['pdf_verified'])

    def test_one_missing_neighbor_does_not_resolve(self):
        result = self.select('# B\nBefore B. Repeat target. After B.', following='')
        self.assertEqual(result['candidate_section'], '')

    def test_two_identical_neighbor_triples_under_different_paths_stay_unknown(self):
        result = self.select('# A\nBefore B. Repeat target. After B.\n# B\nBefore B. Repeat target. After B.')
        self.assertEqual(len(result['supported_matches']), 2)
        self.assertEqual(result['candidate_section'], '')

    def test_intervening_text_is_not_guessed_away(self):
        result = self.select('# B\nBefore B. Intervening material. Repeat target. After B.')
        self.assertEqual(result['candidate_section'], '')

    def test_headingless_evidence_does_not_invent_heading(self):
        result = self.select('Before B. Repeat target. After B.')
        self.assertEqual(result['candidate_section'], '')

    def test_source_snippets_keep_exact_lines_and_validate_ranges(self):
        result = review.source_snippet('# A\nBefore.\nTarget.\nAfter.\n', 3, 3, radius=1)
        self.assertEqual(result['text'], '2: Before.\n3: Target.\n4: After.')
        with self.assertRaises(ValueError):
            review.source_snippet('Only line.', 2, 2)


class PacketTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.version = self.root / 'version'
        self.version.mkdir()
        self.files = []
        self.manifest = dict(version='synthetic-context-v2.1', splits={}, evidence_files={})
        source_manifest, trace = [], []
        for split, doc, source, target, previous, following, status in [
            ('train', 'D1', '# A\nBefore A. Repeat target. After A.\n# B\nBefore B. Repeat target. After B.', 'Repeat target.', 'Before B.', 'After B.', 'multiple_section_paths_needs_review'),
            ('validation', 'D2', '# V\nBefore absent. Other. After absent.', 'Missing target.', 'Before absent.', 'After absent.', 'no_normalized_match_needs_review')]:
            path = self.root / (doc + '.md')
            path.write_text(source, encoding='utf-8')
            self.files.append(path)
            source_manifest.append(dict(document_id=doc, sources=[dict(path=str(path), sha256=review.sha(path.read_bytes()), status='readable_utf8')]))
            row = dict(document_id=doc, text=target, label='OTHER', section='', prev_text=previous, pos='0.5', next_text=following)
            original = self.root / (split + '_frozen.csv')
            write_rows(original, [row])
            self.files.append(original)
            self.manifest['splits'][split] = dict(original_input_file=str(original), original_input_file_sha256=review.sha(original.read_bytes()))
            prepared = self.version / ('training_rows.csv' if split == 'train' else 'validation_rows.csv')
            write_rows(prepared, [row])
            matches = [dict(source_path=str(path), **hit) for hit in find_matches(parse_source(source), target)]
            trace.append(dict(split=split, row_index='0', document_id=doc, text=target, reference_label_unchanged='OTHER',
                              old_section='', new_section='', resolved='False', changed='False', status=status,
                              matches=json.dumps(matches), source_paths=json.dumps([str(path)]), human_verified='False', pdf_verified='False'))
        (self.version / 'source_manifest.json').write_text(json.dumps(source_manifest), encoding='utf-8')
        for name in ['section_changes_and_provenance.csv', 'section_unresolved_for_review.csv']:
            write_rows(self.version / name, trace)
        for path in self.version.iterdir():
            self.manifest['evidence_files'][path.name] = review.sha(path.read_bytes())
        (self.version / 'preparation_manifest.json').write_text(json.dumps(self.manifest), encoding='utf-8')
        self.files.extend(self.version.iterdir())

    def test_packet_export_is_useful_and_leaves_frozen_sources_unchanged(self):
        before = {path: path.read_bytes() for path in self.files}
        output = self.root / 'review'
        summary = review.prepare(self.version, output)
        self.assertEqual(summary['unresolved_input_rows'], 2)
        self.assertEqual(summary['exact_neighbor_supported'], 1)
        self.assertEqual(summary['remaining_unknown'], 1)
        self.assertFalse(summary['dataset_rows_modified'])
        self.assertFalse(summary['semantic_labels_adjudicated'])
        self.assertEqual({path: path.read_bytes() for path in self.files}, before)
        evidence = json.loads((output / 'source_evidence.json').read_text(encoding='utf-8'))
        self.assertEqual(evidence[0]['item']['proposed_section'], 'B')
        self.assertEqual(evidence[1]['item']['proposed_section'], '')
        self.assertTrue(evidence[1]['sources'][0]['neighbors']['prev_text']['shown_matches'])
        self.assertTrue((output / 'review_items.csv').is_file())
        self.assertIn('Tải CSV nhận xét', (output / 'KIEM_TRA_SECTION.html').read_text(encoding='utf-8'))
        with self.assertRaises(FileExistsError):
            review.prepare(self.version, output)

    def test_changed_markdown_source_is_rejected(self):
        (self.root / 'D1.md').write_text('# Different\nSource changed', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Markdown source changed'):
            review.build_evidence(self.version)

    def test_changed_frozen_row_is_rejected(self):
        (self.root / 'train_frozen.csv').write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'Frozen original'):
            review.build_evidence(self.version)

    def test_changed_trace_is_rejected_before_any_packet_write(self):
        (self.version / 'section_unresolved_for_review.csv').write_text('tampered')
        output = self.root / 'review'
        with self.assertRaisesRegex(ValueError, 'evidence hash changed'):
            review.prepare(self.version, output)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
