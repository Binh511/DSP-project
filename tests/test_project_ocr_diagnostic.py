"""Test diagnostic arithmetic and fixture provenance, never OCR model accuracy."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import unicodedata

from tools.evaluate_ocr_diagnostic import (
    aggregate, error_rates, fixture_specs, levenshtein, normalize, reference_text,
    write_references, REFERENCE_ORIGIN,
)


class OCRDiagnosticTests(unittest.TestCase):
    def test_levenshtein_counts_substitutions_insertions_and_deletions(self):
        self.assertEqual(levenshtein('kitten', 'sitting'), 3)
        self.assertEqual(levenshtein('', 'abc'), 3)
        self.assertEqual(levenshtein(['one', 'two'], ['one', 'extra', 'two']), 1)
        self.assertEqual(levenshtein('abcd', 'ad'), 2)
        result = error_rates('one two', 'one three')
        self.assertEqual(result['word_edits'], 1)
        self.assertEqual(result['wer'], .5)

    def test_rates_preserve_empty_denominator_and_can_exceed_one(self):
        result = error_rates('', 'extra words')
        self.assertIsNone(result['cer'])
        self.assertIsNone(result['wer'])
        result = error_rates('a', 'a b c')
        self.assertGreater(result['cer'], 1)
        self.assertEqual(result['wer'], 2)

    def test_normalization_preserves_accents_punctuation_and_composed_unicode(self):
        reference = 'ĐIỆN áp: 24 V.\nNgày 18/01/2027.'
        equivalent = unicodedata.normalize('NFD', reference.lower().replace('\n', '   '))
        self.assertEqual(error_rates(reference, equivalent)['cer'], 0)
        self.assertEqual(normalize('  Mã\n\tA-01.  '), 'mã a-01.')
        self.assertNotEqual(normalize('điện'), normalize('dien'))
        self.assertEqual(normalize('ĐIỆN', remove_diacritics=True), 'dien')

    def test_accent_removal_is_reported_separately_from_true_transcription_error(self):
        reference, prediction = 'Đề xuất bảo trì thiết bị.', 'De xuat bao tri thiet bi.'
        preserved = error_rates(reference, prediction)
        removed = error_rates(reference, prediction, remove_diacritics=True)
        self.assertGreater(preserved['cer'], 0)
        self.assertGreater(preserved['wer'], 0)
        self.assertEqual(removed['cer'], 0)
        self.assertEqual(removed['wer'], 0)

    def test_micro_rates_use_total_reference_lengths_instead_of_average_page_rates(self):
        pages = [
            {'metrics': {'mode': error_rates('a', 'b')}},
            {'metrics': {'mode': error_rates('a a a a', 'a a a a')}},
        ]
        result = aggregate(pages, 'mode')
        self.assertEqual(result['character_edits'], 1)
        self.assertEqual(result['reference_characters'], 8)
        self.assertAlmostEqual(result['micro_cer'], 1 / 8)
        self.assertAlmostEqual(result['micro_wer'], 1 / 5)
        self.assertEqual(result['macro_cer'], .5)

    def test_reference_files_are_known_utf8_before_ocr_with_intact_diacritics_and_hashes(self):
        specs = fixture_specs()
        self.assertEqual(len(specs), 6)
        self.assertEqual({spec['language'] for spec in specs}, {'vi', 'en'})
        for pair in ['prose', 'numbers', 'table']:
            self.assertEqual({spec['language'] for spec in specs if spec['pair_id'] == pair}, {'vi', 'en'})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'references'
            entries = write_references(root, specs)
            for entry in entries:
                raw = (root / entry['transcript']).read_bytes()
                self.assertNotIn(b'\r', raw)
                self.assertEqual(entry['transcript_sha256'], hashlib.sha256(raw).hexdigest())
                self.assertEqual(entry['reference_origin'], REFERENCE_ORIGIN)
                self.assertFalse(entry['human_verified'])
            text = (root / 'vi_numbers.txt').read_text('utf-8')
            self.assertIn('Ngân sách dự kiến là 24.000.000 VND.', text)
            self.assertIn('18/01/2027', text)
            self.assertIn('15/02/2027', text)
            self.assertIn('khôi phục đủ 60 bản ghi', text)
            self.assertEqual(unicodedata.normalize('NFC', text), text)
            table = (root / 'vi_table.txt').read_text('utf-8')
            self.assertIn('A-01 Máy hiện sóng 12 18/01/2027', table)
            self.assertLess(table.index('A-01'), table.index('A-02'))
            self.assertEqual(json.loads((root / 'references.json').read_text('utf-8'))['pages'], entries)
            before = (root / 'vi_numbers.txt').read_bytes()
            with self.assertRaises(FileExistsError):
                write_references(root, specs)
            self.assertEqual((root / 'vi_numbers.txt').read_bytes(), before)

    def test_reference_spec_is_fresh_and_never_depends_on_predictions(self):
        altered = fixture_specs()
        altered[0]['paragraphs'][0] = 'Altered test value'
        self.assertNotIn('Altered test value', reference_text(fixture_specs()[0]))
        self.assertIn('Phòng thí nghiệm cần theo dõi', reference_text(fixture_specs()[0]))


if __name__ == '__main__':
    unittest.main()
