from copy import deepcopy
import hashlib
import unittest

from extraction import extract_document
from planning import generate_plan
from psai_reports import render_report_html


class PortableReportTests(unittest.TestCase):
    def setUp(self):
        self.text = '# Roadmap\nWe will implement the local dashboard.\n<script>alert(1)</script>'
        self.record = extract_document('DEMO', self.text, use_model=False)
        self.plan = generate_plan(self.record, source_text=self.text)
        self.metadata = {'classifier': 'rules', 'source_text_sha256': hashlib.sha256(self.text.encode()).hexdigest()}

    def test_embeds_source_and_escapes_markup_without_remote_dependencies(self):
        output = render_report_html(self.record, self.plan, self.text, self.metadata)
        self.assertIn('&lt;script&gt;alert(1)&lt;/script&gt;', output)
        self.assertNotIn('<script>', output)
        self.assertIn('href="#src-', output)
        self.assertIn('Quy tắc', output)
        self.assertNotIn('https://', output)

    def test_rejects_modified_source_and_fabricated_quote(self):
        with self.assertRaises(ValueError):
            render_report_html(self.record, self.plan, self.text + 'changed', self.metadata)
        modified = deepcopy(self.record)
        modified['fields']['activities'][0]['quote'] = 'Fabricated work.'
        with self.assertRaises(ValueError):
            render_report_html(modified, self.plan, self.text, self.metadata)

    def test_resources_require_embedded_verified_source_snapshot(self):
        resource = 'Python collection guidance for local metrics.'
        digest = hashlib.sha256(resource.encode()).hexdigest()
        hit = {'evidence_id': 'resource:1', 'quote': resource, 'char_start': 0,
               'char_end': len(resource), 'source_sha256': digest, 'verified': True, 'role': 'resource'}
        with self.assertRaises(ValueError):
            render_report_html(self.record, self.plan, self.text, self.metadata, {'hits': [hit]})
        output = render_report_html(self.record, self.plan, self.text, self.metadata,
                                    {'hits': [hit], 'source_texts': {digest: resource}})
        self.assertIn(resource, output)
        self.assertIn('Tài nguyên tham khảo', output)

    def test_resource_snapshot_anchor_ids_are_not_an_html_injection_surface(self):
        resource = 'An authored resource for source verification.'
        digest = hashlib.sha256(resource.encode()).hexdigest()
        identity = 'bad\" onclick=\"alert(1)'
        with self.assertRaises(ValueError):
            render_report_html(self.record, self.plan, self.text, self.metadata,
                               {'sources': {identity: {'source_sha256': digest}},
                                'source_texts': {identity: resource}})


if __name__ == '__main__':
    unittest.main()
