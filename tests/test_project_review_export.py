"""End-to-end export validation for cross-sentence planning evidence, without ML."""
from copy import deepcopy
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import unittest

from extraction import extract_document
from planning import generate_plan
from psai_planner import build_catalog, build_prompt, build_review_context, validate_response
from psai_planner.planner import DEFAULT_CONFIG
from psai_project.report import export_project_report, _validate_local_plan


TEXT = ('# Access review\nWe will implement API access controls.\nWe will test API access controls.\n'
        '## Requirements\nThe API must reject unauthorized updates.\n'
        '## Acceptance\nAcceptance requires API access controls to reject a viewer update request.\n'
        '## Staffing\nThe proposed team has two engineers.\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode('utf-8')).hexdigest()


class Anchors(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links = set(), []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        if tag == 'a' and attrs.get('href', '').startswith('#'):
            self.links.append(attrs['href'][1:])


class ReviewExportTests(unittest.TestCase):
    def fixture(self):
        record = extract_document('case', TEXT, use_model=False)
        record['related_context'] = []
        retrieval = {'mode': 'none', 'hits': [], 'sources': {}}
        catalog = build_catalog(record, TEXT, retrieval)
        acceptance = next(item for item in catalog['context_evidence']
                          if 'Acceptance requires' in item['quote'])
        link = next(item for item in catalog['link_candidates']
                    if item['kind'] == 'checklist' and item['evidence_id'] == acceptance['evidence_id']
                    and item['task_id'] == catalog['tasks'][0]['task_id'])
        raw = json.dumps({'task_ids': [catalog['tasks'][0]['task_id']], 'link_ids': [link['link_id']]})
        messages = build_prompt(catalog)
        plan = validate_response(raw, catalog)
        plan.update({'catalog': catalog, 'messages': messages, 'raw_response': raw,
                     'provenance': {'config': DEFAULT_CONFIG, 'model_loaded': True,
                                    'protocol_version': DEFAULT_CONFIG['protocol_version'],
                                    'catalog_sha256': digest(catalog), 'prompt_sha256': digest(messages)}})
        return record, retrieval, plan

    def export(self, path, record, retrieval, plan, context=None):
        response = {'plan': plan, 'source_plan': generate_plan(record, source_text=TEXT)}
        if context is not None:
            response['review_context'] = context
        return export_project_report(path, TEXT, {'method': 'local_text'},
            {'record': record, 'classifier_predictions': {}},
            {'retrieval': retrieval, 'source_texts': {}}, response,
            {'classifier': 'rules', 'retrieval_mode': 'none', 'planner_mode': 'local'})

    def test_cross_sentence_checklist_exports_and_all_source_links_resolve(self):
        record, retrieval, plan = self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            report = self.export(Path(folder), record, retrieval, plan)
            html = report.read_text('utf-8')
            anchors = Anchors()
            anchors.feed(html)
            self.assertEqual(set(anchors.links) - anchors.ids, set())
            analysis = json.loads((Path(folder) / 'analysis.json').read_text('utf-8'))
            check = analysis['plan']['steps'][0]['checklist'][0]
            self.assertEqual(TEXT[check['char_start']:check['char_end']], check['description'])
            self.assertTrue(check['needs_review'])
            self.assertIsNone(analysis['planning_comparison']['quality_score'])
            self.assertEqual(analysis['review_context']['coverage_status'], 'not_assessed')
            self.assertIn('two engineers', html)

    def test_forged_review_link_cannot_be_exported_as_verified(self):
        record, retrieval, plan = self.fixture()
        context = build_review_context(record, TEXT, retrieval)
        context['human_review_complete'] = True
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            self.export(Path(folder), record, retrieval, plan, context)

    def test_nested_checklist_edits_and_protocol_mismatch_are_rejected(self):
        record, retrieval, plan = self.fixture()
        altered = deepcopy(plan)
        altered['steps'][0]['checklist'][0]['description'] = 'Invent a successful acceptance result'
        with self.assertRaises(ValueError):
            _validate_local_plan(altered, record, TEXT, retrieval, {})
        altered = deepcopy(plan)
        altered['provenance']['protocol_version'] = 'psai-local-evidence-catalog-v1'
        with self.assertRaises(ValueError):
            _validate_local_plan(altered, record, TEXT, retrieval, {})


if __name__ == '__main__':
    unittest.main()
