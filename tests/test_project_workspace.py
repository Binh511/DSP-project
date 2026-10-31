"""Offline workspace identity, review and safe-rendering regression checks."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from psai_project.workspace import initial_progress, progress_summary, render_workspace, validate_progress


SOURCE = 'Heading\nImplement a collector.\nDocument deployment.\n'
HASH = hashlib.sha256(SOURCE.encode()).hexdigest()


def fixture():
    first, second = SOURCE.index('Implement'), SOURCE.index('Document')
    a = {'evidence_id': 'fixture:S00000', 'quote': 'Implement a collector.',
         'value': 'Implement a collector.', 'char_start': first, 'char_end': first + 22, 'verified': True}
    b = {'evidence_id': 'fixture:S00001', 'quote': 'Document deployment.',
         'value': 'Document deployment.', 'char_start': second, 'char_end': second + 20, 'verified': True}
    return {'schema_version': 2, 'document_id': 'fixture',
            'metadata': {'source_text_sha256': HASH, 'source_line_starts': [0, 8, 31, 52]},
            'extraction': {'source_sha256': HASH, 'fields': {'activities': [a, b]}},
            'plan': {'source_sha256': HASH, 'steps': [
                {'step': 1, 'task_id': 'T001', 'description': a['value'], 'parent_task_id': None,
                 'evidence_ids': [a['evidence_id']], 'checklist': [
                     {'description': 'Implement a collector', 'evidence_ids': [a['evidence_id']]}]},
                {'step': 2, 'task_id': 'T002', 'description': b['value'], 'parent_task_id': 'T001',
                 'evidence_ids': [b['evidence_id']], 'checklist': []}],
                'dependency_candidates': [{'before_task_id': 'T001', 'after_task_id': 'T002'}],
                'warnings': ['Human review required.']}}


class ProjectWorkspaceTests(unittest.TestCase):
    def test_initial_progress_is_separate_and_source_remains_unreviewed(self):
        analysis = fixture()
        original = deepcopy(analysis)
        progress = initial_progress(analysis)
        progress['tasks'][0]['status'] = 'done'
        progress['tasks'][0]['checklist'][0]['done'] = True
        progress['tasks'][0]['notes'] = 'Reviewed implementation manually.'
        validated = validate_progress(progress, analysis)
        self.assertEqual(validated['source_review_status'], 'unreviewed')
        self.assertEqual(analysis, original)
        summary = progress_summary(progress, analysis)
        self.assertEqual(summary['task_done_percent'], 50.0)
        self.assertEqual(summary['checklist_done'], 1)
        self.assertFalse(summary['research_quality_measured'])

    def test_hashes_and_analysis_version_reject_foreign_progress(self):
        analysis = fixture()
        for field in ('document_id', 'source_sha256', 'analysis_sha256', 'source_review_status'):
            progress = initial_progress(analysis)
            progress[field] = 'changed'
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_progress(progress, analysis)
        progress = initial_progress(analysis)
        analysis['plan']['steps'][0]['description'] += ' Changed plan.'
        with self.assertRaisesRegex(ValueError, 'analysis_sha256'):
            validate_progress(progress, analysis)

    def test_unknown_and_repeated_task_checklist_and_dependency_ids_are_rejected(self):
        analysis = fixture()
        cases = []
        unknown = initial_progress(analysis); unknown['tasks'][0]['task_id'] = 'UNKNOWN'; cases.append(unknown)
        duplicate = initial_progress(analysis); duplicate['tasks'][1] = deepcopy(duplicate['tasks'][0]); cases.append(duplicate)
        unknown = initial_progress(analysis); unknown['tasks'][0]['checklist'][0]['item_id'] = 'WRONG'; cases.append(unknown)
        unknown = initial_progress(analysis); unknown['dependencies'][0]['dependency_id'] = 'T002->T001'; cases.append(unknown)
        missing = initial_progress(analysis); missing['tasks'][0]['checklist'] = []; cases.append(missing)
        for progress in cases:
            with self.subTest(progress=progress), self.assertRaises(ValueError):
                validate_progress(progress, analysis)

    def test_strict_json_types_and_unknown_fields_are_rejected(self):
        analysis = fixture()
        cases = []
        value = initial_progress(analysis); value['schema_version'] = True; cases.append(value)
        value = initial_progress(analysis); value['tasks'][0]['status'] = 'finished'; cases.append(value)
        value = initial_progress(analysis); value['tasks'][0]['checklist'][0]['done'] = 1; cases.append(value)
        value = initial_progress(analysis); value['tasks'][0]['notes'] = {}; cases.append(value)
        value = initial_progress(analysis); value['model_quality'] = 1.0; cases.append(value)
        value = initial_progress(analysis); value['notes'] = float('nan'); cases.append(value)
        for progress in cases:
            with self.subTest(progress=progress), self.assertRaises(ValueError):
                validate_progress(progress, analysis)

    def test_dependency_review_requires_human_name(self):
        analysis = fixture()
        progress = initial_progress(analysis)
        progress['dependencies'][0]['decision'] = 'approve'
        with self.assertRaisesRegex(ValueError, 'reviewer name'):
            validate_progress(progress, analysis)
        progress['dependencies'][0]['reviewer'] = 'Nguyễn Văn A'
        self.assertEqual(validate_progress(progress, analysis)['dependencies'][0]['decision'], 'approve')

    def test_approved_dependencies_cannot_form_cycle(self):
        analysis = fixture()
        analysis['plan']['dependency_candidates'].append({'before_task_id': 'T002', 'after_task_id': 'T001'})
        progress = initial_progress(analysis)
        for dependency in progress['dependencies']:
            dependency.update(decision='approve', reviewer='Reviewer')
        with self.assertRaisesRegex(ValueError, 'cycle'):
            validate_progress(progress, analysis)

    def test_html_script_and_event_handlers_cannot_be_imported_as_notes(self):
        analysis = fixture()
        for note in ('<script>alert(1)</script>', '<img src=x onerror=alert(1)>',
                     'javascript:alert(1)', 'onload=bad()', '\x00bad'):
            progress = initial_progress(analysis)
            progress['notes'] = note
            with self.subTest(note=note), self.assertRaises(ValueError):
                validate_progress(progress, analysis)

    def test_source_text_cannot_break_out_of_inert_json_or_html_title(self):
        analysis = fixture()
        analysis['document_id'] = '</title><script>alert(1)</script>__DATA__'
        analysis['plan']['steps'][0]['description'] = '</script><script>alert(2)</script>'
        html = render_workspace(analysis)
        self.assertNotIn('<script>alert(1)', html)
        self.assertNotIn('<script>alert(2)', html)
        self.assertIn('\\u003c/script\\u003e', html)
        scripts = re.findall(r'<script id="workspace-data" type="application/json">(.*?)</script>', html, re.S)
        self.assertEqual(len(scripts), 1)
        self.assertEqual(json.loads(scripts[0])['identity']['document_id'], analysis['document_id'])
        self.assertNotIn('fetch(', html)
        self.assertNotIn('XMLHttpRequest', html)
        self.assertNotIn('.innerHTML', html)

    def test_source_links_use_exact_line_starts_and_summary_handles_empty_plan(self):
        analysis = fixture()
        html = render_workspace(analysis)
        self.assertIn(f'report.html#src-{HASH}-L2', html)
        analysis['plan']['steps'] = []
        analysis['plan']['dependency_candidates'] = []
        self.assertEqual(progress_summary(initial_progress(analysis), analysis)['task_done_percent'], 0.0)

    def test_actual_generated_script_passes_node_syntax_check(self):
        node = shutil.which('node')
        if node is None:
            self.skipTest('Node is not installed; Python safe-rendering tests still run')
        html = render_workspace(fixture())
        script, = re.findall(r'<script>(.*?)</script>', html, re.S)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'workspace.js'
            path.write_text(script, encoding='utf-8')
            result = subprocess.run([node, '--check', str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_browser_validator_rejects_injection_invalid_types_and_unsigned_dependency_review(self):
        node = shutil.which('node')
        if node is None:
            self.skipTest('Node is not installed')
        html = render_workspace(fixture())
        script, = re.findall(r'<script>(.*?)</script>', html, re.S)
        data, = re.findall(r'<script id="workspace-data" type="application/json">(.*?)</script>', html, re.S)
        prefix = script.split('function save()', 1)[0]
        runner = 'global.document={getElementById:()=>({textContent:' + json.dumps(data) + '})};\n' + prefix
        runner += r'''
validate(model.initial_progress);
const attempts=[
 x=>x.notes='<script>alert(1)</script>',
 x=>x.notes='javascript:alert(1)',
 x=>x.tasks[0].status=['done'],
 x=>x.tasks[0].checklist[0].done=1,
 x=>x.dependencies[0].decision=['approve'],
 x=>x.dependencies[0].decision='approve',
 x=>x.tasks[1]=clone(x.tasks[0]),
 x=>x.source_sha256='0'.repeat(64)
];
for(const attempt of attempts){const value=clone(model.initial_progress);attempt(value);let rejected=false;try{validate(value);}catch(_){rejected=true;}if(!rejected)throw Error('Unsafe progress accepted');}
const approved=clone(model.initial_progress);approved.dependencies[0].decision='approve';approved.dependencies[0].reviewer='Reviewer';validate(approved);
'''
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'validator.js'
            path.write_text(runner, encoding='utf-8')
            result = subprocess.run([node, str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
