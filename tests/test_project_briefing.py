"""Check that the formal brief reports evidence without overstating quality."""
from copy import deepcopy
from html.parser import HTMLParser
import unittest

from extraction import extract_document
from planning import generate_plan
from psai_reports import render_report_html
from psai_project.briefing import present_project_report
from psai_rules import extract_facts


class Anchors(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets, self.links = set(), []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.targets.add(attrs['id'])
        if tag == 'a' and attrs.get('href', '').startswith('#'):
            self.links.append(attrs['href'][1:])


class ProjectBriefingTests(unittest.TestCase):
    def setUp(self):
        self.text = '# <img src=x onerror=alert(1)>\nWe will implement the dashboard.\nThe budget is 2000 USD.\n'
        self.record = extract_document('demo', self.text, use_model=False)
        self.plan = generate_plan(self.record, source_text=self.text)
        self.facts = extract_facts('demo', self.text)
        self.metadata = {'classifier': 'rules', 'retrieval_mode': 'bm25', 'planner_mode': 'evidence',
                         'created_utc': '2026-11-08T08:00:00+07:00'}

    def render(self, record=None, plan=None, metadata=None, facts=None, review_context=None):
        record = record if record is not None else self.record
        plan = plan if plan is not None else self.plan
        metadata = metadata if metadata is not None else self.metadata
        portable = render_report_html(record, plan, self.text, metadata)
        return present_project_report(portable, record, plan, {}, facts or self.facts, [], metadata, self.text,
                                      review_context=review_context)

    def test_brief_is_escaped_and_every_internal_link_resolves(self):
        result = self.render()
        self.assertNotIn('<img src=x', result)
        self.assertIn('&lt;img src=x onerror=alert(1)&gt;', result)
        self.assertIn('@page{size:A4', result)
        anchors = Anchors()
        anchors.feed(result)
        self.assertTrue(anchors.links)
        self.assertEqual(set(anchors.links) - anchors.targets, set())
        self.assertIn('2000 USD', result)

    def test_a_source_span_in_multiple_groups_is_counted_once(self):
        record = deepcopy(self.record)
        record['fields']['deliverables'] = [deepcopy(record['fields']['activities'][0])]
        result = self.render(record=record)
        spans = {(item['char_start'], item['char_end'])
                 for items in record['fields'].values() for item in items}
        self.assertIn(f'<strong>{len(spans)}</strong><span>Đoạn nguồn phân biệt đã trích</span>', result)
        self.assertIn('không đo độ đầy đủ hoặc độ chính xác', result)

    def test_missing_or_truthy_review_metadata_does_not_certify_quality(self):
        result = self.render(metadata={**self.metadata, 'human_content_review_complete': 'true',
                                       'final_test_evaluated': 1})
        self.assertIn('Chưa hoàn tất duyệt nội dung bởi người', result)
        self.assertIn('Chưa đánh giá final test độc lập', result)
        self.assertNotIn('Đã ghi nhận duyệt nội dung', result)
        self.assertIn('08/11/2026 · 01:00 UTC', result)

    def test_catalog_limits_and_unselected_tasks_are_disclosed_as_review(self):
        plan = {**self.plan, 'catalog_omitted_counts': {'tasks': 3}, 'unselected_task_ids': ['T012']}
        result = self.render(plan=plan)
        self.assertIn('3 công việc ngoài catalog do giới hạn cấu hình', result)
        self.assertIn('1 công việc trong catalog chưa được chọn', result)
        self.assertIn('bảng này chưa đánh giá yêu cầu đã được đáp ứng', result)

    def test_omission_counts_preserve_units_and_distinguish_catalog_from_prompt(self):
        plan = {**self.plan, 'catalog_omitted_counts': {
            'tasks': 3, 'link_candidates': 5, 'prompt_evidence_without_link_candidate': 7}}
        result = self.render(plan=plan)
        self.assertIn('3 công việc ngoài catalog do giới hạn cấu hình', result)
        self.assertIn('5 cặp liên kết gợi ý không vào danh sách lựa chọn', result)
        self.assertIn('7 đoạn bằng chứng có trong catalog nhưng không đưa vào prompt', result)
        self.assertIn('không cộng thành tổng nội dung bị bỏ sót', result)
        self.assertNotIn('15 mục', result)

    def test_no_extracted_items_or_tasks_can_still_be_presented(self):
        text = '# Empty\nBackground only.'
        record = extract_document('empty', text, use_model=False)
        plan = generate_plan(record, source_text=text)
        base = render_report_html(record, plan, text, {})
        result = present_project_report(base, record, plan, {}, extract_facts('empty', text), [], {}, text)
        self.assertIn('Chưa có công việc đủ điều kiện', result)
        self.assertIn('Chưa ghi nhận', result)
        self.assertIn('Chưa nhận diện theo quy tắc', result)

    def source_item(self, quote, identity, categories=None, value=None):
        start = self.text.index(quote)
        return {'evidence_id': identity, 'quote': quote, 'char_start': start,
                'char_end': start + len(quote), 'source_sha256': self.record['source_sha256'],
                'verified': True, 'categories': categories or [], 'value': value}

    def use_review_source(self):
        self.text = ('# Review\nWe will implement the dashboard.\nWe will test the dashboard.\n'
                     'The dashboard must reject viewer writes.\nTests confirm viewer writes are denied.\n'
                     'Records must be retained for 30 days.\nSchema approval is required before ingestion is frozen.\n'
                     'two software engineers and one part-time SRE engineer.\n'
                     'Target: inspect the dashboard within 5 minutes under a simulated failure.\n')
        self.record = extract_document('demo', self.text, use_model=False)
        self.plan = generate_plan(self.record, source_text=self.text)
        self.facts = extract_facts('demo', self.text)
        items = [self.source_item('The dashboard must reject viewer writes.', 'review:R1', ['requirements']),
                 self.source_item('Tests confirm viewer writes are denied.', 'review:A1', ['acceptance']),
                 self.source_item('Records must be retained for 30 days.', 'review:R2', ['requirements']),
                 self.source_item('Schema approval is required before ingestion is frozen.', 'review:D1',
                                  ['dependency_constraints'])]
        # T002 deliberately maps by source evidence to the FIRST displayed step.
        task_evidence = self.plan['steps'][0]['evidence_ids']
        return {'schema_version': 'psai-review-context-v1', 'document_id': 'demo',
                'source_sha256': self.record['source_sha256'], 'evidence': items,
                'task_links': [{'task_id': 'T002', 'task_evidence_ids': task_evidence,
                                'context_evidence_id': item['evidence_id'], 'categories': item['categories'],
                                'status': 'lexical_candidate', 'needs_review': True, 'shared_terms': ['viewer']}
                               for item in items[:2]],
                'review_questions': [{'kind': 'dependency_mapping', 'evidence_ids': ['review:D1'],
                                      'message': 'Confirm which approval gate constrains ingestion.'}],
                'omitted_counts': {}}

    def test_review_context_links_requirements_acceptance_and_unmatched_items_with_source_refs(self):
        context = self.use_review_source()
        result = self.render(review_context=context)
        self.assertIn('Gợi ý liên kết yêu cầu – công việc – nghiệm thu', result)
        self.assertIn('chưa xác nhận công việc xử lý đúng yêu cầu', result)
        self.assertIn('Yêu cầu chưa có liên kết gợi ý với công việc đã chọn (1 đoạn)', result)
        self.assertIn('Tiêu chí nghiệm thu chưa có liên kết gợi ý với công việc đã chọn (0 đoạn)', result)
        self.assertIn('Records must be retained for 30 days.', result)
        self.assertIn('Ràng buộc phụ thuộc được đề cập trong nguồn', result)
        self.assertIn('Chưa tự chuyển thành quan hệ kết thúc–bắt đầu', result)
        self.assertIn('Confirm which approval gate constrains ingestion.', result)
        review_table = result.split('Gợi ý liên kết yêu cầu – công việc – nghiệm thu', 1)[1]
        first_row = review_table.split('<tbody>', 1)[1].split('</tr>', 1)[0]
        self.assertIn('<strong>S001</strong>', first_row)
        self.assertIn('viewer writes', first_row)
        self.assertNotIn('<strong>T002</strong>', first_row)
        anchors = Anchors()
        anchors.feed(result)
        self.assertEqual(set(anchors.links) - anchors.targets, set())

    def test_review_source_span_tampering_is_rejected(self):
        context = self.use_review_source()
        context['evidence'][0]['quote'] += ' changed'
        with self.assertRaisesRegex(ValueError, 'differs from its source'):
            self.render(review_context=context)

    def test_personnel_workload_and_measures_keep_mention_and_target_status(self):
        self.use_review_source()
        facts = deepcopy(self.facts)
        facts['fields']['personnel'] = [self.source_item('one part-time SRE engineer', 'fact:P1',
            value={'quantity': 1, 'role_mention': 'SRE engineer', 'workload_mention': 'part-time',
                   'status': 'source_mention'})]
        facts['fields']['measures'] = [self.source_item(
            'Target: inspect the dashboard within 5 minutes under a simulated failure.', 'fact:M1',
            value={'quantity': '5', 'unit': 'minute', 'comparator': 'at_most', 'status': 'proposed_target'})]
        result = self.render(facts=facts)
        self.assertIn('Nhận diện: 1 SRE engineer · part-time', result)
        self.assertIn('không quy đổi thành người toàn thời gian', result)
        self.assertIn('Mục tiêu dự kiến; chưa kiểm chứng', result)
        self.assertIn('≤ 5 phút', result)
        self.assertIn('không phải phép đo do lượt phân tích này thực hiện', result)
        self.assertIn('số đoạn trích không phải số sản phẩm', result)

    def test_unselected_task_description_is_shown_and_escaped(self):
        step = self.plan['steps'][0]
        plan = {**self.plan, 'catalog': {'tasks': [{'task_id': 'T012',
                'description': 'Review <script>unsafe()</script> before release.',
                'evidence_ids': step['evidence_ids']}]}, 'unselected_task_ids': ['T012']}
        result = self.render(plan=plan)
        self.assertIn('Review &lt;script&gt;unsafe()&lt;/script&gt; before release.', result)
        self.assertNotIn('<script>unsafe()', result)


if __name__ == '__main__':
    unittest.main()
