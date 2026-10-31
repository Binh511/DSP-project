"""Source/provenance and ambiguity tests without ML or external resources."""
from copy import deepcopy
import hashlib
import unittest

from psai_rules import extract_facts, resource_warnings


class FactTests(unittest.TestCase):
    def test_all_detected_spans_exact_unicode_and_stable_ids(self):
        text = "Đề xuất 📄: 2027-03-14; trong 2 tuần với 3 kỹ sư. Ngân sách 1500 VND. Python."
        result = extract_facts("demo", text)
        self.assertEqual(result, extract_facts("demo", text))
        self.assertEqual(result["source_sha256"], hashlib.sha256(text.encode()).hexdigest())
        for entries in result["fields"].values():
            for fact in entries:
                self.assertEqual(text[fact["char_start"]:fact["char_end"]], fact["quote"])
                self.assertEqual(fact["source_sha256"], result["source_sha256"])
                self.assertTrue(fact["verified"])
                self.assertIn("not_semantic", fact["confidence_kind"])
        self.assertTrue(result["fields"]["dates"])
        self.assertTrue(result["fields"]["durations"])
        self.assertTrue(result["fields"]["budgets"])
        self.assertTrue(result["fields"]["personnel"])
        self.assertTrue(result["fields"]["technologies"])

    def test_empty_and_invalid_inputs(self):
        for document_id, source in (("", "text"), (None, "text"), ("a", ""), ("a", " \n"), ("a", None)):
            with self.subTest(document_id=document_id, source=source), self.assertRaises(ValueError):
                extract_facts(document_id, source)

    def test_iso_calendar_validated_no_timezone_guessed(self):
        values = extract_facts("a", "Due 2028-02-29, invalid 2027-02-29 and 2027-13-01.")["fields"]["dates"]
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0]["value"]["iso_date"], "2028-02-29")
        self.assertIsNone(values[0]["value"]["timezone"])

    def test_slash_ambiguous_dates_marked_not_guessed(self):
        values = extract_facts("a", "03/04/2027; 14/03/2027; 04/04/2027; invalid 31/02/2027.")["fields"]["dates"]
        self.assertEqual(len(values), 3)
        self.assertTrue(values[0]["value"]["ambiguous"])
        self.assertTrue(values[0]["needs_review"])
        self.assertIsNone(values[0]["value"]["iso_date"])
        self.assertEqual(set(values[0]["value"]["candidate_iso_dates"]), {"2027-04-03", "2027-03-04"})
        self.assertEqual(values[1]["value"]["iso_date"], "2027-03-14")
        self.assertFalse(values[2]["value"]["ambiguous"])

    def test_currencies_only_with_explicit_currency_no_year_or_plain_id_false_positive(self):
        text = "Year 2027. Project ID: ABC-2027. There are 300 documents and version 2.5. Budget 1500 USD."
        budgets = extract_facts("a", text)["fields"]["budgets"]
        self.assertEqual(len(budgets), 1)
        self.assertEqual(budgets[0]["quote"], "1500 USD")
        self.assertEqual(budgets[0]["value"]["amount"], "1500")

    def test_currency_punctuation_and_decimal_precision(self):
        budgets = extract_facts("a", "Costs USD 1234.50. Also 45,75 EUR; 100 VND.")["fields"]["budgets"]
        self.assertEqual(len(budgets), 3)
        self.assertEqual([item["value"]["amount"] for item in budgets], ["1234.50", "45.75", "100"])
        self.assertEqual(budgets[2]["quote"], "100 VND")

    def test_thousands_separator_ambiguity_preserved(self):
        budgets = extract_facts("a", "USD 1,500; EUR 1.500; VND 1.500.000; USD 1,234.50.")["fields"]["budgets"]
        self.assertEqual(len(budgets), 4)
        for item in budgets[:2]:
            self.assertTrue(item["value"]["numeric_ambiguous"])
            self.assertIsNone(item["value"]["amount"])
            self.assertEqual(set(item["value"]["candidate_amounts"]), {"1500", "1.500"})
        self.assertEqual(budgets[2]["value"]["amount"], "1500000")
        self.assertEqual(budgets[3]["value"]["amount"], "1234.50")

    def test_dollar_country_not_assumed(self):
        budget = extract_facts("a", "Budget $500.")["fields"]["budgets"][0]
        self.assertEqual(budget["value"]["currency"], "UNSPECIFIED_DOLLAR")
        self.assertTrue(budget["value"]["currency_ambiguous"])
        self.assertTrue(budget["needs_review"])

    def test_unsupported_scales_ranges_negative_and_malformed_amounts_not_partial(self):
        for text in ("USD 2 million", "$5 billion", "USD 100-200", "-$500", "USD 1.234.56", "USD 1,23,45"):
            with self.subTest(text=text):
                self.assertFalse(extract_facts("a", text)["fields"]["budgets"])

    def test_duration_units_no_date_conversion(self):
        result = extract_facts("a", "The work takes 2.5 weeks, 3 ngày or 12 hours, not 2027-04-02.")
        values = [item["value"] for item in result["fields"]["durations"]]
        self.assertEqual(values, [{"quantity": "2.5", "unit": "week"}, {"quantity": "3", "unit": "day"},
                                  {"quantity": "12", "unit": "hour"}])
        self.assertFalse(any("start" in value or "end" in value for value in values))

    def test_identifier_requires_label_preserves_literal_value(self):
        result = extract_facts("a", "The year 2027 has ABC-9. Project ID: PRO-123. Mã dự án: DSP-2027.")
        self.assertEqual([item["value"] for item in result["fields"]["identifiers"]], ["PRO-123", "DSP-2027"])

    def test_explicit_milestone_and_known_technology_only(self):
        result = extract_facts("a", "Milestone 1: Deliver the collector\nCột mốc 2: Kiểm thử dữ liệu\nUse Python and Docker; Pythonic is not Python.")
        self.assertEqual(len(result["fields"]["milestones"]), 2)
        self.assertEqual([item["value"] for item in result["fields"]["technologies"]], ["Python", "Docker", "Python"])
        self.assertEqual(len({item["evidence_id"] for entries in result["fields"].values() for item in entries}),
                         sum(len(entries) for entries in result["fields"].values()))

    def test_personnel_mentions_are_not_assignments(self):
        people = extract_facts("a", "Need 2 developers and 3 kỹ sư.")["fields"]["personnel"]
        self.assertEqual([item["value"]["quantity"] for item in people], [2, 3])
        self.assertTrue(all(item["needs_review"] for item in people))
        self.assertTrue(all("not_assignment" in item["normalization_policy"] for item in people))

    def test_repeated_literal_at_distinct_source_locations_kept_once_each(self):
        result = extract_facts("a", "Python Python USD 12 USD 12")
        for field in ("technologies", "budgets"):
            items = result["fields"][field]
            self.assertEqual(len(items), 2)
            self.assertNotEqual(items[0]["evidence_id"], items[1]["evidence_id"])
            self.assertEqual(len({(item["char_start"], item["char_end"]) for item in items}), 2)

    def test_not_detected_warning_does_not_claim_absence(self):
        facts = extract_facts("a", "A project description without matching rules.")
        common = {"document_id": "a", "source_sha256": facts["source_sha256"]}
        warnings = resource_warnings(common, {**common, "steps": []}, facts)
        self.assertEqual(len(warnings), 4)
        self.assertTrue(all(item["status"] == "not_detected" for item in warnings))
        self.assertTrue(all("not proof" in item["message"] for item in warnings))

    def test_detected_mentions_and_llm_associations_still_need_review(self):
        facts = extract_facts("a", "Python needs 2 developers, USD 50 by 2027-01-05.")
        common = {"document_id": "a", "source_sha256": facts["source_sha256"]}
        plan = {**common, "steps": [{"required_resources": [{"evidence_ids": ["a:S1"]}]}],
                "dependency_candidates": [{"before_task_id": "T1", "after_task_id": "T2"}]}
        warnings = resource_warnings(common, plan, facts)
        self.assertEqual(len(warnings), 6)
        self.assertTrue(all(item["status"] == "needs_review" for item in warnings))
        self.assertIn("resource_associations_need_review", [item["code"] for item in warnings])

    def test_warning_source_hash_mismatch_rejected(self):
        facts = extract_facts("a", "Python")
        common = {"document_id": "a", "source_sha256": facts["source_sha256"]}
        with self.assertRaises(ValueError):
            resource_warnings(common, {**common, "source_sha256": "0" * 64}, facts)


if __name__ == "__main__":
    unittest.main()
