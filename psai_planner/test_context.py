"""Source review and v2 protocol regressions without ML dependencies."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from psai_planner import (PlannerValidationError, build_catalog, build_prompt,
                          build_review_context, validate_response, validate_review_context)
from psai_planner.planner import LEGACY_PROTOCOL_VERSION, PROTOCOL_VERSION


def fixture():
    text = ("# University records recovery\n"
            "## Implementation\n"
            "We will implement encrypted backup and restore verification.\n"
            "We will build inventory validation for imported records.\n"
            "## Requirements\n"
            "The application must retain records for 30 days.\n"
            "## Acceptance criteria\n"
            "Acceptance checks require restoring a test backup and matching inventory records.\n"
            "The proposed acceptance target is 95 percent of expected records; no check has passed.\n"
            "## Resources and constraints\n"
            "The team consists of two software engineers and one part-time reviewer.\n"
            "Individual task ownership is not confirmed.\n"
            "Inventory validation must precede final restore approval.\n"
            "## Risks\n"
            "Risk R1 is a disclosure of protected records.\n")
    tasks = []
    for index, line in enumerate(text.splitlines()):
        if line.startswith("We will"):
            start = text.index(line)
            tasks.append({"evidence_id": f"study:S{index}", "quote": line, "value": line,
                          "char_start": start, "char_end": start + len(line),
                          "verified": True, "label_score": .95})
    record = {"document_id": "study", "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
              "fields": {"activities": tasks, "requirements": [], "resources": []}}
    return record, text


def selection():
    return {"tasks": [{"task_id": f"T{index:03d}", "parent_task_id": None,
                       "checklist": [], "required_resources": []} for index in (1, 2)],
            "dependency_candidates": [], "resource_references": []}


class ReviewContextTests(unittest.TestCase):
    def setUp(self):
        self.record, self.text = fixture()
        self.context = build_review_context(self.record, self.text)

    def test_source_context_survives_classifier_omissions(self):
        categories = {category for item in self.context["evidence"] for category in item["categories"]}
        self.assertEqual(categories, {"requirements", "acceptance", "dependency_constraints", "risks",
                                      "personnel", "measures", "open_questions"})
        self.assertTrue(any("30 days" in item["quote"] for item in self.context["evidence"]))
        self.assertTrue(any("part-time reviewer" in item["quote"] for item in self.context["evidence"]))
        self.assertFalse(self.record["fields"]["requirements"])

    def test_all_quotes_rechecked_and_ids_stable(self):
        self.assertEqual(self.context, build_review_context(deepcopy(self.record), self.text))
        validate_review_context(self.context, self.record, self.text)
        for item in self.context["evidence"]:
            self.assertEqual(self.text[item["char_start"]:item["char_end"]], item["quote"])
            self.assertEqual(item["source_sha256"], self.record["source_sha256"])
            self.assertTrue(item["needs_review"])
        for field, value in (("human_review_complete", True), ("coverage_status", "complete")):
            edited = deepcopy(self.context)
            edited[field] = value
            with self.subTest(field=field), self.assertRaises(PlannerValidationError):
                validate_review_context(edited, self.record, self.text)
        edited = deepcopy(self.context)
        edited["evidence"][0]["quote"] = "invented requirement"
        with self.assertRaises(PlannerValidationError):
            validate_review_context(edited, self.record, self.text)
        with self.assertRaises(PlannerValidationError):
            build_review_context(self.record, self.text + "changed")

    def test_links_are_unreviewed_cross_sentence_candidates_not_coverage(self):
        acceptance_ids = {item["evidence_id"] for item in self.context["evidence"]
                          if "restoring a test backup" in item["quote"]}
        self.assertTrue(any(link["task_id"] == "T001" and link["context_evidence_id"] in acceptance_ids
                            for link in self.context["task_links"]))
        for link in self.context["task_links"]:
            self.assertEqual(link["status"], "lexical_candidate")
            self.assertTrue(link["needs_review"])
            self.assertGreaterEqual(len(link["shared_terms"]), 2)
        self.assertEqual(self.context["coverage_status"], "not_assessed")
        self.assertTrue(self.context["unlinked_evidence_ids"])
        self.assertTrue(any(question["kind"] == "dependency_constraints"
                            for question in self.context["review_questions"]))

    def test_sentence_segmentation_preserves_negation_and_approval_qualifiers(self):
        dep = next(item for item in self.context["evidence"] if "precede final" in item["quote"])
        self.assertEqual(dep["quote"], "Inventory validation must precede final restore approval.")
        self.assertNotIn("dependency_candidates", self.context)
        self.assertTrue(any("no check has passed" in item["quote"] for item in self.context["evidence"]))

    def test_generic_examples_include_missed_retention_and_measurements(self):
        path = Path(__file__).resolve().parents[1] / "examples/PSAI_DEMO_01_MINER_MONITORING.md"
        text = path.read_text(encoding="utf-8-sig")
        record = {"document_id": "miner", "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "fields": {"activities": []}}
        context = build_review_context(record, text)
        quotes = "\n".join(item["quote"] for item in context["evidence"])
        for phrase in ("30-day", "five minutes", "95 percent", "two software engineers", "schema approval"):
            self.assertIn(phrase, quotes)
        self.assertFalse(context["task_links"])
        self.assertFalse(context["human_review_complete"])

    def test_document_retrieval_requires_actual_main_source_binding(self):
        quote = "Inventory validation must precede final restore approval."
        start = self.text.index(quote)
        hit = {"role": "document", "doc_id": "study", "evidence_id": "study:R1", "verified": True,
               "source_sha256": self.record["source_sha256"], "quote": quote,
               "char_start": start, "char_end": start + len(quote)}
        self.assertEqual(build_review_context(self.record, self.text, {"hits": [hit]})["retrieval_context"][0]["quote"], quote)
        for key, value in (("source_sha256", "0" * 64), ("doc_id", "other"), ("quote", "fabricated"),
                           ("verified", False), ("char_start", True)):
            bad = {**hit, key: value}
            with self.subTest(key=key), self.assertRaises(PlannerValidationError):
                build_review_context(self.record, self.text, {"hits": [bad]})
        with self.assertRaises(PlannerValidationError):
            build_review_context(self.record, self.text, {"hits": [hit, hit]})

    def test_staff_usage_and_node_availability_do_not_become_staffing_resources(self):
        text = ("The objective is to give operations staff a view of node availability.\n"
                "All staffing and costs in this scenario are hypothetical.\n"
                "The team consists of two engineers and one part-time reviewer.\n")
        record = {"document_id": "university", "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "fields": {"activities": []}}
        context = build_review_context(record, text)
        people = [item for item in context["evidence"] if "personnel" in item["categories"]]
        self.assertEqual(len(people), 1)
        self.assertIn("two engineers", people[0]["quote"])


class PlannerV2Tests(unittest.TestCase):
    def setUp(self):
        self.record, self.text = fixture()
        self.catalog = build_catalog(self.record, self.text, config={"response_format": "expanded_v1"})

    def test_v2_cross_sentence_checklist_is_exact_cited_proposal(self):
        acceptance = next(item for item in self.catalog["context_evidence"] if "restoring a test backup" in item["quote"])
        response = selection()
        response["tasks"][0]["checklist"] = [{"evidence_id": acceptance["evidence_id"], "text": "matching inventory records"}]
        plan = validate_response(json.dumps(response), self.catalog)
        item = plan["steps"][0]["checklist"][0]
        self.assertEqual(self.text[item["char_start"]:item["char_end"]], item["description"])
        self.assertTrue(item["needs_review"])
        self.assertEqual(plan["planning_method"], "local_llm_evidence_catalog_v2")
        response["tasks"][0]["checklist"][0]["text"] = "all inventory checks have passed"
        with self.assertRaises(PlannerValidationError):
            validate_response(json.dumps(response), self.catalog)

    def test_dependency_needs_constraint_citation_and_never_approved(self):
        dependency = next(item for item in self.catalog["context_evidence"] if "dependency_constraints" in item["categories"])
        response = selection()
        edge = {"before_task_id": "T002", "after_task_id": "T001", "status": "suggested",
                "needs_review": True, "evidence_ids": [dependency["evidence_id"]]}
        response["dependency_candidates"] = [edge]
        plan = validate_response(json.dumps(response), self.catalog)
        self.assertEqual(plan["dependency_candidates"][0]["basis"], "source_cited_llm_candidate")
        for modified in ({**edge, "evidence_ids": []}, {**edge, "evidence_ids": ["unknown"]},
                         {**edge, "needs_review": False}, {**edge, "status": "approved"}):
            response["dependency_candidates"] = [modified]
            with self.subTest(modified=modified), self.assertRaises(PlannerValidationError):
                validate_response(json.dumps(response), self.catalog)

    def test_prompt_compacts_duplicated_catalog_fields_and_reports_omissions(self):
        messages = build_prompt(self.catalog)
        payload = json.loads(messages[1]["content"].split("\n")[1])
        self.assertNotIn("char_start", payload)
        self.assertIn("evidence", payload)
        limited = build_catalog(self.record, self.text, config={"max_context_items": 1})
        self.assertEqual(len(limited["context_evidence"]), 1)
        self.assertGreater(limited["catalog_omitted_counts"]["context_evidence"], 0)
        self.assertIn("omitted_counts", build_prompt(limited)[1]["content"])

    def test_document_retrieval_actually_enters_prompt(self):
        quote = "We will implement encrypted backup and restore verification."
        start = self.text.index(quote)
        hit = {"role": "document", "doc_id": "study", "evidence_id": "study:R1", "verified": True,
               "source_sha256": self.record["source_sha256"], "quote": quote,
               "char_start": start, "char_end": start + len(quote)}
        catalog = build_catalog(self.record, self.text, {"hits": [hit]}, {"response_format": "expanded_v1"})
        self.assertEqual(len(catalog["document_references"]), 1)
        self.assertIn(catalog["document_references"][0]["evidence_id"], build_prompt(catalog)[1]["content"])

    def test_legacy_protocol_replays_old_shape_and_restricts_checklist(self):
        legacy = build_catalog(self.record, self.text, config={"protocol_version": LEGACY_PROTOCOL_VERSION})
        self.assertNotIn("protocol_version", legacy)
        self.assertNotIn("context_evidence", legacy)
        self.assertIn("that task's source quote", build_prompt(legacy)[0]["content"])
        plan = validate_response(json.dumps(selection()), legacy)
        self.assertEqual(plan["planning_method"], "local_llm_evidence_catalog_v1")
        response = selection()
        acceptance = next(item for item in self.catalog["context_evidence"] if "restoring a test backup" in item["quote"])
        response["tasks"][0]["checklist"] = [{"evidence_id": acceptance["evidence_id"], "text": "matching inventory records"}]
        with self.assertRaises(PlannerValidationError):
            validate_response(json.dumps(response), legacy)
        with self.assertRaises(PlannerValidationError):
            build_catalog(self.record, self.text, config={"protocol_version": "invented"})

    def test_source_namespace_collision_rejected(self):
        record = deepcopy(self.record)
        record["fields"]["activities"][0]["evidence_id"] = self.catalog["context_evidence"][0]["evidence_id"]
        with self.assertRaises(PlannerValidationError):
            build_catalog(record, self.text)


class CompactProtocolTests(unittest.TestCase):
    def setUp(self):
        self.record, self.text = fixture()
        self.catalog = build_catalog(self.record, self.text, config={"response_format": "links_v1"})

    def response(self):
        acceptance = next(item for item in self.catalog["context_evidence"] if "restoring a test backup" in item["quote"])
        resource = next(item for item in self.catalog["project_resources"] if "two software engineers" in item["quote"])
        constraint = next(item for item in self.catalog["context_evidence"] if "precede final" in item["quote"])
        return {"task_ids": ["T001", "T002"],
                "checklist_links": [{"task_id": "T001", "evidence_id": acceptance["evidence_id"]}],
                "resource_links": [{"task_id": "T001", "evidence_id": resource["evidence_id"]}],
                "dependency_candidates": [{"before_task_id": "T002", "after_task_id": "T001",
                                           "evidence_ids": [constraint["evidence_id"]]}],
                "resource_references": []}

    def test_compact_ids_resolve_to_literal_quotes_and_unreviewed_status(self):
        response = self.response()
        result = validate_response(json.dumps(response), self.catalog)
        self.assertEqual(self.catalog["response_format"], "links_v1")
        self.assertEqual([step["task_id"] for step in result["steps"]], response["task_ids"])
        self.assertTrue(all(step["parent_task_id"] is None for step in result["steps"]))
        for key in ("checklist", "required_resources"):
            item = result["steps"][0][key][0]
            self.assertEqual(self.text[item["char_start"]:item["char_end"]], item["description"])
            self.assertTrue(item["needs_review"])
        self.assertEqual(result["dependency_candidates"][0]["status"], "suggested")
        self.assertFalse(result["human_review_complete"])
        self.assertIn("IDs only", build_prompt(self.catalog)[0]["content"])

    def test_compact_unknown_ids_wrong_uses_freetext_and_hierarchy_rejected(self):
        wrong_use = next(item for item in self.catalog["context_evidence"] if item["categories"] == ["risks"])
        cases = []
        value = self.response(); value["task_ids"] = ["T999"]; cases.append(value)
        value = self.response(); value["checklist_links"][0]["evidence_id"] = "not-known"; cases.append(value)
        value = self.response(); value["checklist_links"][0]["evidence_id"] = wrong_use["evidence_id"]; cases.append(value)
        value = self.response(); value["resource_links"][0]["text"] = "assign Alice full-time"; cases.append(value)
        value = self.response(); value["parent_task_id"] = "T001"; cases.append(value)
        value = self.response(); value["dependency_candidates"][0]["before_task_id"] = "T001"; cases.append(value)
        value = self.response(); value["dependency_candidates"][0]["status"] = "approved"; cases.append(value)
        for index, value in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(PlannerValidationError):
                validate_response(json.dumps(value), self.catalog)

    def test_compact_abstention_is_valid_without_fabricated_links(self):
        value = {"task_ids": [], "checklist_links": [], "resource_links": [],
                 "dependency_candidates": [], "resource_references": []}
        result = validate_response(json.dumps(value), self.catalog)
        self.assertEqual(result["status"], "abstained")
        value["checklist_links"] = self.response()["checklist_links"]
        with self.assertRaises(PlannerValidationError):
            validate_response(json.dumps(value), self.catalog)

    def test_document_dependency_citation_works_when_context_budget_excludes_it(self):
        quote = "Inventory validation must precede final restore approval."
        start = self.text.index(quote)
        hit = {"role": "document", "doc_id": "study", "evidence_id": "study:R1", "verified": True,
               "source_sha256": self.record["source_sha256"], "quote": quote,
               "char_start": start, "char_end": start + len(quote)}
        catalog = build_catalog(self.record, self.text, {"hits": [hit]}, {"max_context_items": 1, "response_format": "links_v1"})
        evidence = catalog["document_references"][0]
        self.assertIn("dependency_constraints", evidence["categories"])
        response = {"task_ids": ["T001", "T002"], "checklist_links": [], "resource_links": [],
                    "dependency_candidates": [{"before_task_id": "T002", "after_task_id": "T001",
                                               "evidence_ids": [evidence["evidence_id"]]}],
                    "resource_references": []}
        result = validate_response(json.dumps(response), catalog)
        self.assertEqual(result["dependency_candidates"][0]["evidence_ids"], [evidence["evidence_id"]])


class CandidateIDProtocolTests(unittest.TestCase):
    def setUp(self):
        self.record, self.text = fixture()
        self.catalog = build_catalog(self.record, self.text)

    def test_known_candidate_hydrates_cross_sentence_quote_without_free_recombination(self):
        candidate = next(item for item in self.catalog["link_candidates"]
                         if item["task_id"] == "T001" and item["kind"] == "checklist")
        raw = json.dumps({"task_ids": ["T001"], "link_ids": [candidate["link_id"]]})
        plan = validate_response(raw, self.catalog)
        entry = plan["steps"][0]["checklist"][0]
        self.assertEqual(entry["evidence_ids"], [candidate["evidence_id"]])
        self.assertEqual(self.text[entry["char_start"]:entry["char_end"]], entry["description"])
        self.assertTrue(entry["needs_review"])
        self.assertFalse(plan["human_review_complete"])
        self.assertEqual(self.catalog["response_format"], "candidate_ids_v1")

    def test_unknown_source_ids_unselected_tasks_duplicates_and_extra_fields_rejected(self):
        candidate = self.catalog["link_candidates"][0]
        cases = [{"task_ids": ["T001", "T002"], "link_ids": ["L999"]},
                 {"task_ids": ["T001", "T002"], "link_ids": [self.record["fields"]["activities"][0]["evidence_id"]]},
                 {"task_ids": [], "link_ids": [candidate["link_id"]]},
                 {"task_ids": ["T001", "T002"], "link_ids": [candidate["link_id"], candidate["link_id"]]},
                 {"task_ids": ["T001", "T002"], "link_ids": [], "checklist_links": []}]
        for index, response in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(PlannerValidationError):
                validate_response(json.dumps(response), self.catalog)

    def test_omitted_link_ids_abstains_explicitly_without_filling_candidates(self):
        result = validate_response('{"task_ids":["T001","T002"]}', self.catalog)
        self.assertEqual(len(result["steps"]), 2)
        self.assertTrue(self.catalog["link_candidates"])
        for step in result["steps"]:
            self.assertEqual(step["checklist"], [])
            self.assertEqual(step["required_resources"], [])
        self.assertEqual(result["dependency_candidates"], [])
        self.assertIn("Model omitted link_ids; no link was inferred", result["warnings"])
        explicit_empty = validate_response('{"task_ids":["T001"],"link_ids":[]}', self.catalog)
        self.assertNotIn("Model omitted link_ids; no link was inferred", explicit_empty["warnings"])
        for raw in ('{"link_ids":[]}', '{"task_ids":["T001"],"unknown":[]}',
                    '{"task_ids":["T001"],"link_ids":null}'):
            with self.subTest(raw=raw), self.assertRaises(PlannerValidationError):
                validate_response(raw, self.catalog)

    def test_candidates_exclude_own_source_and_keep_explicit_limits(self):
        task_map = {item["task_id"]: item for item in self.catalog["tasks"]}
        evidence = {item["evidence_id"]: item for item in self.catalog["context_evidence"]
                    + self.catalog["document_references"] + self.catalog["project_resources"]}
        for candidate in self.catalog["link_candidates"]:
            task, item = task_map[candidate["task_id"]], evidence[candidate["evidence_id"]]
            self.assertGreaterEqual(len(candidate["shared_terms"]), 2)
            self.assertFalse(max(task["char_start"], item["char_start"]) < min(task["char_end"], item["char_end"]))
        limited = build_catalog(self.record, self.text, config={"max_link_candidates": 1})
        self.assertEqual(len(limited["link_candidates"]), 1)
        self.assertGreater(limited["catalog_omitted_counts"]["link_candidates"], 0)
        payload = json.loads(build_prompt(self.catalog)[1]["content"].split("\n")[1])
        self.assertTrue(all(len(row) == 2 for row in payload["tasks"]))
        self.assertIn("link_candidates", payload)
        abstained = validate_response('{"task_ids":[],"link_ids":[]}', self.catalog)
        self.assertEqual(abstained["status"], "abstained")

    def test_default_document_requirement_candidate_enters_prompt_when_not_in_context(self):
        extra = "The encrypted backup must include an inventory record."
        text = self.text + extra + "\n"
        record = deepcopy(self.record)
        record["source_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        start = text.index(extra)
        hit = {"role": "document", "doc_id": "study", "evidence_id": "study:R2", "verified": True,
               "source_sha256": record["source_sha256"], "quote": extra,
               "char_start": start, "char_end": start + len(extra)}
        catalog = build_catalog(record, text, {"hits": [hit]}, {"max_context_items": 1})
        document_id = catalog["document_references"][0]["evidence_id"]
        candidate = next(item for item in catalog["link_candidates"]
                         if item["evidence_id"] == document_id and item["task_id"] == "T001")
        self.assertIn(document_id, build_prompt(catalog)[1]["content"])
        result = validate_response(json.dumps({"task_ids": ["T001"], "link_ids": [candidate["link_id"]]}), catalog)
        self.assertEqual(result["steps"][0]["checklist"][0]["description"], extra)


class PreserveSourceCatalogTests(unittest.TestCase):
    def setUp(self):
        self.record, self.text = fixture()
        self.catalog = build_catalog(self.record, self.text, config={"task_policy": "preserve_source_catalog"})

    def test_subset_is_ordering_hint_and_remaining_source_task_is_preserved(self):
        result = validate_response('{"task_ids":["T002"],"link_ids":[]}', self.catalog)
        self.assertEqual([step["task_id"] for step in result["steps"]], ["T002", "T001"])
        self.assertEqual([step["selection_origin"] for step in result["steps"]],
                         ["llm_selection", "source_catalog_preserved"])
        self.assertEqual(result["source_catalog_preserved_task_ids"], ["T001"])
        self.assertEqual(result["llm_task_selection_status"], "ordering_hints")
        self.assertTrue(any("Preserved 1 source catalog" in warning for warning in result["warnings"]))
        self.assertFalse(result["human_review_complete"])
        self.assertIn("ordering hints", build_prompt(self.catalog)[0]["content"])

    def test_empty_model_selection_preserves_only_bounded_catalog_and_marks_abstention(self):
        catalog = build_catalog(self.record, self.text, config={"task_policy": "preserve_source_catalog", "max_tasks": 1})
        result = validate_response('{"task_ids":[],"link_ids":[]}', catalog)
        self.assertEqual([step["task_id"] for step in result["steps"]], ["T001"])
        self.assertEqual(result["steps"][0]["selection_origin"], "source_catalog_preserved")
        self.assertEqual(result["llm_task_selection_status"], "abstained")
        self.assertEqual(result["catalog_omitted_counts"]["tasks"], 1)
        self.assertEqual(result["status"], "validated_candidate")

    def test_link_to_preserved_task_is_allowed_without_inventing_model_selection(self):
        candidate = next(item for item in self.catalog["link_candidates"] if item["task_id"] == "T001")
        result = validate_response(json.dumps({"task_ids": ["T002"], "link_ids": [candidate["link_id"]]}), self.catalog)
        task = next(step for step in result["steps"] if step["task_id"] == "T001")
        self.assertTrue(task["checklist"])
        self.assertEqual(task["selection_origin"], "source_catalog_preserved")
        with self.assertRaises(PlannerValidationError):
            validate_response('{"task_ids":["T999"],"link_ids":[]}', self.catalog)

    def test_select_only_and_legacy_semantics_stay_unchanged(self):
        catalog = build_catalog(self.record, self.text)
        result = validate_response('{"task_ids":["T002"],"link_ids":[]}', catalog)
        self.assertEqual([step["task_id"] for step in result["steps"]], ["T002"])
        self.assertNotIn("selection_origin", result["steps"][0])
        self.assertNotIn("task_policy", result)
        with self.assertRaises(PlannerValidationError):
            build_catalog(self.record, self.text, config={"task_policy": "preserve_source_catalog", "response_format": "expanded_v1"})


if __name__ == "__main__":
    unittest.main()
