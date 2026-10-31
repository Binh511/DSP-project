"""Protocol tests: no model downloads, GPU, torch or transformers required."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from psai_planner import (PlannerValidationError, build_catalog, build_prompt,
                          plan_project, validate_response)


def fixture():
    lines = ["We will implement the source citation validation module.",
             "We will create a report with source evidence links.",
             "The project requires Python and two developers."]
    text = "\n".join(lines)
    items = []
    for index, quote in enumerate(lines):
        start = text.index(quote)
        items.append({"evidence_id": f"demo:S{index:05d}", "quote": quote, "value": quote,
                      "char_start": start, "char_end": start + len(quote),
                      "verified": True, "label_score": .95})
    record = {"document_id": "demo", "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
              "fields": {"activities": items[:2], "resources": items[2:],
                         "technologies": [deepcopy(items[2])]}}
    return record, text


def selection():
    return {"tasks": [{"task_id": "T001", "parent_task_id": None, "checklist": [], "required_resources": []},
                      {"task_id": "T002", "parent_task_id": None, "checklist": [], "required_resources": []}],
            "dependency_candidates": [], "resource_references": []}


class PlannerProtocolTests(unittest.TestCase):
    def setUp(self):
        self.record, self.text = fixture()
        self.catalog = build_catalog(self.record, self.text)

    def validate(self, response):
        return validate_response(json.dumps(response), self.catalog)

    def test_module_import_never_imports_ml_libraries(self):
        result = subprocess.run([sys.executable, "-c", "import sys,psai_planner; "
                                 "assert not any(name in sys.modules for name in "
                                 "('torch','transformers','sentence_transformers','peft'))"],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_known_tasks_rehydrated_from_source_and_no_fake_entailment(self):
        result = self.validate(selection())
        self.assertEqual(result["status"], "validated_candidate")
        self.assertEqual(result["steps"][0]["description"], self.record["fields"]["activities"][0]["value"])
        self.assertTrue(result["steps"][0]["needs_review"])
        self.assertFalse(result["human_review_complete"])
        self.assertTrue(any("not semantic" in warning for warning in result["warnings"]))

    def test_checklist_exact_span_and_resource_association_require_review(self):
        response = selection()
        response["tasks"][0]["checklist"] = [{"evidence_id": "demo:S00000", "text": "source citation validation module"}]
        response["tasks"][0]["required_resources"] = [{"evidence_id": "demo:S00002", "text": "Python and two developers"}]
        result = self.validate(response)
        for key in ("checklist", "required_resources"):
            item = result["steps"][0][key][0]
            self.assertEqual(self.text[item["char_start"]:item["char_end"]], item["description"])
            self.assertTrue(item["needs_review"])

    def test_source_hash_and_span_rechecked_not_just_verified_flag(self):
        for modification in ("hash", "span", "quote", "value", "confidence"):
            record = deepcopy(self.record)
            item = record["fields"]["activities"][0]
            if modification == "hash":
                record["source_sha256"] = "0" * 64
            elif modification == "span":
                item["char_start"] += 1
            elif modification == "quote":
                item["quote"] += "invented"
            elif modification == "value":
                item["value"] = "Finish by 2027 and spend $900"
            else:
                item["label_score"] = float("nan")
            with self.subTest(modification=modification), self.assertRaises(PlannerValidationError):
                build_catalog(record, self.text)

    def test_repeated_evidence_id_conflicting_span_rejected(self):
        self.record["fields"]["activities"][1]["evidence_id"] = "demo:S00000"
        with self.assertRaises(PlannerValidationError):
            build_catalog(self.record, self.text)

    def test_catalog_candidate_caps_explicit(self):
        catalog = build_catalog(self.record, self.text, config={"max_tasks": 1})
        self.assertEqual(len(catalog["tasks"]), 1)
        self.assertEqual(catalog["catalog_omitted_counts"]["tasks"], 1)
        self.assertTrue(catalog["warnings"])

    def test_unknown_or_repeated_task_ids_rejected(self):
        for tid in ("T999", "T001", ["T001"]):
            response = selection()
            response["tasks"][1]["task_id"] = tid
            with self.subTest(tid=tid), self.assertRaises(PlannerValidationError):
                self.validate(response)

    def test_free_text_description_dates_budgets_duration_not_allowed(self):
        for key, value in (("description", "Build an unrelated payment system"),
                           ("due_date", "2027-01-01"), ("budget", "$5000"), ("duration", "2 weeks")):
            response = selection()
            response["tasks"][0][key] = value
            with self.subTest(key=key), self.assertRaises(PlannerValidationError):
                self.validate(response)

    def test_no_unknown_fields_at_top_level(self):
        response = selection()
        response["budget"] = "$5"
        with self.assertRaises(PlannerValidationError):
            self.validate(response)

    def test_invented_checklist_and_cross_task_citation_rejected(self):
        for eid, text in (("demo:S00000", "Create an unauthorized payment service"),
                          ("demo:S00001", "report with source evidence links"),
                          ("missing-id", "source citation validation module")):
            response = selection()
            response["tasks"][0]["checklist"] = [{"evidence_id": eid, "text": text}]
            with self.subTest(eid=eid, text=text), self.assertRaises(PlannerValidationError):
                self.validate(response)

    def test_hierarchy_cycle_and_unknown_parent_rejected(self):
        for cyclic in (True, False):
            response = selection()
            response["tasks"][0]["parent_task_id"] = "T002" if cyclic else "T999"
            if cyclic:
                response["tasks"][1]["parent_task_id"] = "T001"
            with self.subTest(cyclic=cyclic), self.assertRaises(PlannerValidationError):
                self.validate(response)

    def test_subtask_parent_supported(self):
        response = selection()
        response["tasks"][1]["parent_task_id"] = "T001"
        result = self.validate(response)
        self.assertEqual(result["steps"][1]["parent_task_id"], "T001")

    def test_dependencies_suggestions_only_and_cycles_rejected(self):
        forward = {"before_task_id": "T001", "after_task_id": "T002", "status": "suggested", "needs_review": True}
        response = selection()
        response["dependency_candidates"] = [forward]
        result = self.validate(response)
        self.assertEqual(result["dependency_candidates"][0]["basis"], "llm_inference_not_source_fact")
        self.assertEqual(result["dependency_candidates"][0]["evidence_ids"], [])
        for invalid in ({**forward, "status": "confirmed"}, {**forward, "needs_review": False},
                        {**forward, "after_task_id": "T001"}, {**forward, "after_task_id": "T999"}):
            response["dependency_candidates"] = [invalid]
            with self.subTest(invalid=invalid), self.assertRaises(PlannerValidationError):
                self.validate(response)
        response["dependency_candidates"] = [forward, {**forward, "before_task_id": "T002", "after_task_id": "T001"}]
        with self.assertRaises(PlannerValidationError):
            self.validate(response)

    def test_repeated_dependency_and_checklist_rejected(self):
        response = selection()
        entry = {"evidence_id": "demo:S00000", "text": "source citation validation module"}
        response["tasks"][0]["checklist"] = [entry, entry]
        with self.assertRaises(PlannerValidationError):
            self.validate(response)
        response = selection()
        edge = {"before_task_id": "T001", "after_task_id": "T002", "status": "suggested", "needs_review": True}
        response["dependency_candidates"] = [edge, edge]
        with self.assertRaises(PlannerValidationError):
            self.validate(response)

    def test_resource_must_be_project_resource_evidence(self):
        response = selection()
        response["tasks"][0]["required_resources"] = [{"evidence_id": "demo:S00001", "text": "report with source evidence links"}]
        with self.assertRaises(PlannerValidationError):
            self.validate(response)

    def test_external_resource_has_snapshot_validation_and_reference_only_role(self):
        source = "External tool documentation is reference material."
        external = {"evidence_id": "resource:tool:C001", "role": "resource", "verified": True,
                    "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "snapshot_id": "tool-snapshot",
                    "char_start": 0, "char_end": len(source), "quote": source, "citation_source": "retrieval_sources/tool.txt"}
        retrieval = {"hits": [external], "source_texts": {"tool-snapshot": source}}
        catalog = build_catalog(self.record, self.text, retrieval)
        response = selection()
        response["resource_references"] = [external["evidence_id"]]
        result = validate_response(json.dumps(response), catalog)
        self.assertEqual(result["resource_references"][0]["role"], "external_reference")
        self.assertTrue(result["resource_references"][0]["needs_review"])
        response["tasks"][0]["required_resources"] = [{"evidence_id": external["evidence_id"], "text": "External tool documentation"}]
        with self.assertRaises(PlannerValidationError):
            validate_response(json.dumps(response), catalog)
        external["quote"] = "invented"
        with self.assertRaises(PlannerValidationError):
            build_catalog(self.record, self.text, retrieval)

    def test_external_snapshot_missing_excluded_with_warning(self):
        catalog = build_catalog(self.record, self.text, {"hits": [{"role": "resource", "evidence_id": "ref:1"}]})
        self.assertFalse(catalog["external_references"])
        self.assertTrue(any("snapshot missing" in warning for warning in catalog["warnings"]))

    def test_repeated_external_id_rejected_even_without_snapshot(self):
        hit = {"role": "resource", "evidence_id": "ref:1"}
        with self.assertRaises(PlannerValidationError):
            build_catalog(self.record, self.text, {"hits": [hit, hit]})

    def test_json_repeated_keys_markdown_extra_text_and_nonfinite_rejected(self):
        for raw in ('{"tasks":[],"tasks":[],"dependency_candidates":[],"resource_references":[]}',
                    '```json\n{}\n```', '{} trailing explanation', '[]', '', '{"tasks":NaN}'):
            with self.subTest(raw=raw), self.assertRaises(PlannerValidationError):
                validate_response(raw, self.catalog)

    def test_empty_selection_abstains_honestly(self):
        result = self.validate({"tasks": [], "dependency_candidates": [], "resource_references": []})
        self.assertEqual(result["status"], "abstained")
        self.assertFalse(result["steps"])
        self.assertEqual(result["unselected_task_ids"], ["T001", "T002"])

    def test_zero_eligible_tasks_do_not_load_any_model(self):
        record, text = fixture()
        record["fields"]["activities"] = []
        with patch("psai_planner.planner._generate_local_response", side_effect=AssertionError("must not load")):
            result = plan_project(record, text)
        self.assertEqual(result["status"], "abstained")
        self.assertIsNone(result["raw_response"])
        self.assertFalse(result["provenance"]["model_loaded"])

    def test_mocked_model_success_retains_hashes_raw_response_and_prompt(self):
        raw = json.dumps(selection())
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "config.json").write_text('{"model_type":"fixture"}', encoding="utf-8")
            (folder / "model.safetensors").write_bytes(b"test weights only")
            with patch("psai_planner.planner._generate_local_response", return_value=(raw, {"input_tokens": 100})):
                result = plan_project(self.record, self.text, model_dir=folder)
        self.assertEqual(result["raw_response"], raw)
        self.assertEqual(result["messages"], build_prompt(self.catalog))
        self.assertEqual(result["provenance"]["response_sha256"], hashlib.sha256(raw.encode()).hexdigest())
        self.assertTrue(result["provenance"]["model_loaded"])

    def test_malformed_model_output_preserved_on_error_not_silent_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "config.json").write_text('{}', encoding="utf-8")
            (folder / "model.safetensors").write_bytes(b"fixture")
            with patch("psai_planner.planner._generate_local_response", return_value=("not JSON", {})):
                with self.assertRaises(PlannerValidationError) as raised:
                    plan_project(self.record, self.text, model_dir=folder)
        self.assertEqual(raised.exception.raw_response, "not JSON")
        self.assertTrue(raised.exception.messages)

    def test_impossible_context_budget_rejected_before_model_load(self):
        with self.assertRaisesRegex(PlannerValidationError, "token budgets"):
            plan_project(self.record, self.text, config={"max_input_tokens": 4000, "max_new_tokens": 1024})

    def test_actual_token_count_rejected_without_loading_weights(self):
        from psai_planner.planner import DEFAULT_CONFIG, _generate_local_response
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = "oversized prompt"
        tokenizer.return_value = {"input_ids": SimpleNamespace(shape=(1, 4000))}
        tokenizers = Mock()
        tokenizers.from_pretrained.return_value = tokenizer
        models = Mock()
        fake = SimpleNamespace(AutoTokenizer=tokenizers, AutoModelForCausalLM=models)
        with patch.dict(sys.modules, {"torch": SimpleNamespace(), "transformers": fake}):
            with self.assertRaisesRegex(PlannerValidationError, "4000 tokens"):
                _generate_local_response(build_prompt(self.catalog), Path("unused"), dict(DEFAULT_CONFIG))
        models.from_pretrained.assert_not_called()

    def test_missing_model_requires_explicit_directory(self):
        with self.assertRaisesRegex(PlannerValidationError, "local planner model"):
            plan_project(self.record, self.text)


if __name__ == "__main__":
    unittest.main()
