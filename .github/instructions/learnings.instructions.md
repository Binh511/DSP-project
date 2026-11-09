---
description: Preserve the distinction between source evidence, proposed planning links and independent evaluation.
applyTo: 'psai_planner/**/*.py,psai_project/**/*.py,psai_rules/**/*.py,tests/**/*.py'
---

Extracting a valid quote does not establish semantic correctness, requirement coverage or an approved plan. Preserve exact source spans and hashes, and mark cross-sentence task/checklist/resource/dependency associations as suggestions needing human review. When a proposal already names staff or constraints, distinguish a missed extraction from genuinely missing allocation or approval; do not fill in owners, dates or budgets. Add regressions on varied wording and retain an explicit legacy protocol path when changing catalogs or prompts used by frozen reports; keep model weights and research metrics unchanged.

For a small local planner, prefer compact selections of known source IDs and rehydrate wording in code instead of spending most output tokens repeating empty task objects. Validate with actual inference on more than one source: mocked JSON tests do not reveal model abstention, token exhaustion or invalid hierarchy output. Never repair those failures by accepting unknown citations, inventing missing associations or reporting a fallback as successful model inference.

In the application's preserve-source policy, model omission is not evidence that a catalogued task is unnecessary. Retain omitted source tasks with an explicit `source_catalog_preserved` origin and keep the exact model response; do not attribute deterministic preservation or lexical review suggestions to the LLM.
