"""Compare plan structure with fixed extraction/retrieval, without quality labels."""
from __future__ import annotations

import hashlib
import json


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _summary(plan):
    steps = plan['steps']
    return {'tasks': len(steps),
            'model_selected_tasks': sum(step.get('selection_origin') == 'llm_selection' for step in steps),
            'source_preserved_tasks': (len(steps) if plan['planning_method'] == 'evidence_ordered_source_items' else
                                       sum(step.get('selection_origin') == 'source_catalog_preserved' for step in steps)),
            'checklist_excerpts': sum(len(step.get('checklist', [])) for step in steps),
            'resource_associations': sum(len(step.get('required_resources', [])) for step in steps),
            'dependency_candidates': len(plan.get('dependency_candidates', []))}


def compare_plans(record, source_text, retrieval, source_plan, selected_plan):
    """Rebuild the baseline; neither this comparison nor counts score semantics.

    The baseline is derived from the very same extracted record used by the
    selected planner. This isolates planning structure; it does not compare
    FAST against FULL, which would also change classifier and retrieval.
    """
    from planning import generate_plan, _evidence_index, validate_plan

    source_hash = hashlib.sha256(source_text.encode('utf-8')).hexdigest()
    if record.get('source_sha256') != source_hash:
        raise ValueError('Planning comparison requires the exact source')
    if source_plan != generate_plan(record, source_text=source_text):
        raise ValueError('Planning baseline differs from the fixed extracted record')
    evidence = _evidence_index(record, source_text)
    validate_plan(selected_plan, set(evidence), evidence_by_id=evidence)
    if (selected_plan.get('document_id') != record['document_id']
            or selected_plan.get('source_sha256') != source_hash):
        raise ValueError('Compared planners must use the same document/source')
    baseline_order = [step['evidence_ids'] for step in source_plan['steps']]
    selected_order = [step['evidence_ids'] for step in selected_plan['steps']]
    baseline_set = {tuple(ids) for ids in baseline_order}
    selected_set = {tuple(ids) for ids in selected_order}
    common_order = [ids for ids in baseline_order if tuple(ids) in selected_set]
    return {'method': 'same_extraction_retrieval_structural_comparison_v1',
            'fixed_inputs': {'source_sha256': source_hash,
                             'extraction_sha256': _digest(record),
                             'retrieval_sha256': _digest(retrieval)},
            'baseline_method': source_plan['planning_method'],
            'selected_method': selected_plan['planning_method'],
            'baseline': _summary(source_plan), 'selected': _summary(selected_plan),
            'selection_changed': baseline_set != selected_set,
            'relative_order_changed': common_order != selected_order,
            'quality_score': None, 'independent_evaluation': False,
            'scope': 'Structural counts on fixed inputs; more links do not prove better planning. '
                     'Semantic correctness, completeness and time savings require separate human evaluation.'}
