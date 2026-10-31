# Classifier PSAI v2 — model card

Model: models/controlled_study/classifier_source_clean_context_study_v1/baseline_seed_2026. Base: microsoft/mdeberta-v3-base.

Task: five-way sentence classification with text/context. Labels: OBJECTIVE, REQUIREMENT, ACTIVITY, EXPECTED_RESULT, OTHER.

Selected by configuration mean over three validation seeds, then best exported checkpoint in that configuration. Selected baseline seed 2026; validation Macro-F1 0.632077, accuracy 0.760061, n=1317. Baseline mean 0.615912 ± 0.014231 sample SD on one development split. These are not independent-test results.

Train: 3942 rows/147 IDs after removing 202 rows from three full-source duplicate IDs. Validation: 1317 rows/29 IDs. Original labels retained; no proposed AI changes applied. Raw/normalized full-source matches removed; unknown near-duplicate/exposure risks not certified absent.

Inputs: text600 characters, full study-v1 context, ctx_version3, 256 tokens total paired input. Inference: FP32, five-way softmax argmax, OTHER bias0, no confidence cutoff, no calibrated correctness probability. Training:5epochs,1235updates,LR3e-5,batch1×GA16,weighted CE,smoothing0.1,BF16 train.

Weight SHA256: bf36171708f69d6e7b4d07de95351d8ac8f9ee886f951898ff2801c0c28c3f4a

Model/tokenizer/code/data bindings: [FINAL_MODEL_MANIFEST.json](FINAL_MODEL_MANIFEST.json). Selected configuration and recorded runtime: [final_selection.json](final_selection.json).

No superiority claim over deployed ensemble. No final-test score, expert-label certification, PDF-extraction or report-generation quality claim. V1 and production ensemble preserved.

Run: `python tools/demo_classifier_final.py` from the project root. See [DEMO.md](DEMO.md).
