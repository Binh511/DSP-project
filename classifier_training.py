"""Weighted classification loss with explicit Trainer accumulation semantics."""
import torch
from transformers import Trainer


class WeightedClassificationTrainer(Trainer):
    def __init__(self, *args, class_weights, label_smoothing=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
        if not 0 <= label_smoothing < 1:
            raise ValueError('label_smoothing must be in [0, 1)')
        self.loss_label_smoothing = label_smoothing
        # compute_loss returns a MICROBatch mean and does not use the total
        # accumulated num_items_in_batch. Trainer must divide by actual GA steps.
        # A model's **kwargs signature must not silently disable that division.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = dict(inputs)
        labels = inputs.pop('labels')
        out = model(**inputs)
        logits = out.logits.float()
        per_sample = torch.nn.functional.cross_entropy(
            logits, labels, label_smoothing=self.loss_label_smoothing, reduction='none')
        loss = (per_sample * self.class_weights.to(logits.device)[labels]).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite classification loss; training/evaluation cannot continue')
        return (loss, out) if return_outputs else loss
