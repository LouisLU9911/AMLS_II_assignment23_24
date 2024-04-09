import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForImageClassification
from .configuration_moe import MoEConfig


def subgate(num_out):
    layers = nn.Sequential(
        nn.Flatten(),
        nn.Linear(224 * 224 * 3, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, num_out),
    )
    return layers


class MoEModelForImageClassification(PreTrainedModel):
    config_class = MoEConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_classes = config.num_classes
        self.switch_gate_model = AutoModelForImageClassification.from_pretrained(
            config.switch_gate
        )
        self.baseline_model = AutoModelForImageClassification.from_pretrained(
            config.baseline_model
        )
        self.expert_model_1 = AutoModelForImageClassification.from_pretrained(
            config.experts[0]
        )
        self.expert_model_2 = AutoModelForImageClassification.from_pretrained(
            config.experts[1]
        )

        self.subgate = subgate(2)

        # Freeze all params
        for module in [
            self.switch_gate_model,
            self.baseline_model,
            self.expert_model_1,
            self.expert_model_2,
        ]:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, pixel_values, labels=None):
        switch_gate_result = self.switch_gate_model(pixel_values).logits
        expert1_result = self.expert_model_1(pixel_values).logits
        expert2_result = self.expert_model_2(pixel_values).logits

        # Gating Network
        experts_result = torch.stack(
            [expert1_result, expert2_result], dim=1
        ) * switch_gate_result.unsqueeze(-1)

        experts_result = experts_result.sum(dim=1)
        baseline_model_result = self.baseline_model(pixel_values).logits

        subgate_result = self.subgate(pixel_values)
        subgate_prob = F.softmax(subgate_result, dim=-1)

        experts_and_base_result = torch.stack(
            [experts_result, baseline_model_result], dim=1
        ) * subgate_prob.unsqueeze(-1)

        logits = experts_and_base_result.sum(dim=1)
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
