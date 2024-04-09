import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForImageClassification
from .configuration_moe import MoEConfig


def subgate(num_classes):
    layers = nn.Sequential(
        nn.Flatten(),
        nn.Linear(224 * 224 * 3, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes * 2),
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
        self.base_model1 = AutoModelForImageClassification.from_pretrained(
            config.base_model
        )
        self.base_model2 = AutoModelForImageClassification.from_pretrained(
            config.base_model
        )
        self.expert_model_1 = AutoModelForImageClassification.from_pretrained(
            config.experts[0]
        )
        self.expert_model_2 = AutoModelForImageClassification.from_pretrained(
            config.experts[1]
        )

        self.subgate1 = subgate(config.num_classes)
        self.subgate2 = subgate(config.num_classes)

        # Freeze all params
        for module in [
            self.switch_gate_model,
            self.base_model1,
            self.base_model2,
            self.expert_model_1,
            self.expert_model_2,
        ]:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, pixel_values, labels=None):
        switch_gate_result = self.switch_gate_model(pixel_values).logits
        base_model1_result = self.base_model1(pixel_values).logits
        base_model2_result = self.base_model2(pixel_values).logits

        expert1_result = self.expert_model_1(pixel_values).logits
        expert2_result = self.expert_model_2(pixel_values).logits

        subgate1_result = self.subgate1(pixel_values)
        subgate1_result = torch.reshape(subgate1_result, (2, -1, self.num_classes))

        subgate2_result = self.subgate2(pixel_values)
        subgate2_result = torch.reshape(subgate2_result, (2, -1, self.num_classes))

        expert1_and_base_res = (
            expert1_result * subgate1_result[0, :, :]
            + base_model1_result * subgate1_result[1, :, :]
        )
        expert2_and_base_res = (
            expert2_result * subgate2_result[0, :, :]
            + base_model2_result * subgate2_result[1, :, :]
        )

        # Gating Network
        expert1_and_base_res = expert1_and_base_res * switch_gate_result[
            :, 0
        ].unsqueeze(1)
        expert2_and_base_res = expert2_and_base_res * switch_gate_result[
            :, 1
        ].unsqueeze(1)

        logits = expert1_and_base_res + expert2_and_base_res
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
