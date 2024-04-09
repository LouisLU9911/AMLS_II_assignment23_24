import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForImageClassification
from .configuration_moe import MoEConfig


class MoEModelForImageClassification(PreTrainedModel):
    config_class = MoEConfig

    def __init__(self, config):
        super().__init__(config)
        experts = config.experts
        switch_gate = config.switch_gate
        base_model = config.base_model
        self.switch_gate_model = AutoModelForImageClassification.from_pretrained(
            switch_gate
        )
        self.expert_model_1 = AutoModelForImageClassification.from_pretrained(
            experts[0]
        )
        self.expert_model_2 = AutoModelForImageClassification.from_pretrained(
            experts[1]
        )

        # Freeze all params
        for module in [
            self.switch_gate_model,
            self.expert_model_1,
            self.expert_model_2,
        ]:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, pixel_values, labels=None):
        expert1_result = self.expert_model_1(pixel_values).logits
        expert2_result = self.expert_model_2(pixel_values).logits
        gate_result = F.one_hot(
            torch.argmax(self.switch_gate_model(pixel_values).logits, dim=1)
        )
        expert1_result = expert1_result.mul(gate_result[:, 0])
        expert2_result = expert2_result.mul(gate_result[:, 1])
        logits = expert1_result + expert2_result
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
