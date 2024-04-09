from transformers import PretrainedConfig
from typing import Dict, List


MOE_MODEL_TYPE = "moe"
DEFAULT_HUGGINGFACE_ACCOUNT = "louislu9911"
model_checkpoint = "convnextv2-base-1k-224"

EXPERTS = [
    f"{DEFAULT_HUGGINGFACE_ACCOUNT}/Expert1-leaf-disease-{model_checkpoint}-0_4",
    f"{DEFAULT_HUGGINGFACE_ACCOUNT}/Expert2-leaf-disease-{model_checkpoint}-1_2_3",
]
SWITCH_GATE = (
    f"{DEFAULT_HUGGINGFACE_ACCOUNT}/switch_gate-leaf-disease-{model_checkpoint}"
)
BASE_MODEL = (
    f"{DEFAULT_HUGGINGFACE_ACCOUNT}/BaseModel-leaf-disease-{model_checkpoint}-0_1_2_3_4"
)


class MoEConfig(PretrainedConfig):
    model_type = MOE_MODEL_TYPE

    def __init__(
        self,
        experts: List[str] = EXPERTS,
        switch_gate: str = SWITCH_GATE,
        base_model: str = BASE_MODEL,
        num_classes: int = 5,
        expert_class_mapping: Dict[int, List[int]] = None,
        **kwargs,
    ):
        self.experts = experts
        self.switch_gate = switch_gate
        self.base_model = base_model
        self.num_classes = num_classes
        self.expert_class_mapping = expert_class_mapping
        super().__init__(**kwargs)
