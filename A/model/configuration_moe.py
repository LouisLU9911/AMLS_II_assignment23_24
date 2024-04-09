from transformers import PretrainedConfig
from typing import Dict, List


MOE_MODEL_TYPE = "moe"


class MoEConfig(PretrainedConfig):
    model_type = MOE_MODEL_TYPE

    def __init__(
        self,
        experts: List[str],
        switch_gate: str,
        base_model: str,
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
