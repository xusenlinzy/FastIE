from transformers import (
    BertConfig,
    RoFormerConfig,
)


class BertGrteRelConfig(BertConfig):
    def __init__(
        self,
        schemas=None,
        rounds=3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.schemas = schemas or []
        self.rounds = rounds


class RoFormerGrteRelConfig(RoFormerConfig):
    def __init__(
        self,
        schemas=None,
        rounds=3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.schemas = schemas or []
        self.rounds = rounds
