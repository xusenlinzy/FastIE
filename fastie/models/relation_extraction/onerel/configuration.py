from transformers import (
    BertConfig,
    RoFormerConfig,
)


class BertOneRelRelConfig(BertConfig):
    def __init__(
        self,
        schemas=None,
        entity_pair_dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.schemas = schemas or []
        self.entity_pair_dropout = entity_pair_dropout


class RoFormerOneRelRelConfig(RoFormerConfig):
    def __init__(
        self,
        schemas=None,
        entity_pair_dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.schemas = schemas or []
        self.entity_pair_dropout = entity_pair_dropout
