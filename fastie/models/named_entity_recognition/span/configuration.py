from transformers import (
    BertConfig,
    RoFormerConfig,
)


class BertSpanNerConfig(BertConfig):
    def __init__(self, schemas=None, **kwargs):
        super().__init__(**kwargs)
        self.schemas = schemas or []


class RoFormerSpanNerConfig(RoFormerConfig):
    def __init__(self, schemas=None, **kwargs):
        super().__init__(**kwargs)
        self.schemas = schemas or []
