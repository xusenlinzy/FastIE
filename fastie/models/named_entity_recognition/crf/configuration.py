from transformers import BertConfig, RoFormerConfig


class BertCrfNerConfig(BertConfig):
    def __init__(
        self,
        schemas=None,
        use_lstm=False,
        mid_hidden_size=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.schemas = schemas or []
        self.use_lstm = use_lstm
        if mid_hidden_size:
            self.mid_hidden_size = mid_hidden_size


class RoFormerCrfNerConfig(RoFormerConfig):
    def __init__(
        self,
        schemas=None,
        use_lstm=False,
        mid_hidden_size=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.schemas = schemas or []
        self.use_lstm = use_lstm
        if mid_hidden_size:
            self.mid_hidden_size = mid_hidden_size


class BertCascadeCrfNerConfig(BertConfig):
    def __init__(self, schemas=None, **kwargs):
        super().__init__(**kwargs)
        self.schemas = schemas or []


class RoFormerCascadeCrfNerConfig(RoFormerConfig):
    def __init__(self, schemas, **kwargs):
        super().__init__(**kwargs)
        self.schemas = schemas or []
