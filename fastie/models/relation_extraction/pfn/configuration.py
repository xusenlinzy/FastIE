from transformers import (
    BertConfig,
    RoFormerConfig,
)


class BertPfnRelConfig(BertConfig):
    def __init__(
        self,
        schemas=None,
        pfn_hidden_size=300,
        decode_thresh=0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.schemas = schemas or []
        self.pfn_hidden_size = pfn_hidden_size
        self.decode_thresh = decode_thresh


class RoFormerPfnRelConfig(RoFormerConfig):
    def __init__(
        self,
        schemas=None,
        pfn_hidden_size=300,
        decode_thresh=0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.schemas = schemas or []
        self.pfn_hidden_size = pfn_hidden_size
        self.decode_thresh = decode_thresh
