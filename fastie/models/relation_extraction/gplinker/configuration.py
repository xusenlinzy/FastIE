from transformers import (
    BertConfig,
    RoFormerConfig,
)


class BertGPLinkerRelConfig(BertConfig):
    def __init__(
        self,
        schemas=None,
        head_size=64,
        decode_thresh=0.,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.schemas = schemas or []
        self.head_size = head_size
        self.decode_thresh = decode_thresh


class RoFormerGPLinkerRelConfig(RoFormerConfig):
    def __init__(
        self,
        schemas=None,
        head_size=64,
        decode_thresh=0.,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.schemas = schemas or []
        self.head_size = head_size
        self.decode_thresh = decode_thresh
