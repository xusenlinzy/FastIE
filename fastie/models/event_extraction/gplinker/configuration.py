from transformers import (
    BertConfig,
    RoFormerConfig,
)


class BertGPLinkerEventExtractionConfig(BertConfig):
    def __init__(
        self,
        schemas=None,
        head_size=64,
        split="@",
        decode_thresh=0.,
        trigger=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.schemas = schemas or []
        self.head_size = head_size
        self.decode_thresh = decode_thresh
        self.trigger = trigger
        self.split = split


class RoFormerGPLinkerEventExtractionConfig(RoFormerConfig):
    def __init__(
        self,
        schemas=None,
        head_size=64,
        split="@",
        decode_thresh=0.,
        trigger=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.schemas = schemas or []
        self.head_size = head_size
        self.decode_thresh = decode_thresh
        self.trigger = trigger
        self.split = split
