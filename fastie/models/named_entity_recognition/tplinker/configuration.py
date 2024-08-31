from transformers import (
    BertConfig,
    RoFormerConfig,
)


class BertTPLinkerNerConfig(BertConfig):
    def __init__(
        self,
        schemas=None,
        shaking_type="cln_plus",
        decode_thresh=0.,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.schemas = schemas or []
        self.shaking_type = shaking_type
        self.decode_thresh = decode_thresh


class RoFormerTPLinkerNerConfig(RoFormerConfig):
    def __init__(
        self,
        schemas=None,
        shaking_type="cln_plus",
        decode_thresh=0.,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.schemas = schemas or []
        self.shaking_type = shaking_type
        self.decode_thresh = decode_thresh
