from transformers import (
    BertConfig,
    RoFormerConfig,
)


class BertGlobalPointerNerConfig(BertConfig):
    def __init__(
        self,
        schemas=None,
        head_size=64,
        use_rope=True,
        is_sparse=True,
        efficient=True,
        decode_thresh=0.,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.schemas = schemas or []
        self.head_size = head_size
        self.use_rope = use_rope
        self.is_sparse = is_sparse
        self.efficient = efficient
        self.decode_thresh = decode_thresh


class RoFormerGlobalPointerNerConfig(RoFormerConfig):
    def __init__(
        self,
        schemas=None,
        head_size=64,
        use_rope=True,
        is_sparse=True,
        efficient=True,
        decode_thresh=0.,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.schemas = schemas or []
        self.head_size = head_size
        self.use_rope = use_rope
        self.is_sparse = is_sparse
        self.efficient = efficient
        self.decode_thresh = decode_thresh
