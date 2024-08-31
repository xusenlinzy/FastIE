from transformers import (
    BertConfig,
    RoFormerConfig,
)


class BertCnnNerConfig(BertConfig):
    def __init__(
        self,
        schemas=None,
        size_embed_dim=0,
        biaffine_size=200,
        num_heads=4,
        cnn_hidden_size=200,
        kernel_size=3,
        cnn_depth=3,
        decode_thresh=0.5,
        allow_nested=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.schemas = schemas or []
        self.size_embed_dim = size_embed_dim
        self.biaffine_size = biaffine_size
        self.num_heads = num_heads
        self.cnn_hidden_size = cnn_hidden_size
        self.kernel_size = kernel_size
        self.cnn_depth = cnn_depth
        self.decode_thresh = decode_thresh
        self.allow_nested = allow_nested


class RoFormerCnnNerConfig(RoFormerConfig):
    def __init__(
        self,
        schemas=None,
        size_embed_dim=0,
        biaffine_size=200,
        num_heads=4,
        cnn_hidden_size=200,
        kernel_size=3,
        cnn_depth=3,
        decode_thresh=0.5,
        allow_nested=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.schemas = schemas or []
        self.size_embed_dim = size_embed_dim
        self.biaffine_size = biaffine_size
        self.num_heads = num_heads
        self.cnn_hidden_size = cnn_hidden_size
        self.kernel_size = kernel_size
        self.cnn_depth = cnn_depth
        self.decode_thresh = decode_thresh
        self.allow_nested = allow_nested
