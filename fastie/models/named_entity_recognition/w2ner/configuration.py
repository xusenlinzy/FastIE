from transformers import (
    BertConfig,
    RoFormerConfig,
)


class BertW2nerConfig(BertConfig):
    def __init__(
        self,
        schemas=None,
        use_last_4_layers=False,
        dist_emb_size=20,
        type_emb_size=20,
        lstm_hidden_size=512,
        conv_hidden_size=96,
        biaffine_size=512,
        ffn_hidden_size=288,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.schemas = schemas or []
        self.use_last_4_layers = use_last_4_layers
        self.dist_emb_size = dist_emb_size
        self.type_emb_size = type_emb_size
        self.lstm_hidden_size = lstm_hidden_size
        self.conv_hidden_size = conv_hidden_size
        self.biaffine_size = biaffine_size
        self.ffn_hidden_size = ffn_hidden_size


class RoFormerW2nerConfig(RoFormerConfig):
    def __init__(
        self,
        schemas=None,
        use_last_4_layers=False,
        dist_emb_size=20,
        type_emb_size=20,
        lstm_hidden_size=512,
        conv_hidden_size=96,
        biaffine_size=512,
        ffn_hidden_size=288,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.schemas = schemas or []
        self.use_last_4_layers = use_last_4_layers
        self.dist_emb_size = dist_emb_size
        self.type_emb_size = type_emb_size
        self.lstm_hidden_size = lstm_hidden_size
        self.conv_hidden_size = conv_hidden_size
        self.biaffine_size = biaffine_size
        self.ffn_hidden_size = ffn_hidden_size
