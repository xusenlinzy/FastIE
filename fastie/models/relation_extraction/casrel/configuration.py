from transformers import (
    BertConfig,
    RoFormerConfig,
)


class BertCasrelRelConfig(BertConfig):
    def __init__(
        self,
        schemas=None,
        start_thresh=0.5,
        end_thresh=0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.schemas = schemas or []
        self.start_thresh = start_thresh
        self.end_thresh = end_thresh


class RoFormerCasrelRelConfig(RoFormerConfig):
    def __init__(
        self,
        schemas=None,
        start_thresh=0.5,
        end_thresh=0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.schemas = schemas or []
        self.start_thresh = start_thresh
        self.end_thresh = end_thresh
