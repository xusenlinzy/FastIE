import json

from .extras.env import VERSION
from .models.event_extraction import EVENT_MODEL_MAPPING
from .models.named_entity_recognition import NER_MODEL_MAPPING
from .models.relation_extraction import REL_MODEL_MAPPING
from .models.text_classification import CLS_MODEL_MAPPING
from .models.uie import UIE_MODEL_MAPPING

__version__ = VERSION

SUPPORTED_MODELS = list(CLS_MODEL_MAPPING) + list(NER_MODEL_MAPPING) + list(REL_MODEL_MAPPING) + list(EVENT_MODEL_MAPPING) + list(UIE_MODEL_MAPPING)


def print_supported_models(suffix=None):
    _SUPPORTED_MODELS = SUPPORTED_MODELS
    if isinstance(suffix, str):
        _SUPPORTED_MODELS = [m for m in SUPPORTED_MODELS if m.endswith(suffix)]
    print(json.dumps(_SUPPORTED_MODELS, indent=4))


def get_supported_models(suffix=None):
    _SUPPORTED_MODELS = SUPPORTED_MODELS
    if isinstance(suffix, str):
        _SUPPORTED_MODELS = [m for m in SUPPORTED_MODELS if m.endswith(suffix)]
    return _SUPPORTED_MODELS
