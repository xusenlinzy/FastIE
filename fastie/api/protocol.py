import secrets
import time
from enum import Enum, unique
from typing import (
    Any,
    Dict,
    List,
    Union,
    Optional,
)

from pydantic import BaseModel, Field
from typing_extensions import Literal


@unique
class Task(str, Enum):
    CLS = "text-classification"
    NER = "named-entity-recognition"
    REL = "relation-extraction"
    EVENT = "event-extraction"
    UIE = "uie"


class IECreateParams(BaseModel):
    texts: Union[str, List[str]]
    ie_schema: Optional[Any] = None
    batch_size: Optional[int] = 16
    max_length: Optional[int] = 512
    language: Optional[str] = "zh"


class CLSResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cls-{secrets.token_hex(12)}")
    object: Literal["text-classification"] = "text-classification"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    labels: List[Any]


class NERResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"ner-{secrets.token_hex(12)}")
    object: Literal["named-entity-recognition"] = "named-entity-recognition"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    labels: List[Dict[str, Any]]


class RELResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"rel-{secrets.token_hex(12)}")
    object: Literal["relation-extraction"] = "relation-extraction"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    labels: List[Dict[str, Any]]


class EVENTResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"event-{secrets.token_hex(12)}")
    object: Literal["event-extraction"] = "event-extraction"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    labels: List[Dict[str, Any]]


class UIEResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"uie-{secrets.token_hex(12)}")
    object: Literal["uie"] = "uie"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    labels: List[Dict[str, Any]]


RESPONSE_MAP = {
    "classification": CLSResponse,
    "ner": NERResponse,
    "relextraction": RELResponse,
    "eventextraction": EVENTResponse,
    "uiemodel": UIEResponse,
}


def get_response_model(architecture: str):
    for k, v in RESPONSE_MAP.items():
        if architecture.endswith(k):
            return v
