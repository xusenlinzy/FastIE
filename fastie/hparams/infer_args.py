from dataclasses import dataclass, field


@dataclass
class InferArguments:
    host: str = field(
        default="0.0.0.0",
        metadata={"help": "API server host."},
    )
    port: int = field(
        default=8000,
        metadata={"help": "API server port."},
    )
