from dataclasses import dataclass, field
from typing import Optional


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
    num_api_servers: Optional[int] = field(
        default=None,
        metadata={"help": "Launch servers on multiple process or thread."},
    )
    accelerator: str = field(
        default="auto",
        metadata={"help": "The type of hardware to use (cpu, GPUs, mps)."},
    )
    devices: str = field(
        default="auto",
        metadata={"help": "The number of (CPUs, GPUs) to use for the server."},
    )
    workers_per_device: Optional[int] = field(
        default=1,
        metadata={"help": "Number of workers (processes) per device."},
    )
    max_batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "The max number of requests to batch together."},
    )
    timeout: Optional[float] = field(
        default=30,
        metadata={"help": "The timeout (in seconds) for a request."},
    )
    batch_timeout: Optional[float] = field(
        default=0.0,
        metadata={"help": "The timeout (in ms) until the server stops waiting to batch inputs."},
    )

