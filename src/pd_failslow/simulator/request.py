from dataclasses import dataclass
from typing import Optional


@dataclass
class Request:
    request_id: str

    arrival_time_ms: float
    prompt_tokens: int
    output_tokens: int

    prefill_worker_id: Optional[str] = None
    decode_worker_id: Optional[str] = None
    link_id: Optional[str] = None

    prefill_queue_enter_ms: Optional[float] = None
    prefill_start_ms: Optional[float] = None
    prefill_completion_ms: Optional[float] = None

    transfer_start_ms: Optional[float] = None
    transfer_completion_ms: Optional[float] = None

    decode_admission_ms: Optional[float] = None
    decode_start_ms: Optional[float] = None
    decode_completion_ms: Optional[float] = None

    kv_cache_size_mb: Optional[float] = None