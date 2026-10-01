class IdleState:
    idle_since_ms: int | None
    gpu_dispatched: bool
    full_dispatched: bool
    def __init__(self, idle_since_ms: int | None, gpu_dispatched: bool, full_dispatched: bool) -> None: ...

class Decision:
    state: IdleState
    action: str | None

def idle_decision(state: IdleState, now_ms: int, busy: bool, gpu_timeout_ms: int | None, full_timeout_ms: int | None) -> Decision: ...

class ModelIdentity:
    path: str
    size: int
    mtime_ns: int
    role: str
    loading_config: str
    def __init__(self, path: str, size: int, mtime_ns: int, role: str, loading_config: str) -> None: ...
    def matches(self, other: ModelIdentity) -> bool: ...
