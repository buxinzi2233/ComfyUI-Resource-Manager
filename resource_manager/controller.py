"""Local queue connector; only the existing execution worker mutates model resources.

Dispatch only posts shared flags: there is no acknowledgement, ownership, or failure channel.
Another caller may overwrite, merge, or consume flags; disappearance is not worker receipt.
A newly queued task may execute before a posted release; the single native worker
applies resource operations between executions, never in this connector.
GPU-only offload retains CPU references. Full release resets executor caches but cannot
promise that third-party references, allocator arenas, or all process RSS disappear.
Only normal prompt-worker activity is coordinated; third-party background loading
is not tracked or protected by the queue mutex.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated, Literal

import psutil
import torch
from execution import PromptQueue
from pydantic import BaseModel, ConfigDict, Field

from ._native import IdleState, idle_decision

Action = Literal["unload_gpu", "release_models"]
OperationState = Literal["waiting_for_idle", "dispatched", "flags_consumed_unconfirmed", "cancelled"]
Seconds = Annotated[int, Field(strict=True, ge=1, le=86400)]


class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, frozen=True)
    unload_gpu_seconds: Seconds | None
    release_models_seconds: Seconds | None
    unload_on_model_switch: bool = True


class ManualRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, frozen=True)
    action: Literal["cache_only", "unload_gpu", "release_models"]


class MemorySample(BaseModel):
    rss_bytes: int
    cuda_initialized: bool
    cuda_allocated_bytes: int | None
    cuda_reserved_bytes: int | None
    sampled_at_ms: int


class Operation(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: int
    action: Action
    source: Literal["manual", "idle"]
    state: OperationState
    before: MemorySample
    after: MemorySample | None
    task_counter: int


class Status(BaseModel):
    settings: Settings
    busy: bool
    pending_queue: int
    running_tasks: int
    idle_since_ms: int | None
    memory: MemorySample
    operations: list[Operation]
    monitor_error: str | None
    completion_acknowledgement: bool
    cache_only_supported: bool
    managed_switch_supported: bool


class OperationConflictError(RuntimeError):
    """An existing pending operation prevents another request."""


class UnsupportedOperationError(RuntimeError):
    """The core lacks a safe worker command for the requested action."""


class OperationNotFoundError(LookupError):
    """The requested waiting operation does not exist."""


def clock_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def memory_sample(now_ms: int) -> MemorySample:
    initialized = torch.cuda.is_initialized()
    allocated = sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())) if initialized else None
    reserved = sum(torch.cuda.memory_reserved(i) for i in range(torch.cuda.device_count())) if initialized else None
    return MemorySample(
        rss_bytes=psutil.Process().memory_info().rss,
        cuda_initialized=initialized,
        cuda_allocated_bytes=allocated,
        cuda_reserved_bytes=reserved,
        sampled_at_ms=now_ms,
    )


def read_settings(path: Path) -> Settings:
    if not path.exists():
        return Settings(unload_gpu_seconds=None, release_models_seconds=None)
    return Settings.model_validate_json(path.read_text(encoding="utf-8"))


def write_settings(path: Path, settings: Settings) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(path)


def timeout_ms(seconds: int | None) -> int | None:
    return None if seconds is None else seconds * 1000


class QueueBridge:
    """External-system connector, owned and called exclusively by the aiohttp event loop.

    The queue mutex atomically validates idle and posts native flags. It is not an
    executor lock: no model or allocator calls are made here, even while holding it.
    """

    def __init__(self, queue: PromptQueue, config_path: Path) -> None:
        self.queue = queue
        self.config_path = config_path
        self.settings = read_settings(config_path)
        self.idle = IdleState(None, False, False)
        self.operations: list[Operation] = []
        self.next_id = 1
        self.last_task_counter = queue.task_counter
        self.monitor_error: str | None = None
        self.managed_switch_supported = False

    def configure(self, settings: Settings) -> Settings:
        write_settings(self.config_path, settings)
        self.settings = settings
        self.idle = IdleState(None, False, False)
        return settings

    def request(self, request: ManualRequest, now_ms: int) -> Operation:
        if request.action == "cache_only":
            raise UnsupportedOperationError(
                "cache_only cannot be dispatched safely: this ComfyUI worker has no cache-only command; "
                "free_memory also resets executor outputs. No resource operation was performed."
            )
        if any(op.state == "waiting_for_idle" for op in self.operations):
            raise OperationConflictError("A manual release is already waiting; cancel it before requesting another.")
        operation = Operation(
            id=self.next_id, action=request.action, source="manual", state="waiting_for_idle",
            before=memory_sample(now_ms), after=None, task_counter=self.queue.task_counter,
        )
        self.next_id += 1
        self.operations = [*self.operations[-19:], operation]
        self.tick(now_ms)
        return next(op for op in self.operations if op.id == operation.id)

    def cancel(self, operation_id: int) -> Operation:
        for index, operation in enumerate(self.operations):
            if operation.id == operation_id:
                if operation.state != "waiting_for_idle":
                    raise OperationConflictError(
                        "Only waiting requests can be cancelled; native flags already dispatched are not owned by this plugin."
                    )
                cancelled = operation.model_copy(update={"state": "cancelled"})
                self.operations = [*self.operations[:index], cancelled, *self.operations[index + 1:]]
                return cancelled
        raise OperationNotFoundError(f"Waiting resource operation id={operation_id} was not found.")

    def tick(self, now_ms: int) -> None:
        with self.queue.mutex:
            flags_pending = "unload_models" in self.queue.flags or "free_memory" in self.queue.flags
            busy = bool(self.queue.queue or self.queue.currently_running)
            for index, operation in enumerate(self.operations):
                if operation.state == "dispatched" and not flags_pending:
                    observed = operation.model_copy(update={
                        "state": "flags_consumed_unconfirmed",
                    })
                    self.operations = [*self.operations[:index], observed, *self.operations[index + 1:]]

            # Refresh observations throughout the same idle cycle, not just at flag removal.
            # Stop sampling an operation when another task starts; samples are not acknowledgements.
            for index, operation in enumerate(self.operations):
                if (operation.state == "flags_consumed_unconfirmed" and not busy
                        and self.queue.task_counter == operation.task_counter):
                    observed = operation.model_copy(update={"after": memory_sample(now_ms)})
                    self.operations = [*self.operations[:index], observed, *self.operations[index + 1:]]

            if self.queue.task_counter != self.last_task_counter:
                self.idle = IdleState(None, False, False)
                self.last_task_counter = self.queue.task_counter
            if busy:
                self.idle = idle_decision(self.idle, now_ms, True, None, None).state
                return
            if flags_pending:
                return

            for index, operation in enumerate(self.operations):
                if operation.state == "waiting_for_idle":
                    self._dispatch(operation.action)
                    dispatched = operation.model_copy(update={
                        "state": "dispatched", "before": memory_sample(now_ms),
                        "task_counter": self.queue.task_counter,
                    })
                    self.operations = [*self.operations[:index], dispatched, *self.operations[index + 1:]]
                    self.idle = IdleState(
                        self.idle.idle_since_ms if self.idle.idle_since_ms is not None else now_ms,
                        True, operation.action == "release_models" or self.idle.full_dispatched,
                    )
                    return

            decision = idle_decision(
                self.idle, now_ms, False,
                timeout_ms(self.settings.unload_gpu_seconds), timeout_ms(self.settings.release_models_seconds),
            )
            if decision.action is not None:
                if decision.action not in ("unload_gpu", "release_models"):
                    raise UnsupportedOperationError(f"Rust returned unsupported action={decision.action!r}.")
                before = memory_sample(now_ms)
                self._dispatch(decision.action)
                operation = Operation(
                    id=self.next_id, action=decision.action, source="idle", state="dispatched",
                    before=before, after=None, task_counter=self.queue.task_counter,
                )
                self.next_id += 1
                self.operations = [*self.operations[-19:], operation]
            self.idle = decision.state

    def _dispatch(self, action: Action) -> None:
        self.queue.set_flag("unload_models", True)
        if action == "release_models":
            self.queue.set_flag("free_memory", True)

    def status(self, now_ms: int) -> Status:
        with self.queue.mutex:
            pending = len(self.queue.queue)
            running = len(self.queue.currently_running)
        return Status(
            settings=self.settings, busy=bool(pending or running), pending_queue=pending, running_tasks=running,
            idle_since_ms=self.idle.idle_since_ms, memory=memory_sample(now_ms), operations=list(self.operations),
            monitor_error=self.monitor_error, completion_acknowledgement=False,
            cache_only_supported=False, managed_switch_supported=self.managed_switch_supported,
        )
