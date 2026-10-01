import asyncio
import gc
import os
import threading
import time
import weakref
from pathlib import Path

import folder_paths
import nodes
import pytest
import torch
from app.assets.manager import default_asset_manager
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from execution import CacheType, PromptExecutor
from pydantic import ValidationError
from server import PromptServer

from resource_manager._native import IdleState, idle_decision
from resource_manager.comfy_plugin import IdentityRequest, ResourceService, attach, model_identity
from resource_manager.controller import (
    ManualRequest, OperationConflictError, QueueBridge, Settings, UnsupportedOperationError,
)


def test_bridge_busy_wait_cancel_and_native_flags(tmp_path: Path) -> None:
    async def run() -> None:
        server = PromptServer(asyncio.get_running_loop(), default_asset_manager())
        queue = server.prompt_queue
        bridge = QueueBridge(queue, tmp_path / "config.json")
        item = (0, "busy", {}, {}, [], {})
        queue.put(item)
        queued = bridge.request(ManualRequest(action="release_models"), 0)
        assert queued.state == "waiting_for_idle"
        assert queue.get_flags(False) == {}
        bridge.cancel(queued.id)
        assert bridge.operations[-1].state == "cancelled"
        with pytest.raises(OperationConflictError):
            bridge.cancel(queued.id)
        _, task_id = queue.get(timeout=0)
        running = bridge.request(ManualRequest(action="unload_gpu"), 1)
        assert running.state == "waiting_for_idle"
        bridge.tick(10_000)
        assert queue.get_flags(False) == {}
        queue.task_done(task_id, {"outputs": {}}, status=None)
        bridge.tick(10_001)
        assert queue.get_flags(False) == {"unload_models": True}
        assert bridge.operations[-1].state == "dispatched"
        with pytest.raises(OperationConflictError):
            bridge.cancel(running.id)
        queue.get_flags()
        bridge.tick(10_002)
        assert bridge.operations[-1].state == "flags_consumed_unconfirmed"
        assert bridge.operations[-1].after is not None
        assert bridge.status(10_003).completion_acknowledgement is False
        with pytest.raises(UnsupportedOperationError):
            bridge.request(ManualRequest(action="cache_only"), 10_004)
        assert queue.get_flags(False) == {}
    asyncio.run(run())


def test_idle_two_levels_once_and_fast_task_invalidates_timer(tmp_path: Path) -> None:
    async def run() -> None:
        server = PromptServer(asyncio.get_running_loop(), default_asset_manager())
        queue = server.prompt_queue
        bridge = QueueBridge(queue, tmp_path / "config.json")
        bridge.configure(Settings(unload_gpu_seconds=1, release_models_seconds=3))
        bridge.tick(0)
        bridge.tick(999)
        assert not queue.get_flags(False)
        bridge.tick(1000)
        assert queue.get_flags() == {"unload_models": True}
        bridge.tick(1500)
        assert not queue.get_flags(False)
        bridge.tick(3000)
        assert queue.get_flags() == {"unload_models": True, "free_memory": True}
        bridge.tick(10_000)
        assert not queue.get_flags(False)
        queue.put((0, "fast-task", {}, {}, [], {}))
        _, task_id = queue.get(timeout=0)
        queue.task_done(task_id, {"outputs": {}}, status=None)
        bridge.tick(10_001)
        assert bridge.idle.idle_since_ms == 10_001
        assert not queue.get_flags(False)
        bridge.tick(11_001)
        assert queue.get_flags() == {"unload_models": True}
        assert len(bridge.operations) == 3
        restored = QueueBridge(queue, tmp_path / "config.json")
        assert restored.settings == bridge.settings
    asyncio.run(run())


def test_pending_queue_invalidates_idle_and_external_flags_are_not_overwritten(tmp_path: Path) -> None:
    async def run() -> None:
        server = PromptServer(asyncio.get_running_loop(), default_asset_manager())
        queue = server.prompt_queue
        bridge = QueueBridge(queue, tmp_path / "config.json")
        bridge.configure(Settings(unload_gpu_seconds=None, release_models_seconds=1))
        bridge.tick(0)
        queue.put((0, "pending", {}, {}, [], {}))
        bridge.tick(2000)
        assert bridge.idle.idle_since_ms is None
        assert queue.get_flags(False) == {}
        _, task_id = queue.get(timeout=0)
        queue.task_done(task_id, {"outputs": {}}, status=None)
        queue.set_flag("unload_models", False)
        bridge.tick(2001)
        assert queue.get_flags() == {"unload_models": False}
        bridge.tick(3000)
        bridge.tick(4000)
        assert queue.get_flags() == {"unload_models": True, "free_memory": True}
    asyncio.run(run())


def test_rust_policy_does_not_mutate_input_and_has_required_arguments() -> None:
    initial = IdleState(None, False, False)
    decision = idle_decision(initial, 100, False, None, 10)
    assert initial.idle_since_ms is None
    assert decision.state.idle_since_ms == 100
    with pytest.raises(TypeError):
        idle_decision(initial, 100, False)
    with pytest.raises(AttributeError):
        initial.gpu_dispatched = True


@pytest.mark.parametrize("payload", [
    {"unload_gpu_seconds": 0, "release_models_seconds": None},
    {"unload_gpu_seconds": True, "release_models_seconds": None},
    {"unload_gpu_seconds": 1.5, "release_models_seconds": None},
    {"unload_gpu_seconds": None},
])
def test_settings_reject_invalid_external_data(payload: dict[str, int | float | bool | None]) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate(payload)


def test_identity_same_model_replacement_precision_and_multiple_models(tmp_path: Path) -> None:
    directory = tmp_path / "models"
    directory.mkdir()
    a = directory / "a.safetensors"
    b = directory / "b.safetensors"
    a.write_bytes(b"A")
    b.write_bytes(b"B")
    folder_paths.add_model_folder_path("diffusion_models", str(directory), is_default=True)
    request_a = IdentityRequest(category="diffusion_models", filename=a.name, loading_config="default")
    request_b = IdentityRequest(category="diffusion_models", filename=b.name, loading_config="default")
    first_a = model_identity(request_a)
    first_b = model_identity(request_b)
    second_a = model_identity(request_a)
    assert first_a.matches(second_a)
    assert not first_a.matches(first_b)
    stat = a.stat()
    a.write_bytes(b"C")
    os.utime(a, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    current_a = model_identity(request_a)
    assert not first_a.matches(current_a)
    assert not current_a.matches(model_identity(IdentityRequest(
        category="diffusion_models", filename=a.name, loading_config="fp8_e4m3fn",
    )))
    assert sorted(path.name for path in directory.iterdir()) == [a.name, b.name]


def test_http_endpoints_and_browser_independent_idle(tmp_path: Path) -> None:
    async def run() -> None:
        server = PromptServer(asyncio.get_running_loop(), default_asset_manager())
        bridge = QueueBridge(server.prompt_queue, tmp_path / "config.json")
        service = ResourceService(bridge)
        app = web.Application()
        routes = web.RouteTableDef()
        attach(service, app, routes)
        app.add_routes(routes)
        async with TestClient(TestServer(app)) as client:
            status_response = await client.get("/resource-manager/status")
            assert status_response.status == 200
            status = await status_response.json()
            assert status["settings"] == {"unload_gpu_seconds": None, "release_models_seconds": None,
                                          "unload_on_model_switch": True}
            assert not status["memory"]["cuda_initialized"]
            invalid = await client.post("/resource-manager/settings", json={"unload_gpu_seconds": True})
            assert invalid.status == 400
            invalid_json = await client.post(
                "/resource-manager/settings", data="{", headers={"Content-Type": "application/json"},
            )
            assert invalid_json.status == 400
            unsupported = await client.post("/resource-manager/release", json={"action": "cache_only"})
            assert unsupported.status == 409
            missing = await client.delete("/resource-manager/release/999")
            assert missing.status == 404
            release = await client.post("/resource-manager/release", json={"action": "release_models"})
            assert release.status == 202
            operation = await release.json()
            assert operation["state"] == "dispatched"
            assert server.prompt_queue.get_flags() == {"unload_models": True, "free_memory": True}
            await asyncio.sleep(0.3)
            configure = await client.post("/resource-manager/settings", json={
                "unload_gpu_seconds": None, "release_models_seconds": 1, "ignored_extra": "allowed",
            })
            assert configure.status == 200
            # No further HTTP polling is needed for the backend timer to dispatch.
            await asyncio.sleep(1.6)
            assert server.prompt_queue.get_flags() == {"unload_models": True, "free_memory": True}
            assert service.task is not None and not service.task.done()
        assert service.task.cancelled()
    asyncio.run(run())


def test_actual_executor_cache_reset_releases_tensor(tmp_path: Path) -> None:
    async def run() -> None:
        server = PromptServer(asyncio.get_running_loop(), default_asset_manager())
        executor = PromptExecutor(server, cache_type=CacheType.CLASSIC, cache_args={"ram": 0, "ram_inactive": 0})
        prompt = {"1": {"class_type": "EmptyLatentImage", "inputs": {"width": 64, "height": 64, "batch_size": 1}}}
        await executor.execute_async(prompt, "cpu-reference-probe", {}, ["1"])
        assert executor.success
        entry = await executor.caches.outputs.get("1")
        tensor = entry.outputs[0][0]["samples"]
        assert tensor.device.type == "cpu"
        reference = weakref.ref(tensor)
        del entry, tensor
        assert reference() is not None
        executor.reset()
        gc.collect()
        assert reference() is None
        await executor.execute_async(prompt, "cpu-reference-probe-2", {}, ["1"])
        assert executor.success
        assert (await executor.caches.outputs.get("1")).outputs[0][0]["samples"].shape == (1, 4, 8, 8)
    asyncio.run(run())


def test_actual_prompt_worker_handles_flags_and_runs_again(tmp_path: Path) -> None:
    async def run() -> None:
        from main import prompt_worker

        server = PromptServer(asyncio.get_running_loop(), default_asset_manager())
        queue = server.prompt_queue
        bridge = QueueBridge(queue, tmp_path / "config.json")
        worker = threading.Thread(target=prompt_worker, args=(queue, server, default_asset_manager()), daemon=True)
        worker.start()
        prompt = {"1": {"class_type": "EmptyLatentImage", "inputs": {"width": 64, "height": 64, "batch_size": 1}}}

        async def execute(prompt_id: str, sequence: int) -> None:
            queue.put((sequence, prompt_id, prompt, {}, ["1"], {}))
            deadline = time.monotonic() + 15
            while prompt_id not in queue.get_history():
                assert worker.is_alive(), "Isolated ComfyUI worker exited unexpectedly"
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Isolated CPU prompt id={prompt_id} did not finish within 15 seconds.")
                await asyncio.sleep(0.02)
            assert queue.get_history(prompt_id)[prompt_id]["status"]["status_str"] == "success"

        await execute("cpu-before-release", 0)
        for index, action in enumerate(["unload_gpu", "release_models"], start=1):
            bridge.request(ManualRequest(action=action), int(time.monotonic() * 1000))
            deadline = time.monotonic() + 5
            while queue.get_flags(False):
                assert worker.is_alive()
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Native worker did not consume action={action} within 5 seconds.")
                await asyncio.sleep(0.02)
            await execute(f"cpu-after-{action}", index)
        assert worker.is_alive()
        assert not torch.cuda.is_initialized()
    asyncio.run(run())


def test_comfyui_registers_extension_without_workflow_nodes() -> None:
    async def run() -> None:
        server = PromptServer(asyncio.get_running_loop(), default_asset_manager())
        plugin = Path(__file__).resolve().parents[1]
        assert await nodes.load_custom_node(str(plugin), ignore=set(), module_parent="custom_nodes")
        assert any(route.path == "/resource-manager/status" for route in server.routes)
        assert any(Path(directory) == plugin / "resource_manager/web" for directory in nodes.EXTENSION_WEB_DIRS.values())
        assert all(signal for signal in [server.app.on_startup, server.app.on_cleanup])
    asyncio.run(run())


def test_merged_flags_and_competing_consumer_never_imply_completion(tmp_path: Path) -> None:
    async def run() -> None:
        server = PromptServer(asyncio.get_running_loop(), default_asset_manager())
        queue = server.prompt_queue
        bridge = QueueBridge(queue, tmp_path / "config.json")
        operation = bridge.request(ManualRequest(action="unload_gpu"), 0)
        assert operation.state == "dispatched"
        queue.set_flag("unload_models", True)
        queue.set_flag("free_memory", True)
        bridge.tick(1)
        assert bridge.operations[-1].state == "dispatched"
        assert bridge.operations[-1].after is None
        flags = queue.get_flags()
        assert flags == {"unload_models": True, "free_memory": True}
        bridge.tick(2)
        status = bridge.status(3)
        assert status.operations[-1].state == "flags_consumed_unconfirmed"
        assert status.completion_acknowledgement is False
        with pytest.raises(OperationConflictError):
            bridge.cancel(operation.id)
        assert queue.get_flags(False) == {}
        queue.set_flag("free_memory", True)
        waiting = bridge.request(ManualRequest(action="release_models"), 4)
        assert waiting.state == "waiting_for_idle"
        bridge.cancel(waiting.id)
        assert queue.get_flags(False) == {"free_memory": True}
    asyncio.run(run())


def test_task_arrives_after_idle_dispatch_and_native_worker_finishes_it(tmp_path: Path) -> None:
    async def run() -> None:
        from main import prompt_worker

        server = PromptServer(asyncio.get_running_loop(), default_asset_manager())
        queue = server.prompt_queue
        bridge = QueueBridge(queue, tmp_path / "config.json")
        bridge.configure(Settings(unload_gpu_seconds=None, release_models_seconds=1))
        bridge.tick(0)
        prompt = {"1": {"class_type": "EmptyLatentImage", "inputs": {"width": 64, "height": 64, "batch_size": 1}}}
        attempted = threading.Event()
        enqueued = threading.Event()

        def submit() -> None:
            attempted.set()
            queue.put((0, "arrival-after-dispatch", prompt, {}, ["1"], {}))
            enqueued.set()

        with queue.mutex:
            submitter = threading.Thread(target=submit)
            submitter.start()
            if not attempted.wait(timeout=2):
                raise TimeoutError("The isolated submitter did not reach queue.put within 2 seconds.")
            bridge.tick(1000)
            assert queue.flags == {"unload_models": True, "free_memory": True}
            assert not enqueued.is_set()
            assert bridge.operations[-1].state == "dispatched"
            with pytest.raises(OperationConflictError):
                bridge.cancel(bridge.operations[-1].id)
            assert queue.flags == {"unload_models": True, "free_memory": True}
        submitter.join(timeout=2)
        assert enqueued.is_set() and not submitter.is_alive()
        assert queue.get_tasks_remaining() == 1
        worker = threading.Thread(target=prompt_worker, args=(queue, server, default_asset_manager()), daemon=True)
        worker.start()
        deadline = time.monotonic() + 15
        while "arrival-after-dispatch" not in queue.get_history() or queue.get_flags(False):
            assert worker.is_alive(), "The isolated worker failed while processing a task after release dispatch"
            if time.monotonic() >= deadline:
                raise TimeoutError("Task arrival after idle dispatch did not finish within 15 seconds.")
            await asyncio.sleep(0.02)
        assert queue.get_history("arrival-after-dispatch")["arrival-after-dispatch"]["status"]["status_str"] == "success"
        assert not torch.cuda.is_initialized()
        bridge.tick(1001)
        assert bridge.operations[-1].state == "flags_consumed_unconfirmed"
        assert not bridge.status(1002).completion_acknowledgement
    asyncio.run(run())
