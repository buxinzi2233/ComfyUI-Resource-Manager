"""Exercise real HTTP controls against populated native-worker caches, without mocks."""
import asyncio
import json
import logging
import sys
import threading
import time
import weakref
from pathlib import Path

import pytest
import torch
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from app.assets.manager import default_asset_manager
import comfy.model_management as model_management
from comfy.model_patcher import ModelPatcher
from comfy.cli_args import args
from execution import CacheEntry, PromptExecutor
from server import PromptServer

from resource_manager.comfy_plugin import ResourceService, attach
from resource_manager.controller import QueueBridge, clock_ms


def worker_executor(worker: threading.Thread) -> PromptExecutor:
    frame = sys._current_frames()[worker.ident]
    while frame is not None:
        if frame.f_code.co_name == "prompt_worker":
            executor = frame.f_locals["e"]
            assert isinstance(executor, PromptExecutor)
            return executor
        frame = frame.f_back
    raise RuntimeError("Isolated native worker has no prompt_worker frame.")


@pytest.mark.parametrize("source", ["manual", "idle"])
def test_http_controls_release_real_worker_model_and_cache(tmp_path: Path, source: str) -> None:
    async def run() -> None:
        from main import prompt_worker

        server = PromptServer(asyncio.get_running_loop(), default_asset_manager())
        queue = server.prompt_queue
        bridge = QueueBridge(queue, tmp_path / "config.json")
        app = web.Application()
        routes = web.RouteTableDef()
        attach(ResourceService(bridge), app, routes)
        app.add_routes(routes)
        worker = threading.Thread(target=prompt_worker, args=(queue, server, default_asset_manager()), daemon=True)
        worker.start()
        prompt = {"1": {"class_type": "EmptyLatentImage", "inputs": {"width": 64, "height": 64, "batch_size": 1}}}

        async def execute(prompt_id: str) -> None:
            queue.put((0, prompt_id, prompt, {}, ["1"], {}))
            deadline = time.monotonic() + 15
            while prompt_id not in queue.get_history():
                assert worker.is_alive()
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Real worker did not finish prompt={prompt_id} within 15 seconds.")
                await asyncio.sleep(0.01)
            assert queue.get_history(prompt_id)[prompt_id]["status"]["status_str"] == "success"

        async def wait_for_release(action: str) -> None:
            deadline = time.monotonic() + 5
            while (model_management.current_loaded_models if action == "unload_gpu" else tensor_ref() is not None):
                assert worker.is_alive()
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Real resource release action={action}, source={source} did not occur.")
                await asyncio.sleep(0.01)

        evidence: list[dict[str, str | int | bool | None]] = []
        async with TestClient(TestServer(app)) as client:
            await execute(f"populated-{source}")
            executor = worker_executor(worker)
            entry = await executor.caches.outputs.get("1")
            assert entry is not None
            device = torch.device("cpu" if args.cpu else "cuda:0")
            model = torch.nn.Linear(32, 32)
            patcher = ModelPatcher(model, device, torch.device("cpu"), 0, False, False)
            model_management.load_models_gpu([patcher], 0, False, None, True)
            tensor = torch.ones(2 * 1024 * 1024, device=device)
            tensor_ref = weakref.ref(tensor)
            model_ref = weakref.ref(model)
            patcher_ref = weakref.ref(patcher)
            await executor.caches.outputs.set("1", CacheEntry(
                ui=entry.ui, outputs=[[{"samples": entry.outputs[0][0]["samples"],
                                       "probe_model": patcher, "probe_tensor": tensor}]],
            ))
            del entry, model, patcher, tensor
            assert len(model_management.current_loaded_models) == 1
            assert tensor_ref() is not None and model_ref() is not None
            assert patcher_ref().loaded_size() == 4224

            for action in ("unload_gpu", "release_models"):
                if source == "manual":
                    response = await client.post("/resource-manager/release", json={"action": action})
                    assert response.status == 202
                else:
                    response = await client.post("/resource-manager/settings", json={
                        "unload_gpu_seconds": 1 if action == "unload_gpu" else None,
                        "release_models_seconds": 1 if action == "release_models" else None,
                    })
                    assert response.status == 200
                    now = clock_ms()
                    bridge.tick(now)
                    bridge.tick(now + 1001)
                await wait_for_release(action)
                if action == "unload_gpu":
                    assert patcher_ref() is not None and patcher_ref().loaded_size() == 0
                    assert tensor_ref() is not None and model_ref() is not None
                else:
                    assert patcher_ref() is None and model_ref() is None and tensor_ref() is None
                now = clock_ms()
                bridge.tick(now)
                observed = bridge.operations[-1]
                assert observed.state == "flags_consumed_unconfirmed"
                assert observed.after is not None and observed.after.sampled_at_ms == now
                bridge.tick(now + 1)
                assert bridge.operations[-1].after.sampled_at_ms == now + 1
                evidence.append({"source": source, "device": str(device), "action": action,
                                  "registered_models": len(model_management.current_loaded_models),
                                  "model_reference_alive": model_ref() is not None,
                                  "cached_8MiB_tensor_alive": tensor_ref() is not None,
                                  "cuda_allocated_bytes": torch.cuda.memory_allocated() if device.type == "cuda" else None})
                logging.info("Resource release evidence", extra={"evidence": evidence[-1]})
            last_sample = bridge.operations[-1].after
            await execute(f"after-real-release-{source}")
            bridge.tick(clock_ms())
            assert bridge.operations[-1].after == last_sample
            assert worker.is_alive()
            evidence.append({"source": source, "device": str(device), "subsequent_prompt": "success"})
            mode = "cpu" if args.cpu else "gpu"
            destination = Path(__file__).resolve().parents[1] / "dist" / f"release-evidence-{mode}-{source}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    asyncio.run(run())
