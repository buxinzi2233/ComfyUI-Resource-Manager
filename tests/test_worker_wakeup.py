"""Release requests must survive a notification sent before the worker waits."""
import asyncio
import threading
import time
from pathlib import Path

import pytest
from app.assets.manager import default_asset_manager
from server import PromptServer

from resource_manager.controller import ManualRequest, QueueBridge, Settings


@pytest.mark.parametrize("source", ["manual", "idle"])
def test_worker_handles_release_posted_before_wait(tmp_path: Path, source: str) -> None:
    async def run() -> None:
        from main import prompt_worker

        server = PromptServer(asyncio.get_running_loop(), default_asset_manager())
        queue = server.prompt_queue
        bridge = QueueBridge(queue, tmp_path / "config.json")
        if source == "manual":
            bridge.request(ManualRequest(action="release_models"), 0)
        else:
            bridge.configure(Settings(unload_gpu_seconds=None, release_models_seconds=1))
            bridge.tick(0)
            bridge.tick(1000)
        assert queue.get_flags(False) == {"unload_models": True, "free_memory": True}
        worker = threading.Thread(target=prompt_worker, args=(queue, server, default_asset_manager()), daemon=True)
        worker.start()
        deadline = time.monotonic() + 2
        try:
            while queue.get_flags(False) and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            assert worker.is_alive()
            assert queue.get_flags(False) == {}, "Worker slept despite a pending release request"
        finally:
            # Wake a failing worker too, so it does not retain the test's flags.
            with queue.not_empty:
                queue.not_empty.notify()
    asyncio.run(run())
