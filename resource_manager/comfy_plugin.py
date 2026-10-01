"""ComfyUI routes, idle monitor, and worker-side model-switch cache policy.

Automatic model switching retains caches needed by the next executable graph and
releases obsolete entries before any new node loads a model. File identity lookup
remains diagnostic; it does not hash model contents or replace loader semantics.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Literal

import folder_paths
from aiohttp import web
from pydantic import BaseModel, ConfigDict, ValidationError
from server import PromptServer

from ._native import ModelIdentity
from .model_switch import ModelSwitchPolicy

try:
    from comfy_execution.caching import register_cache_retention_policy
except ImportError:
    register_cache_retention_policy = None
from .controller import (
    ManualRequest, OperationConflictError, OperationNotFoundError, QueueBridge,
    Settings, UnsupportedOperationError, clock_ms,
)


class IdentityRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, frozen=True)
    category: Literal["checkpoints", "diffusion_models"]
    filename: str
    loading_config: Literal["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]


def model_identity(request: IdentityRequest) -> ModelIdentity:
    if request.category == "checkpoints" and request.loading_config != "default":
        raise ValueError("CheckpointLoaderSimple only supports loading_config=default.")
    path = Path(folder_paths.get_full_path_or_raise(request.category, request.filename)).resolve(strict=True)
    stat = path.stat()
    return ModelIdentity(str(path), stat.st_size, stat.st_mtime_ns, request.category, request.loading_config)


@web.middleware
async def resource_errors(
    request: web.Request, handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    if not request.path.startswith("/resource-manager/"):
        return await handler(request)
    try:
        return await handler(request)
    except (ValidationError, json.JSONDecodeError, ValueError) as error:
        return web.json_response({"error": str(error), "path": request.path, "status": 400}, status=400)
    except (OperationConflictError, UnsupportedOperationError) as error:
        return web.json_response({"error": str(error), "path": request.path, "status": 409}, status=409)
    except (OperationNotFoundError, FileNotFoundError) as error:
        return web.json_response({"error": str(error), "path": request.path, "status": 404}, status=404)
    except OSError as error:
        logging.exception("Resource manager filesystem operation failed", extra={"path": request.path})
        return web.json_response({"error": str(error), "path": request.path, "status": 500}, status=500)


class ResourceService:
    """Aiohttp connector; failures stop the monitor visibly rather than silently restarting it."""

    def __init__(self, bridge: QueueBridge) -> None:
        self.bridge = bridge
        self.task: asyncio.Task[None] | None = None

    async def monitor(self) -> None:
        while True:
            self.bridge.tick(clock_ms())
            await asyncio.sleep(0.25)

    def monitor_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self.bridge.monitor_error = f"{type(error).__name__}: {error}"
            logging.error(
                "Resource manager monitor stopped", extra={"error": self.bridge.monitor_error},
                exc_info=(type(error), error, error.__traceback__),
            )

    async def start(self, app: web.Application) -> None:
        self.task = asyncio.create_task(self.monitor(), name="resource-manager-idle")
        self.task.add_done_callback(self.monitor_done)

    async def stop(self, app: web.Application) -> None:
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def status(self, request: web.Request) -> web.Response:
        return web.json_response(self.bridge.status(clock_ms()).model_dump())

    async def configure(self, request: web.Request) -> web.Response:
        settings = Settings.model_validate(await request.json())
        return web.json_response(self.bridge.configure(settings).model_dump())

    async def release(self, request: web.Request) -> web.Response:
        body = ManualRequest.model_validate(await request.json())
        return web.json_response(self.bridge.request(body, clock_ms()).model_dump(), status=202)

    async def cancel(self, request: web.Request) -> web.Response:
        return web.json_response(self.bridge.cancel(int(request.match_info["id"])).model_dump())

    async def identity(self, request: web.Request) -> web.Response:
        body = IdentityRequest.model_validate(await request.json())
        identity = model_identity(body)
        return web.json_response({
            "path": identity.path, "size": identity.size, "mtime_ns": identity.mtime_ns,
            "role": identity.role, "loading_config": identity.loading_config,
            "strong_content_hash": False, "managed_switch_supported": self.bridge.managed_switch_supported,
        })


def attach(service: ResourceService, app: web.Application, routes: web.RouteTableDef) -> None:
    app.middlewares.append(resource_errors)
    app.on_startup.append(service.start)
    app.on_cleanup.append(service.stop)
    routes.get("/resource-manager/status")(service.status)
    routes.post("/resource-manager/settings")(service.configure)
    routes.post("/resource-manager/release")(service.release)
    routes.delete("/resource-manager/release/{id}")(service.cancel)
    routes.post("/resource-manager/model-identity")(service.identity)


def install() -> None:
    server = PromptServer.instance
    bridge = QueueBridge(server.prompt_queue, Path(__file__).resolve().parents[1] / "config.json")
    if register_cache_retention_policy is not None:
        register_cache_retention_policy(
            "resource-manager", lambda: ModelSwitchPolicy(lambda: bridge.settings.unload_on_model_switch),
        )
        bridge.managed_switch_supported = True
    else:
        logging.warning("[Resource Manager] Model-switch unloading needs the ComfyUI cache-retention patch.")
    attach(ResourceService(bridge), server.app, server.routes)
