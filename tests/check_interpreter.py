"""Run with the installation's Python; register the extension without starting a server."""
import asyncio
import importlib.metadata
import json
import os
import sys
import sysconfig
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
PLUGIN = Path(__file__).resolve().parents[1]
CORE = PLUGIN.parents[1]
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(PLUGIN))


async def check_registration() -> None:
    from app.assets.manager import default_asset_manager
    import nodes
    import torch
    from server import PromptServer
    from resource_manager import _native

    expected_native = PLUGIN / f"resource_manager/_native{sysconfig.get_config_var('EXT_SUFFIX')}"
    if Path(_native.__file__).resolve() != expected_native:
        raise RuntimeError(f"Expected the in-place extension at {expected_native}; loaded {_native.__file__}.")
    identity = _native.ModelIdentity("/cpu-diagnostic", 1, 1, "diffusion_models", "default")
    if not identity.matches(identity):
        raise RuntimeError("The in-place Rust extension failed its identity call.")
    server = PromptServer(asyncio.get_running_loop(), default_asset_manager())
    registered = await nodes.load_custom_node(str(PLUGIN), ignore=set(), module_parent="custom_nodes")
    resource_routes = sorted(route.path for route in server.routes if route.path.startswith("/resource-manager/"))
    expected_routes = {
        "/resource-manager/status", "/resource-manager/settings", "/resource-manager/release",
        "/resource-manager/release/{id}", "/resource-manager/model-identity",
    }
    web_directories = sorted(
        str(Path(directory).resolve()) for directory in nodes.EXTENSION_WEB_DIRS.values()
        if Path(directory).resolve() == PLUGIN / "resource_manager/web"
    )
    if not registered or set(resource_routes) != expected_routes or not web_directories:
        raise RuntimeError(f"Registration failed: loaded={registered}, routes={resource_routes}, web={web_directories}.")
    if torch.cuda.is_initialized():
        raise RuntimeError("The isolated registration check unexpectedly initialized CUDA.")
    payload = {
        "sys_executable": sys.executable,
        "resolved_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "soabi": sysconfig.get_config_var("SOABI"),
        "native_path": _native.__file__,
        "native_call_passed": True,
        "plugin_registered": registered,
        "resource_routes": resource_routes,
        "web_directories": web_directories,
        "cuda_initialized": False,
        "server_started": False,
        "network_requests_sent": False,
        "runtime_dependencies": {
            name: importlib.metadata.version(name) for name in ("torch", "pydantic", "aiohttp", "psutil")
        },
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    (PLUGIN / ".cache").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="actual-interpreter-", dir=PLUGIN / ".cache") as sandbox:
        for name in ("user", "temp"):
            (Path(sandbox) / name).mkdir()
        sys.argv = [
            "resource-manager-actual-interpreter-check", "--cpu", "--disable-dynamic-vram",
            "--disable-all-custom-nodes", "--base-directory", sandbox,
            "--user-directory", str(Path(sandbox) / "user"), "--temp-directory", str(Path(sandbox) / "temp"),
        ]
        import comfy.options

        comfy.options.enable_args_parsing(True)
        import folder_paths

        folder_paths.set_user_directory(str(Path(sandbox) / "user"))
        folder_paths.set_temp_directory(str(Path(sandbox) / "temp"))
        asyncio.run(check_registration())
