"""Use the real ComfyUI dependencies in a CPU-only, isolated filesystem context."""
import os
import sys
import tempfile
from pathlib import Path

GPU_TEST = os.environ.get("RESOURCE_MANAGER_TEST_GPU") == "1"
if not GPU_TEST:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
PLUGIN = Path(__file__).resolve().parents[1]
CORE = PLUGIN.parents[1]
CORE_SITE = next((path for path in [CORE / ".venv/Lib/site-packages", *sorted((CORE / ".venv/lib").glob("python*/site-packages"))] if path.is_dir()), None)
if CORE_SITE is not None:
    sys.path.insert(0, str(CORE_SITE))
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(PLUGIN))
(PLUGIN / ".cache").mkdir(exist_ok=True)
SANDBOX = tempfile.TemporaryDirectory(prefix="cpu-integration-", dir=PLUGIN / ".cache")
for name in ("user", "temp"):
    (Path(SANDBOX.name) / name).mkdir()
original_argv = sys.argv
sys.argv = [
    "resource-manager-tests", *([] if GPU_TEST else ["--cpu"]), "--disable-dynamic-vram", "--disable-all-custom-nodes",
    "--base-directory", SANDBOX.name, "--user-directory", str(Path(SANDBOX.name) / "user"),
    "--temp-directory", str(Path(SANDBOX.name) / "temp"),
]
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing(True)
from comfy.cli_args import args  # noqa: E402

assert args.cpu == (not GPU_TEST)
sys.argv = original_argv
import folder_paths  # noqa: E402

folder_paths.set_user_directory(str(Path(SANDBOX.name) / "user"))
folder_paths.set_temp_directory(str(Path(SANDBOX.name) / "temp"))
