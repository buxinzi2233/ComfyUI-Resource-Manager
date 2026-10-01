import asyncio
import gc
import weakref
from pathlib import Path

import nodes
import pytest
import torch
from app.assets.manager import default_asset_manager
from comfy.model_patcher import ModelPatcher
from execution import CacheType, PromptExecutor
from server import PromptServer

from resource_manager.controller import Settings, read_settings, write_settings
from resource_manager.model_switch import ModelSwitchPolicy


@pytest.fixture
def loaders(monkeypatch):
    references = {}
    loads = []
    snapshots = []

    class Loader(nodes.UNETLoader):
        def load_unet(self, unet_name, weight_dtype):
            snapshots.append((unet_name, {name for name, ref in references.items() if ref() is not None}))
            model = torch.nn.Linear(16, 16)
            patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"), 0, False, False)
            references[unet_name] = weakref.ref(model)
            loads.append(unet_name)
            return (patcher,)

    class Clone:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"model": ("MODEL",)}}
        RETURN_TYPES = ("MODEL",)
        FUNCTION = "clone"
        def clone(self, model):
            return (model.clone(),)

    class Use:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"model": ("MODEL",), "run": ("INT", {"default": 0})}}
        RETURN_TYPES = ()
        FUNCTION = "use"
        OUTPUT_NODE = True
        def use(self, model, run):
            assert model.model.weight.shape == (16, 16)
            return ()

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "UNETLoader", Loader)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "SwitchTestClone", Clone)
    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "SwitchTestUse", Use)
    yield references, loads, snapshots
    gc.collect()


def graph(*names, run=0):
    prompt = {}
    outputs = []
    for index, name in enumerate(names):
        loader, clone, sink = str(index * 3), str(index * 3 + 1), str(index * 3 + 2)
        prompt[loader] = {"class_type": "UNETLoader", "inputs": {"unet_name": name, "weight_dtype": "default"}}
        prompt[clone] = {"class_type": "SwitchTestClone", "inputs": {"model": [loader, 0]}}
        prompt[sink] = {"class_type": "SwitchTestUse", "inputs": {"model": [clone, 0], "run": run}}
        outputs.append(sink)
    return prompt, outputs


def executor(cache_type, enabled=lambda: True):
    server = PromptServer(asyncio.get_running_loop(), default_asset_manager())
    result = PromptExecutor(server, cache_type, {"lru": 100, "ram": 0, "ram_inactive": 0})
    result.cache_retention_policies = [ModelSwitchPolicy(enabled)]
    return result


@pytest.mark.parametrize("cache_type", [CacheType.CLASSIC, CacheType.LRU, CacheType.RAM_PRESSURE])
def test_switch_releases_old_model_and_clones_before_new_load(loaders, cache_type):
    async def run():
        refs, loads, snapshots = loaders
        e = executor(cache_type)
        for index, name in enumerate(["A", "A", "B", "A"]):
            prompt, outputs = graph(name, run=index)
            await e.execute_async(prompt, str(index), {}, outputs)
            assert e.success
        assert loads == ["A", "B", "A"]
        assert snapshots == [("A", set()), ("B", set()), ("A", set())]
        assert refs["B"]() is None
        assert refs["A"]() is not None
        if cache_type != CacheType.CLASSIC:
            assert len(e.caches.outputs.used_generation) == 3
        if cache_type == CacheType.RAM_PRESSURE:
            assert len(e.caches.outputs.timestamps) == 3
    asyncio.run(run())


@pytest.mark.parametrize("cache_type", [CacheType.CLASSIC, CacheType.LRU, CacheType.RAM_PRESSURE])
def test_switch_preserves_other_models_needed_by_current_graph(loaders, cache_type):
    async def run():
        refs, loads, snapshots = loaders
        e = executor(cache_type)
        prompt, outputs = graph("A", "B")
        await e.execute_async(prompt, "ab", {}, outputs)
        original_a = refs["A"]
        prompt, outputs = graph("A", "C", run=1)
        await e.execute_async(prompt, "ac", {}, outputs)
        assert e.success
        assert loads == ["A", "B", "C"]
        assert snapshots[-1] == ("C", {"A"})
        assert refs["A"] is original_a and original_a() is not None
        assert refs["B"]() is None
    asyncio.run(run())


def test_disconnected_old_loader_is_not_kept_alive(loaders):
    async def run():
        refs, _, snapshots = loaders
        e = executor(CacheType.RAM_PRESSURE)
        prompt, outputs = graph("A", "B")
        await e.execute_async(prompt, "ab", {}, outputs)
        prompt, outputs = graph("A", "C", run=1)
        await e.execute_async(prompt, "c-only", {}, outputs[1:])
        assert e.success
        assert snapshots[-1] == ("C", set())
        assert refs["A"]() is refs["B"]() is None
    asyncio.run(run())


def test_disabled_then_enabled_cleans_retained_old_models(loaders):
    async def run():
        refs, loads, _ = loaders
        enabled = False
        e = executor(CacheType.RAM_PRESSURE, lambda: enabled)
        for name in ("A", "B"):
            prompt, outputs = graph(name)
            await e.execute_async(prompt, name, {}, outputs)
            assert e.success
        assert refs["A"]() is not None and refs["B"]() is not None
        enabled = True
        prompt, outputs = graph("B", run=1)
        await e.execute_async(prompt, "enabled", {}, outputs)
        assert e.success
        assert refs["A"]() is None and refs["B"]() is not None
        assert loads == ["A", "B"]
    asyncio.run(run())


def test_switch_with_cache_disabled(loaders):
    async def run():
        e = executor(CacheType.NONE)
        for name in ("A", "B"):
            prompt, outputs = graph(name)
            await e.execute_async(prompt, name, {}, outputs)
            assert e.success
    asyncio.run(run())


def test_computed_selection_does_not_guess_which_models_to_release():
    policy = ModelSwitchPolicy(lambda: True)
    prompt, outputs = graph("A")
    assert policy(prompt, outputs) is None
    prompt["0"]["inputs"]["unet_name"] = ["dynamic", 0]
    prompt["dynamic"] = {"class_type": "DynamicName", "inputs": {}}
    assert policy(prompt, outputs) is None


@pytest.mark.parametrize("class_type,inputs", [
    ("CheckpointLoaderSimple", {"ckpt_name": "A"}),
    ("CheckpointLoader", {"ckpt_name": "A", "config_name": "v1.yaml"}),
    ("CLIPLoader", {"clip_name": "A", "type": "stable_diffusion"}),
    ("DualCLIPLoader", {"clip_name1": "A", "clip_name2": "shared", "type": "sdxl"}),
    ("VAELoader", {"vae_name": "A"}),
])
def test_supported_loader_selections(class_type, inputs):
    policy = ModelSwitchPolicy(lambda: True)
    prompt = {"1": {"class_type": class_type, "inputs": dict(inputs)}}
    assert policy(prompt, ["1"]) is None
    field = next(iter(inputs))
    prompt["1"]["inputs"][field] = "B"
    assert policy(prompt, ["1"]) == {"1"}


def test_old_settings_enable_switching_and_explicit_disable_persists(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text('{"unload_gpu_seconds": null, "release_models_seconds": null}')
    assert read_settings(path).unload_on_model_switch
    settings = Settings(unload_gpu_seconds=None, release_models_seconds=None, unload_on_model_switch=False)
    write_settings(path, settings)
    assert read_settings(path) == settings
