"""Select live cache entries before execution; never load or unload from HTTP handlers."""
from __future__ import annotations

import logging
from collections.abc import Callable

from comfy_execution.graph_utils import is_link

LOADER_TYPES = {
    "CheckpointLoaderSimple", "CheckpointLoader", "UNETLoader",
    "CLIPLoader", "DualCLIPLoader", "VAELoader",
}


class ModelSwitchPolicy:
    def __init__(self, enabled: Callable[[], bool]) -> None:
        self.enabled = enabled
        self.selections: set[tuple] = set()

    def __call__(self, prompt: dict, output_ids: list[str]) -> set[str] | None:
        active: set[str] = set()
        pending = list(output_ids)
        while pending:
            node_id = pending.pop()
            if node_id in active or node_id not in prompt:
                continue
            active.add(node_id)
            pending.extend(value[0] for value in prompt[node_id]["inputs"].values() if is_link(value))

        selections = set()
        for node_id in active:
            node = prompt[node_id]
            if node["class_type"] not in LOADER_TYPES:
                continue
            inputs = node["inputs"]
            # Computed selections are not known until execution. Keep existing caches
            # rather than guess which of their models a later node will need.
            if any(not isinstance(value, (str, int, float, bool, type(None))) for value in inputs.values()):
                return None
            selections.add((node["class_type"], tuple(sorted(inputs.items()))))

        obsolete = self.selections - selections
        if not self.enabled():
            self.selections.update(selections)
            return None
        self.selections = selections
        if not obsolete:
            return None
        logging.info("[Resource Manager] Model selection changed: releasing obsolete caches before loading (%d previous selections).", len(obsolete))
        return active
