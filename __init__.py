"""ComfyUI extension: model-switch cleanup enabled; idle release is opt-in."""
from .resource_manager.comfy_plugin import install

install()
NODE_CLASS_MAPPINGS = {}
WEB_DIRECTORY = "./resource_manager/web"
