import comfy_annotations

# Simply importing your module gives it a chance to add the @ComfyFunc nodes.
import example.example_nodes  # noqa: F401

NODE_CLASS_MAPPINGS = comfy_annotations.NODE_CLASS_MAPPINGS.copy()
NODE_DISPLAY_NAME_MAPPINGS = comfy_annotations.NODE_DISPLAY_NAME_MAPPINGS.copy()

# Do whatever else you need to do to set up any non-ComfyFunc node types here. e.g.:
# NODE_CLASS_MAPPINGS.update(example.example_nodes.NODE_CLASS_MAPPINGS) 
# NODE_DISPLAY_NAME_MAPPINGS.update(example.example_nodes.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
