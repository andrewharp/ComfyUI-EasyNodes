NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Add the @ComfyFunc nodes.
import example.example_nodes
import comfy_annotations
NODE_CLASS_MAPPINGS.update(comfy_annotations.NODE_CLASS_MAPPINGS) 
NODE_DISPLAY_NAME_MAPPINGS.update(comfy_annotations.NODE_DISPLAY_NAME_MAPPINGS)

# Do whatever else you need to do to set up your non-ComfyFunc node types here. e.g.:
# NODE_CLASS_MAPPINGS.update(example.example_nodes.NODE_CLASS_MAPPINGS) 
# NODE_DISPLAY_NAME_MAPPINGS.update(example.example_nodes.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']