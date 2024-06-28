import easy_nodes
easy_nodes.initialize_easy_nodes(default_category="EasyNodes Examples", auto_register=False)

# Simply importing your module gives the ComfyNode decorator a chance to register your nodes.
from .example_nodes import *  # noqa: F403, E402

NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = easy_nodes.get_node_mappings()
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]