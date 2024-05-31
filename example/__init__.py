# Simply importing your module gives it a chance to add the @ComfyFunc nodes since
# EasyNodes will automatically export the NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS
# for you.
from .example_nodes import *


# Alternatively, to export yourself, you can do the following:
# import easy_nodes
# easy_nodes.init(auto_register=False)
# import example.example_nodes
# NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = easy_nodes.get_node_mappings()
# __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]