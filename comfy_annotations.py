# This file is deprecated, and provided only for backwards compatibility with the original comfy_annotations module.
import logging

from easy_nodes.easy_nodes import (  # noqa: F401
    Choice,
    ComfyNode,
    ImageTensor,
    NumberInput,
    StringInput,
    create_field_setter_node,
    register_type,
)

logging.warning("comfy_annotations is deprecated. Please use easy_nodes instead.")

ComfyFunc = ComfyNode