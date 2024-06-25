from easy_nodes.easy_nodes import (  # noqa: F401
    AnyType,
    Choice,
    ComfyNode,
    ImageTensor,
    MaskTensor,
    NumberInput,
    StringInput,
    show_image,
    show_text,
    create_field_setter_node,
    initialize_easy_nodes,
    register_type,
    get_node_mappings
)

# For backwards compatibility with the original comfy_annotations module.
ComfyFunc = ComfyNode
