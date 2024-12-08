from easy_nodes.comfy_types import (  # noqa: F401
    Color,
    ConditioningTensor,
    ImageTensor,
    LatentTensor,
    MaskTensor,
    ModelTensor,
    NumberType,
    PhotoMaker,
    SigmasTensor,
)
from easy_nodes.easy_nodes import (  # noqa: F401
    AnyType,
    AutoDescriptionMode,
    BoolInput,
    CheckSeverityMode,
    Choice,
    ComfyNode,
    CustomVerifier,
    NumberInput,
    StringInput,
    TensorVerifier,
    TypeVerifier,
    create_field_setter_node,
    get_node_mappings,
    initialize_easy_nodes,
    register_type,
    save_node_list,
    show_image,
    show_text,
)

# For backwards compatibility with the original comfy_annotations module.
ComfyFunc = ComfyNode
