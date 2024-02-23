# ComfyUI Node-Creating Decorator

This module provides an annotation-ingesting decorator, @ComfyFunc, to streamline adding custom node types in [ComfyUI](https://github.com/comfyanonymous/ComfyUI). It processes your function's signature to create a wrapped function and custom node definition required for ComfyUI, eliminating all the boilerplate code. In most cases you can just add a @ComfyFunc("category") decorator to your existing function.

```
from comfy_annotations import ComfyFunc, ImageTensor, MaskTensor

@ComfyFunc("Example category")
def mask_image(image: ImageTensor, mask: MaskTensor) -> ImageTensor:
    """Applies a mask to an image."""
    return image * mask

NODE_CLASS_MAPPINGS.update(comfy_annotations.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS(comfy_annotations.NODE_DISPLAY_NAME_MAPPINGS)
```

That's it! Now your operation is ready for ComfyUI. More example definitions can be found in [example/example_nodes.py](example/example_nodes.py).

## Features

- **@ComfyFunc Decorator**: Simplifies the declaration of custom nodes with automagic node declaration based on Python type annotations. Existing Python functions can be converted to ComfyUI nodes with a simple "@ComfyFunc()"
- **Type Support**: Includes several custom types (`ImageTensor`, `MaskTensor`, `NumberInput`, etc.) to facilitate specific UI controls like sliders, choices, and text inputs.
- **Dual purpose**: @ComfyFunc-decorated functions remain regular Python functions too.
- **Automatic list and tuple handling**: Simply annotate the type as e.g. ```list[torch.Tensor]``` and your function will automatically make sure you get passed a list. It will also auto-tuple your return value for you internally (or leave it alone if you just want to copy your existing code).
- **Supports most ComfyUI node definition features**: validate_input, is_output_node, etc can be specified as parameters to the ComfyFunc decorator.

## Installation

To use this module in your ComfyUI project, follow these steps:

1. **Install the Module**: Run the following command to install the ComfyUI Annotations module:

    ```bash
    pip install git+https://github.com/andrewharp/ComfyUI-Annotations.git
    ```
    
2. **Integrate into Your Project**:
    - Open your ComfyUI project's `__init__.py` (e.g. `ComfyUI/custom_nodes/my_project/__init__.py`).
    - Edit to make your project export the ComfyFunc nodes as shown below:

    ```python
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

    # Add the @ComfyFunc nodes.
    import example.example_nodes
    import comfy_annotations
    NODE_CLASS_MAPPINGS.update(comfy_annotations.NODE_CLASS_MAPPINGS) 
    NODE_DISPLAY_NAME_MAPPINGS.update(comfy_annotations.NODE_DISPLAY_NAME_MAPPINGS)

    # Set up any non-ComfyFunc node types as needed.
    # NODE_CLASS_MAPPINGS.update(example.example_nodes.NODE_CLASS_MAPPINGS) 
    # NODE_DISPLAY_NAME_MAPPINGS.update(example.example_nodes.NODE_DISPLAY_NAME_MAPPINGS)

    # Export so that ComfyUI can pick them up.
    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
    ```


## Usage

1. **Annotate Functions with @ComfyFunc**: Decorate your processing functions with `@ComfyFunc`. The decorator accepts the following parameters:
   - `category`: Specifies the category under which the node will be listed in ComfyUI. Default is `"ComfyFunc"`.
   - `display_name`: Optionally specifies a human-readable name for the node as it will appear in ComfyUI. If not provided, a name is generated based on the function name.
   - `workflow_name`: The internal unique identifier for this node type. If not provided, a name is generated based on the function name.
   - `is_output_node`: maps to ComfyUI's IS_OUTPUT_NODE
   - `validate_inputs`: maps to ComfyUI's VALIDATE_INPUTS
   - `is_changed`: maps to ComfyUI's IS_CHANGED
   - `debug`: A boolean that makes this operator print out extra information during its lifecycle.

    Example:
    ```python
    from comfy_annotations import ComfyFunc, ImageTensor, NumberInput

    @ComfyFunc(category="Image Processing",
               display_name="Enhance Image",
               is_output_node=True,  # ComfyUI will always run this node
               validate_inputs=lambda factor: return factor > 0,
               debug=True)
    def enhance_image(image: ImageTensor, factor: NumberInput(0.5, 0, 1, 0.1)) -> ImageTensor:
        # Function implementation
    ```

2. **Annotate your function inputs and outputs**: Fully annotate function parameters and return types, using `list` to wrap types as appropriate. This information is used to generate the fields of the internal class definition `@ComfyFunc` sends to ComfyUI. If you don't annotate the inputs, the input will be treated as a wildcard. If you don't annotate the output, you won't see anything at all in ComfyUI.

    Example:
    ```python
    @ComfyFunc("Utilities")
    def add_value(the_list: list[ImageTensor], val: int) -> list[int]:
        return [img + the_value for img in the_list]
    ```

### Example Node Definition from ComfyUI's [example_node.py.example](https://github.com/comfyanonymous/ComfyUI/blob/master/custom_nodes/example_node.py.example), converted:

```python
from comfy_annotations import ComfyFunc, ImageTensor, MaskTensor, NumberInput, Choice, StringInput

@ComfyFunc("Example")
def annotated_example(image: ImageTensor, 
                      string_field: str = StringInput("Hello World!", multiline=False),
                      int_field: int = NumberInput(0, 0, 4096, 64, "number"), 
                      float_field: float = NumberInput(1.0, 0, 10.0, 0.01, 0.001),
                      print_to_screen: str = Choice(["enabled", "disabled"])) -> ImageTensor:
    """Inverts the input image and prints input parameters based on `print_to_screen` choice."""
    if print_to_screen == "enable":
        print(f"""Your input contains:
            string_field: {string_field}
            int_field: {int_field}
            float_field: {float_field}
        """)
    return 1.0 - image
```

More examples can be found in [example_nodes.py](https://github.com/andrewharp/ComfyUI-Annotations/blob/main/example/example_nodes.py)


## Contributing

Contributions are welcome! Please submit pull requests or open issues for any bugs, features, or improvements.
