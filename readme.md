# Effortless Nodes for ComfyUI

This package aims to make adding new [ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes as easy as possible, and to provide functionality through pure Python that was previously only accessible with custom JavaScript.

It processes your function's Python signature to create the node definition ComfyUI is expecting. All you have to do is annotate your inputs and outputs and add the `@ComfyNode` decorator.

For example:
```python
from easy_nodes import ComfyNode, ImageTensor, MaskTensor, NumberInput

@ComfyNode("Example category", color="#0066cc", bg_color="#ffcc00", return_names=["Below", "Above"])
def threshold_image(image: ImageTensor,
                    threshold: float = NumberInput(0.5, 0, 1, 0.01, display="slider")) -> tuple[MaskTensor, MaskTensor]:
    """Returns separate masks for values above and below the threshold value."""
    mask_below = torch.any(image < threshold, dim=-1)
    return mask_below.float(), (~mask_below).float()
```

That's it! Now your node is ready for ComfyUI. More examples can be found [here](example/example_nodes.py).

Sample node with tooltip and deep source link:

<img src="assets/threshold_example.png" alt="The new node with tooltip" width="50%">

New settings:

<img src="assets/menu_options.png" alt="New menu options" width="50%">


Note that ImageTensor/MaskTensor are just syntactic sugar for semantically differentiating the annotations (allowing ComfyUI to know what plugs into what); your function will still get passed genunine torch.Tensor objects.

For more control, you can call [easy_nodes.initialize_easy_nodes(...)](https://github.com/andrewharp/ComfyUI-EasyNodes?tab=readme-ov-file#initialization-options) before creating nodes and and turn on some advanced settings that will apply to all nodes you create.

## New in 1.1:

- Custom verifiers for types on input and output for your nodes. For example, it will automatically verify that images always have 1, 3 or 4 channels (B&W, RGB and RGBA). Set `verify_level` when calling initialize_easy_nodes to either CheckSeverityMode OFF, WARN, or FATAL (default is WARN). You can write your own verifiers. See [comfy_types.py](comfy_types.py) for examples of types with verifiers.
- Expanded ComfyUI type support. See [comfy_types.py](comfy_types.py) for the full list of registered types.
- Added warnings if relying on node auto-registration without explicitly asking for it (while also supporting get_node_mappings() at the same time). This is because the default for auto_register will change to False in a future release, in order to make ComfyUI-EasyNodes more easily findable by indexers like ComfyUI-Manager, which expect your nodes to be found in your `__init__.py`. You can enable auto-registration explicitly with `easy_nodes.initialize_easy_nodes(auto_register=True)`.

## New in 1.0:

- Renamed to ComfyUI-EasyNodes from ComfyUI-Annotations to better reflect the package's goal (rather than the means)
  - Package is now `easy_nodes` rather than `comfy_annotations`
- Now on pip/PyPI! ```pip install ComfyUI-EasyNodes```
- Set node foreground and background color via Python argument, no JS required: `@ComfyNode(color="FF0000", bg_color="00FF00")`
- Add previews to nodes without JavaScript. Just drop either of these in the body of your node's function:
  - `easy_nodes.show_text("hello world")`
  - `easy_nodes.show_image(image)`
- Automatically create nodes from existing Python classes. The dynamic node will automatically add a widget for every field.
- Info tooltip on nodes auto-generated from your function's docstring
- New optional settings features:
  - Make images persist across browser refreshes via a settings option (provided they're still on the server)
  - Automatic module reloading: if you turn on the setting, immediately see the changes to code on the next run.
  - LLM-based debugging: optionally have ChatGPT take a crack at fixing your code
  - Deep links to source code if you set a base source path (e.g. to github or your IDE)
- Bug fixes

## Features

- **@ComfyNode Decorator**: Simplifies the declaration of custom nodes with automagic node declaration based on Python type annotations. Existing Python functions can be converted to ComfyUI nodes with a simple "@ComfyNode()"
- **Built-in text and image previews**: Just call `easy_nodes.add_preview_text()` and `easy_nodes.add_preview_image()` in the body of your function and EasyNodes will automatically display it, no JavaScript hacking required.
- **Set node color easily**: No messing with JavaScript, just tell the decorator what color you want the node to be.
- **Type Support**: Includes several custom types (`ImageTensor`, `MaskTensor`, `NumberInput`, `Choice`, etc.) to support ComfyUI's connection semantics and UI functionality. Register additional types with `register_type`.
- **Automatic list and tuple handling**: Simply annotate the type as e.g. ```list[torch.Tensor]``` and your function will automatically make sure you get passed a list. It will also auto-tuple your return value for you internally (or leave it alone if you just want to copy your existing code).
- **Init-time checking**: Less scratching your head when your node doesn't fire off properly later. For example, if you copy-paste a node definition and forget to rename it, @ComfyNode will alert you immediately about duplicate nodes rather than simply overwriting the earlier definition.
- **Supports most ComfyUI node definition features**: validate_input, is_output_node, etc can be specified as parameters to the ComfyNode decorator.
- **Convert existing data classes to ComfyUI nodes**: pass `create_field_setter_node` a type, and it will automatically create a new node type with widgets to set all the fields.
- **LLM-based debugging**: Optional debugging and auto-fixing of exceptions during node execution. Will automatically create a prompt with the relevent context and send it to ChatGPT, create a patch and fix your code.


## Installation

To use this module in your ComfyUI project, follow these steps:

1. **Install the Module**: Run the following command to install the ComfyUI-EasyNodes module:

    ```bash
    pip install ComfyUI-EasyNodes
    ```
    or, if you want to have an editable version:
    ```bash
    git clone https://github.com/andrewharp/ComfyUI-EasyNodes
    pip install -e ComfyUI-EasyNodes
    ```
    Note that this is not a typical ComfyUI nodepack, so does not itself live under custom_nodes.
    
    However, after installing you can copy the example node directory into custom_nodes to test them out:
    ```bash
    git clone --depth=1 https://github.com/andrewharp/ComfyUI-EasyNodes.git /tmp/easynodes
    mv /tmp/easynodes/example $COMFYUI_DIR/custom_nodes/easynodes
    ```

3. **Integrate into Your Project**:
    In `__init__.py`:

    ```python
    import easy_nodes
    easy_nodes.initialize_easy_nodes(default_category=my_category, auto_register=False)

    # This must come after calling initialize_easy_nodes.
    import your_node_module  # noqa: E402

    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = easy_nodes.get_node_mappings()

    # Export so that ComfyUI can pick them up.
    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
    ```

    You can also initialize with auto_register=True, in which can you won't have to do anything else after the import. However, this may be problematic for having your nodes indexed so will default to False in a future update (currently not setting it explicitly will auto-register and complain).


## Initialization options

The options passed to `easy_nodes.initialize_easy_nodes` will apply to all nodes registered until the next time `easy_nodes.initialize_easy_nodes` is called.

The settings mostly control defaults and some optional features that I find nice to have, but which may not work for everybody, so some are turned off by default.

- `default_category`: The default category for nodes. Defaults to "EasyNodes".
- `auto_register`: Whether to automatically register nodes with ComfyUI (so you don't have to export). Previously defaulted to True; now defaults to half-true (will auto-register, allow you to export, and print a warning). In a future release will default to False.
- `docstring_mode`: The mode for generating node descriptions that show up in tooltips. Defaults to AutoDescriptionMode.FULL.
- `verify_tensors`: Whether to verify tensors for shape and data type according to ComfyUI type (MASK, IMAGE, etc). Runs on inputs and outputs. Defaults to False, as I've made some assumptions about shapes that may not be universal.
- `auto_move_tensors`: Whether to automatically move torch Tensors to the GPU before your function gets called, and then to the CPU on output. Defaults to False.


## Using the decorator

1. **Annotate Functions with @ComfyNode**: Decorate your processing functions with `@ComfyNode`. The decorator accepts the following parameters:
   - `category`: Specifies the category under which the node will be listed in ComfyUI. Default is `"ComfyNode"`.
   - `display_name`: Optionally specifies a human-readable name for the node as it will appear in ComfyUI. If not provided, a name is generated based on the function name.
   - `workflow_name`: The internal unique identifier for this node type. If not provided, a name is generated based on the function name.
   - `description`: An optional description for the node. If not provided the function's docstring, if any, will be used according to `easy_nodes.docstring_mode`.
   - `is_output_node`: Maps to ComfyUI's IS_OUTPUT_NODE.
   - `return_types`: Maps to ComfyUI's RETURN_TYPES. Use if the return type of the function itself is dynamic.
   - `return_names`: Maps to ComfyUI's RETURN_NAMES.
   - `validate_inputs`: Maps to ComfyUI's VALIDATE_INPUTS.
   - `is_changed`: Maps to ComfyUI's IS_CHANGED.
   - `always_run`: Makes the node always run by generating a random IS_CHANGED.
   - `debug`: A boolean that makes this node print out extra information during its lifecycle.
   - `color`: Changes the node's color.
   - `bg_color`: Changes the node's color. If color is set and not bg_color, bg_color will just be a slightly darker color.

    Example:
    ```python
    from easy_nodes import ComfyNode, ImageTensor, NumberInput

    @ComfyNode(category="Image Processing",
               display_name="Enhance Image",
               is_output_node=True,
               debug=True,
               color="#FF00FF")
    def enhance_image(image: ImageTensor, factor: NumberInput(0.5, 0, 1, 0.1)) -> ImageTensor:
        output_image = enhance_my_image(image, factor)
        easy_nodes.show_image(output_image)  # Will show the image on the node, so you don't need a separate PreviewImage node.
        return output_image
    ```

2. **Annotate your function inputs and outputs**: Fully annotate function parameters and return types, using `list` to wrap types as appropriate. `tuple[output1, output2]` should be used if you have multiple outputs, otherwise you can just return the naked type (in the example below, that would be `list[int]`). This information is used to generate the fields of the internal class definition `@ComfyNode` sends to ComfyUI. If you don't annotate the inputs, the input will be treated as a wildcard. If you don't annotate the output, you won't see anything at all in ComfyUI.

    Example:
    ```python
    @ComfyNode("Utilities")
    def add_value(img_list: list[ImageTensor], val: int) -> list[int]:
        return [img + val for img in img_list]
    ```

### Registering new types:

Say you want a new type of special Tensor that ComfyUI will treat differently from Images. Say, a rotation matrix. Just create a placeholder class for it and use that in your annotations -- it's just for semantics; internally your functions will get whatever type of class they're handed.

```python
class RotationMatrix(torch.Tensor):
    def __init__(self):
        raise TypeError("!") # Will never be instantiated

easy_nodes.register_type(RotationMatrix, "ROTATION_MATRIX")

@ComfyNode()
def rotate_matrix_more(rot1: RotationMatrix, rot2: RotationMatrix) -> RotationMatrix:
    return rot1 * rot2
```

Making the class extend a torch.Tensor is not necessary, but it will give you nice type hints in IDEs.

### Creating dynamic nodes from classes

You can also automatically create nodes that will expose the fields of a class as widgets (as long as it has a default constructor). Say you have a complex options class from a third-party library you want to pass to a node.

```python
from some_library import ComplexOptions

easy_nodes.register_type(ComplexOptions)

easy_nodes.create_field_setter_node(ComplexOptions)
```

Now you should be should find a node named ComplexOptions that will have all the basic field types (str, int, float, bool) exposed as widgets.

## Automatic LLM Debugging

To enable the experimental LLM-based debugging, set your OPENAI_API_KEY prior to starting ComfyUI.

e.g.:
```bash
export OPENAI_API_KEY=sk-P#$@%J345jsd...
python main.py
```

Then open settings and turn the LLM debugging option to either "On" or "AutoFix".

Behavior:
  * "On": any exception in execution by an EasyNodes node (not regular nodes) will cause EasyNodes to collect all the relevent data and package it into a prompt for ChatGPT, which is instructed to reply with a fixed version of your function function from which a patch is created. That patch is displayed in the console and also saved to disk for evaluation.
  * "AutoFix": All of the above, and EasyNodes will also apply the patch and attempt to run the prompt again. This will repeat up to the configurable retry limit.

This feature is very experimental, and any contributions for things like improving the prompt flow and suporting other LLMs are welcome! You can find the implementation in [easy_nodes/llm_debugging.py](easy_nodes/llm_debugging.py).

## Contributing

Contributions are welcome! Please submit pull requests or open issues for any bugs, features, or improvements.
