import logging
from random import random
from easy_nodes import (
    NumberInput,
    ComfyNode,
    MaskTensor,
    StringInput,
    ImageTensor,
    Choice,
)
import easy_nodes
import torch


my_category = "EasyNodes Examples"

# By default EasyNodes will auto-register any decorated function (automatically insert it into ComfyUI's node registry).
# If you want to manually register your nodes the regular way, turn off
# auto_register and call easy_nodes.get_node_mappings()
easy_nodes.initialize_easy_nodes(default_category=my_category, auto_register=True)

# This is the converted example node from ComfyUI's example_node.py.example file.
@ComfyNode(my_category)
def annotated_example(
    image: ImageTensor,
    string_field: str = StringInput("Hello World!", multiline=False),
    int_field: int = NumberInput(0, 0, 4096, 64, "number"),
    float_field: float = NumberInput(1.0, 0, 10.0, 0.01, 0.001),
    print_to_screen: str = Choice(["enabled", "disabled"]),
) -> ImageTensor:
    if print_to_screen == "enable":
        print(
            f"""Your input contains:
            string_field aka input text: {string_field}
            int_field: {int_field}
            float_field: {float_field}
        """
        )
    # do some processing on the image, in this example I just invert it
    image = 1.0 - image
    return image  # Internally this gets auto-converted to (image,) for ComfyUI.


# You can wrap existing functions with ComfyFunc to expose them to ComfyUI as well.
def another_function(foo: float = 1.0):
    """Docstrings will be passed to the DESCRIPTION field on the node in ComfyUI."""
    print("Hello World!", foo)


ComfyNode(my_category, is_changed=lambda: random.random())(another_function)


# You can register arbitrary classes to be used as inputs or outputs.
class MyFunClass:
    def __init__(self):
        self.width = 640
        self.height = 640
        self.color = 0.5        
easy_nodes.register_type(MyFunClass, "FUN_CLASS")


# If you don't want to create a node manually to create the class, you can use the
# create_field_setter_node to automatically create a node that sets the fields on the class.
easy_nodes.create_field_setter_node(MyFunClass)


@ComfyNode(my_category, is_output_node=True, color="#4F006F")
def my_fun_class_node_processor(fun_class: MyFunClass) -> ImageTensor:
    print(f"Processing MyFunClass: {fun_class.width} {fun_class.height} {fun_class.color}")
    my_image = torch.rand((1, fun_class.height, fun_class.width, 3)) * fun_class.color
    return my_image


@ComfyNode(my_category)
def create_random_image(width: int=NumberInput(128, 128, 1024), 
                        height: int=NumberInput(128, 128, 1024)) -> ImageTensor:
    return torch.rand((1, height, width, 3))


# You can also wrap a method on a class and thus maintain state between calls.
#
# Note that you can only expose one method per class, and you have to define the
# full class before manually calling the decorator on the method.
class ExampleClass:
    def __init__(self):
        self.counter = 42

    def my_method(self) -> int:
        print(f"ExampleClass Hello World! {self.counter}")
        self.counter += 1
        return self.counter


def my_is_changed_func():
    return random()

ComfyNode(
    my_category,
    is_changed=my_is_changed_func,
    description="Descriptions can also be passed in manually. This operation increments a counter",
)(ExampleClass.my_method)


# Preview text and images right in the nodes.
@ComfyNode(my_category, is_output_node=True)
def preview_example(str2: str = StringInput("")) -> str:
    easy_nodes.show_text(f"hello: {str2}")
    return str2


# Wrapping a class method
class AnotherExampleClass:
    class_counter = 42

    @classmethod
    def my_class_method(cls, foo: float):
        print(f"AnotherExampleClass Hello World! {cls.class_counter} {foo}")
        cls.class_counter += 1


ComfyNode(my_category, is_changed=lambda: random.random())(
    AnotherExampleClass.my_class_method
)


# ImageTensors and MaskTensors are both just torch.Tensors. Use them in annotations to
# differentiate between images and masks in ComfyUI. This is purely cosmetic, and they
# are interchangeable in Python. If you annotate the type of a parameter as torch.Tensor
# it will be treated as an ImageTensor.
@ComfyNode(my_category, color="#00FF00")
def convert_to_image(mask: MaskTensor) -> ImageTensor:
    return mask


@ComfyNode(my_category)
def text_repeater(text: str=StringInput("Sample text"), 
                  times: int=NumberInput(10, 1, 100)) -> list[str]:
    return [text] * times


# If you wrap your input types in list[], under the hood the decorator will make sure you get
# everything in a single call with the list inputs passed to you as lists automatically.
# If you don't, then you'll get multiple calls with a single item on each call.
@ComfyNode(my_category)
def combine_lists(
    image1: list[ImageTensor], image2: list[ImageTensor]
) -> list[ImageTensor]:
    combined_lists = image1 + image2
    return combined_lists


# Adding a default for a param makes it optional, so ComfyUI won't require it to run your node.
@ComfyNode(my_category)
def add_images(
    image1: ImageTensor, image2: ImageTensor, image3: ImageTensor = None
) -> ImageTensor:
    combined_tensors = image1 + image2
    if image3 is not None:
        combined_tensors += image3
    return combined_tensors


@ComfyNode(my_category, is_output_node=True, color="#006600")
def example_show_mask(mask: MaskTensor) -> MaskTensor:
    easy_nodes.show_image(mask)
    logging.info("Showing mask")
    return mask


# Multiple outputs can be returned by annotating with tuple[].
# Pass return_names if you want to give them labels in ComfyUI.
@ComfyNode("Example category", color="#0066cc", bg_color="#ffcc00", return_names=["Below", "Above"])
def threshold_image(image: ImageTensor, threshold_value: float = NumberInput(0.5, 0, 1, 0.0001, display="slider")) -> tuple[MaskTensor, MaskTensor]:
    """Returns separate masks for values above and below the threshold value."""
    mask_below = torch.any(image < threshold_value, dim=-1)
    return mask_below.float(), (~mask_below).float()


# ImageTensor and MaskTensor are just torch.Tensors, so you can treat them as such.
@ComfyNode(my_category, color="#0000FF")
def example_mask_image(image: ImageTensor, 
                       mask: MaskTensor,
                       value: float=NumberInput(0, 0, 1, 0.0001, display="slider")) -> ImageTensor:
    image = image.clone()
    image[mask == 0] = value
    easy_nodes.show_image(image)
    return image


# As long as Python is happy, ComfyUI will be happy with whatever you tell it the return type is.
# You can set the node color by passing in a color argument to the decorator.
@ComfyNode(my_category, color="#FF0000")
def convert_to_mask(image: ImageTensor, threshold: float = NumberInput(0.5, 0, 1, 0.0001, display="slider")) -> MaskTensor:
    return (image > threshold).float()


# The decorated functions remain normal Python functions, so we can nest them inside each other too.
@ComfyNode(my_category)
def mask_image_with_image(
    image: ImageTensor, image_to_use_as_mask: ImageTensor
) -> ImageTensor:
    mask = convert_to_mask(image_to_use_as_mask)
    return example_mask_image(image, mask)


# And of course you can use the code in normal Python scripts too.
if __name__ == "__main__":
    tensor = torch.rand((5, 5))
    tensor_inverted = annotated_example(tensor, "hello", 5, 0.5, "enable")
    assert torch.allclose(tensor, 1.0 - tensor_inverted)

    tensor_inverted_again = annotated_example(tensor_inverted, "Hi!")
    assert torch.allclose(tensor, tensor_inverted_again)
