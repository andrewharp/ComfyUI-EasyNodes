from random import random
from comfy_annotations import (
    NumberInput,
    ComfyFunc,
    MaskTensor,
    StringInput,
    ImageTensor,
    Choice,
)
import torch

my_category = "Comfy Annotation Examples"


# This is the converted example node from ComfyUI's example_node.py.example file.
@ComfyFunc(my_category)
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
    print("Hello World!")


ComfyFunc(my_category, is_changed=lambda: random.random())(another_function)


# You can also wrap a method on a class and thus maintain state between calls.
#
# Note that you can only expose one method per class, and you have to define the
# full class before manually calling the decorator on the method.
class ExampleClass:
    def __init__(self):
        self.counter = 42

    def my_method(self):
        print(f"ExampleClass Hello World! {self.counter}")
        self.counter += 1


ComfyFunc(my_category, is_changed=lambda: random.random())(ExampleClass.my_method)


# Wrapping a class method
class AnotherExampleClass:
    class_counter = 42

    @classmethod
    def my_class_method(cls, foo: float):
        print(f"AnotherExampleClass Hello World! {cls.class_counter} {foo}")
        cls.class_counter += 1


ComfyFunc(my_category, is_changed=lambda: random.random())(
    AnotherExampleClass.my_class_method
)


# ImageTensors and MaskTensors are both just torch.Tensors. Use them in annotations to 
# differentiate between images and masks in ComfyUI. This is purely cosmetic, and they 
# are interchangeable in Python. If you annotate the type of a parameter as torch.Tensor
# it will be treat as an ImageTensor.
@ComfyFunc(my_category)
def convert_to_image(mask: MaskTensor) -> ImageTensor:
    return mask


# If you wrap your input types in list[], under the hood the decorator will make sure you get
# everything in a single call with the list inputs passed to you as lists automatically.
# If you don't, then you'll get multiple calls with a single item on each call.
@ComfyFunc(my_category)
def combine_lists(
    image1: list[torch.Tensor], image2: list[torch.Tensor]
) -> list[torch.Tensor]:
    combined_lists = image1 + image2
    return combined_lists


# Adding a default for a param makes it optional, so ComfyUI won't require it to run your node.
@ComfyFunc(my_category)
def combine_tensors(
    image1: torch.Tensor, image2: torch.Tensor, image3: torch.Tensor = None
) -> torch.Tensor:
    combined_tensors = image1 + image2
    if image3 is not None:
        combined_tensors += image3
    return combined_tensors


# Multiple outputs can be returned by annotating with tuple[].
# Pass return_names if you want to give them labels in ComfyUI.
@ComfyFunc(my_category, return_names=["below_threshold", "above_threshold"])
def threshold_image(
    image: torch.Tensor, threshold_value: float
) -> tuple[MaskTensor, MaskTensor]:
    return image < threshold_value, image > threshold_value


# ImageTensor and MaskTensor are just torch.Tensors, so you can treat them as such.
@ComfyFunc(my_category)
def example_mask_image(image: ImageTensor, mask: MaskTensor) -> ImageTensor:
    image = image.clone()
    image[mask == 0] = 1.0
    return image


# As long as Python is happy, ComfyUI will be happy with whatever you tell it the return type is.
@ComfyFunc(my_category)
def convert_to_mask(image: ImageTensor, threshold: float = 0.5) -> MaskTensor:
    return (image > threshold).float()


# The decorated functions remain normal Python functions, so we can nest them inside each other too.
@ComfyFunc(my_category)
def mask_image_with_image(
    image: ImageTensor, image_to_use_as_mask: ImageTensor
) -> ImageTensor:
    mask = convert_to_mask(image_to_use_as_mask)
    return mask_image(image, mask)


# And of course you can use the code in normal Python scripts too.
if __name__ == "__main__":
    tensor = torch.rand((5, 5))
    tensor_inverted = annotated_example(tensor, "hello", 5, 0.5, "enable")
    assert torch.allclose(tensor, 1.0 - tensor_inverted)

    tensor_inverted_again = annotated_example(tensor_inverted, "Hi!")
    assert torch.allclose(tensor, tensor_inverted_again)
