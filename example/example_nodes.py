from comfy_annotations import BoundedNumber, ComfyFunc, MaskTensor, StringInput, ImageTensor, Choice
import torch

my_category = "Comfy Annotation Examples"

# This is the converted example node from ComfyUI's example_node.py.example file.
@ComfyFunc(category=my_category)
def annotated_example(image: ImageTensor, 
                string_field: str = StringInput("Hello World!", multiline=False),
                int_field: int = BoundedNumber(0, 0, 4096, 64, "number"), 
                float_field: float = BoundedNumber(1.0, 0, 10.0, 0.01, 0.001),
             print_to_screen: str = Choice(["enabled", "disabled"])) -> ImageTensor:
    if print_to_screen == "enable":
        print(f"""Your input contains:
            string_field aka input text: {string_field}
            int_field: {int_field}
            float_field: {float_field}
        """)
    #do some processing on the image, in this example I just invert it
    image = 1.0 - image
    return image  # Internally this gets auto-converted to (image,) for ComfyUI.


# ImageTensors and MaskTensors are both just torch.Tensors, but they are help to differentiate between
# images and masks in ComfyUI. This is purely cosmetic, and they are interchangeable in Python.
# If you annotate the type of a parameter as torch.Tensor, ComfyUI will treat it as an ImageTensor.
@ComfyFunc(category=my_category)
def convert_to_image(mask: MaskTensor) -> ImageTensor:
    return mask


@ComfyFunc(category=my_category)
def combine_lists(image1: list[torch.Tensor], image2: list[torch.tensor]) -> list[torch.Tensor]:
    combined_lists = image1 + image2
    return combined_lists


@ComfyFunc(category=my_category)
def combine_tensors(image1: torch.Tensor, image2: torch.Tensor, image3: torch.Tensor=None) -> torch.Tensor:
    combined_tensors = image1 + image2
    
    if image3 is not None:
        combined_tensors += image3
    
    return combined_tensors


@ComfyFunc(category=my_category)
def threshold_image(image: torch.Tensor, threshold_value: float) -> tuple[MaskTensor, MaskTensor]:
    return image < threshold_value, image > threshold_value


@ComfyFunc(category=my_category)
def mask_image(image: ImageTensor, mask: MaskTensor) -> ImageTensor:
    return image * mask


@ComfyFunc(category=my_category)
def convert_to_mask(image: ImageTensor, threshold: float=0.5) -> MaskTensor:
    return (image > threshold).float()


# We can also use it as a regular method. The special Number/Choice/ImageTensor types are only used by ComfyUI,
# and turn into plain Python types/torch.Tensors when used here.
if __name__ == '__main__':
    tensor = torch.rand((5, 5))
    tensor_inverted = annotated_example(tensor, "hello", 5, 0.5, "enable")
    assert torch.allclose(tensor, 1.0 - tensor_inverted)
    
    tensor_inverted_again = annotated_example(tensor_inverted, "Hi!")
    assert torch.allclose(tensor, tensor_inverted_again)