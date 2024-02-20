import functools
import hashlib
import inspect
import logging
import os
import pathlib
import sys

import inspect
from functools import wraps
from typing import NewType, get_args, get_origin
import torch
import logging
import sys

default_category = "ComfyFunc"


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


class Choice(str):
    def __new__(cls, choices: list[str]):
        instance = super().__new__(cls, choices[0])
        instance.choices = choices
        return instance

    def __str__(self):
        return self.choices[0]


class StringInput(str):
    def __new__(cls, value, multiline=False):
        instance = super().__new__(cls, value)
        instance.multiline = multiline
        return instance


class BoundedNumber(int):
    def __new__(
        cls,
        value,
        min_value=None,
        max_value=None,
        step=None,
        round=None,
        display: str = "number",
    ):
        if min_value is not None and value < min_value:
            raise ValueError(
                f"Value {value} is less than the minimum allowed {min_value}."
            )
        if max_value is not None and value > max_value:
            raise ValueError(
                f"Value {value} is greater than the maximum allowed {max_value}."
            )
        instance = super().__new__(cls, value)
        instance.min_value = min_value
        instance.max_value = max_value
        instance.display = display
        instance.step = step
        instance.round = round
        return instance

    def __repr__(self):
        return f"{super().__repr__()} (Min: {self.min_value}, Max: {self.max_value})"


PathString = NewType("PathString", str)


# Our any instance wants to be a wildcard string
any_type = AnyType("*")

DEFAULT_CAT = "unspecified"


class ImageTensor(torch.Tensor):
    def __new__(cls):
        raise TypeError("Do not instantiate this class directly.")


class MaskTensor(torch.Tensor):
    def __new__(cls):
        raise TypeError("Do not instantiate this class directly.")
    

ANNOTATION_TO_COMFYUI_TYPE = {
    torch.Tensor: "IMAGE",
    ImageTensor: "IMAGE",
    MaskTensor: "MASK",
    int: "INT",
    float: "FLOAT",
    str: "STRING",
    bool: "BOOLEAN",
    AnyType: any_type,
}


def register_type(cls, name: str):
    ANNOTATION_TO_COMFYUI_TYPE[cls] = name


def get_type_str(the_type) -> str:
    return ANNOTATION_TO_COMFYUI_TYPE[the_type]


def annotate_input(
    type_name, default=inspect.Parameter.empty, debug=False
) -> tuple[tuple, bool]:
    if type_name in ["INT", "FLOAT"]:
        default_value = 0
        if default != inspect.Parameter.empty:
            default_value = default
            if isinstance(default_value, BoundedNumber):
                metadata = {
                    "default": default_value,
                    "display": default_value.display,
                    "min_value": default_value.min_value,
                    "max_value": default_value.max_value,
                    "step": default_value.step,
                    "round": default_value.round,
                }
                metadata = {k: v for k, v in metadata.items() if v is not None}
                return (type_name, metadata), False
        if debug:
            print(f"Default value for {type_name} is {default_value}")
        return (type_name, {"default": default_value, "display": "number"}), False
    elif type_name in ["STRING"]:
        default_value = default if default != inspect.Parameter.empty else ""
        if isinstance(default_value, Choice):
            return (default_value.choices,), False
        if isinstance(default_value, StringInput):
            return (
                type_name,
                {"default": default_value, "multiline": default_value.multiline},
            ), False
        return (type_name, {"default": default_value}), False
    return (type_name,), default != inspect.Parameter.empty and type_name not in [
        "BOOLEAN"
    ]


def infer_input_types_from_annotations(func, debug=False):
    """
    Infer input types based on function annotations.
    """
    input_is_list = {}
    sig = inspect.signature(func)
    input_types = {}
    optional_input_types = {}
    for param_name, param in sig.parameters.items():
        input_is_list[param_name] = get_origin(param.annotation) is list
        if debug:
            print("Param default:", param.default)
        comfyui_type = ANNOTATION_TO_COMFYUI_TYPE.get(param.annotation, any_type)
        the_param, is_optional = annotate_input(comfyui_type, param.default, debug)
        if not is_optional:
            input_types[param_name] = the_param
        else:
            optional_input_types[param_name] = the_param
    return input_types, optional_input_types, input_is_list


def infer_return_types_from_annotations(func, debug=False):
    """
    Infer whether each element in a function's return tuple is a list or a single item.
    """
    return_annotation = inspect.signature(func).return_annotation
    return_args = get_args(return_annotation)
    origin = get_origin(return_annotation)

    if debug:
        print(f"return_annotation: '{return_annotation}'")
        print(f"return_args: '{return_args}'")
        print(f"origin: '{origin}'")
        print(type(return_annotation))

    types_mapped = []
    output_is_list = []
    if origin is tuple:
        for arg in return_args:
            if debug:
                logging.error(get_origin(arg))

            if get_origin(arg) == list:
                output_is_list.append(True)
                list_arg = get_args(arg)[
                    0
                ]  # Assuming only one type argument for the list
                types_mapped.append(ANNOTATION_TO_COMFYUI_TYPE.get(list_arg, None))
            else:
                output_is_list.append(False)
                types_mapped.append(ANNOTATION_TO_COMFYUI_TYPE.get(arg, None))
    elif origin is list:
        if debug:
            print(ANNOTATION_TO_COMFYUI_TYPE.get(return_annotation, None))
            print(return_annotation)
            print(return_args)
        types_mapped.append(ANNOTATION_TO_COMFYUI_TYPE.get(return_args[0], None))
        output_is_list.append(origin is list)
    else:
        types_mapped.append(ANNOTATION_TO_COMFYUI_TYPE.get(return_annotation, None))
        output_is_list.append(False)

    return_types_tuple = tuple(types_mapped)
    output_is_lists_tuple = tuple(output_is_list)
    if debug:
        print(
            f"return_types_tuple: '{return_types_tuple}', output_is_lists_tuple: '{output_is_lists_tuple}'"
        )

    return return_types_tuple, output_is_lists_tuple


def clone_func(func, wrapped_func):
    sig = inspect.signature(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Unpack the dictionary when passing it as **kwargs
        return wrapped_func(**bound_args.arguments)

    wrapper.__signature__ = sig
    return wrapper


# def _custom_validate(**kwargs):
#     for key, value in kwargs.items():
#         logging.info("%s %s", key, type(value))
#         if isinstance(value, str):
#             logging.info(f"'{value}'")
#         if isinstance(value, pathlib.Path):
#             if not value.exists():
#                 raise FileNotFoundError(f"Doesn't exist!")
#     return True


# def _custom_is_changed(**kwargs):
#     logging.error("_custom_is_changed: " + str(kwargs.keys()))
#     hash_input = ''
#     for key, value in kwargs.items():
#         logging.info("%s %s", key, type(value))
#         # if isinstance(value, Path):
#         if isinstance(value, str):
#             logging.info(f"'{value}'")
#             hash_input += str(value)
#     hash_object = hashlib.sha256(hash_input.encode())
#     digest = hash_object.hexdigest()
#     if len(hash_input) > 0:
#         raise Exception("This actually ran?")
#     logging.info("IS_CHANGED %d %d '%s' '%s'", len(kwargs), len(kwargs.keys()), hash_input, digest)
#     return digest


def create_comfy_node(
    cname,
    category,
    process_function,
    display_name,
    required_inputs,
    optional_inputs,
    input_is_list,
    return_types,
    output_is_list,
    is_output_node=False,
    validate_inputs=None,
    is_changed=None,
    debug=False,
):
    logger = logging.getLogger(cname)
    if debug:
        logger.info(
            "-------------------------------------------------------------------"
        )
        logger.info(f"Creating Comfy node for {process_function.__name__}")

    if debug:
        logger.info(return_types)
        logger.info(output_is_list)

    all_inputs = {"required": required_inputs, "optional": optional_inputs}

    if debug:
        print(f"Final returns: {return_types}")

    # Initial class dictionary setup
    class_dict = {
        "INPUT_TYPES": classmethod(lambda cls: all_inputs),
        "CATEGORY": category,
        "RETURN_TYPES": return_types,
        "FUNCTION": cname,
        "INPUT_IS_LIST": input_is_list,
        "OUTPUT_IS_LIST": output_is_list,
        "OUTPUT_NODE": is_output_node,
        process_function.__name__: staticmethod(process_function),
    }

    if validate_inputs:
        class_dict["VALIDATE_INPUTS"] = staticmethod(validate_inputs)

    if is_changed:
        class_dict["IS_CHANGED"] = staticmethod(is_changed)

    for key, value in class_dict.items():
        assert (
            value is not None
        ), f"Value for {key} cannot be None in class_dict for {cname}"

    camel_case = "".join(x.title() for x in cname.split("_"))
    assert (
        camel_case not in NODE_CLASS_MAPPINGS
    ), f"Node class '{camel_case} ({cname})' already exists!"

    if not display_name:
        display_name = " ".join(x.title() for x in cname.split("_"))
    assert (
        display_name not in NODE_DISPLAY_NAME_MAPPINGS.values()
    ), f"Display name '{display_name}' already exists!"

    node_class = type(camel_case, (object,), class_dict)

    NODE_CLASS_MAPPINGS[camel_case] = node_class
    NODE_DISPLAY_NAME_MAPPINGS[camel_case] = display_name


def ComfyFunc(
    category="default",
    display_name=None,
    is_output_node=False,
    validate_inputs=None,
    is_changed=None,
    debug=False):
    def decorator(func):
        unique_class_name = func.__name__
        # validator = clone_func(func, _custom_validate)

        required_inputs, optional_inputs, input_is_list_map = (
            infer_input_types_from_annotations(func, debug)
        )
        return_types, output_is_list = infer_return_types_from_annotations(
            func, debug
        )
        
        input_is_list = any(input_is_list_map.values())

        @functools.wraps(func)
        def wrapper(**kwargs):
            for i, arg in enumerate(kwargs):
                print("kwarg", i, type(arg), len(arg))
            
            # If the python function didn't annotate it as a list,
            # but INPUT_TYPES does, then we need to convert make it not a list.
            if input_is_list:
                for arg_name in kwargs.keys():
                    print(arg_name, len(kwargs[arg_name]))
                    if not input_is_list_map[arg_name]:
                        assert len(kwargs[arg_name]) == 1
                        kwargs[arg_name] = kwargs[arg_name][0]
            
            # if not validator(**kwargs):
            #     print("Inputs not validated")
            #     raise Exception("Inputs not validated")

            result = func(**kwargs)
            if not isinstance(result, tuple):
                return (result,)
            return result

        create_comfy_node(
            unique_class_name,
            category,
            wrapper,
            display_name,
            required_inputs,
            optional_inputs,
            input_is_list,
            return_types,
            output_is_list,
            is_output_node=is_output_node,
            validate_inputs=validate_inputs,
            is_changed=is_changed,
            debug=debug
        )

        return func

    return decorator


def return_with_status_text(result, text, unique_id=None, extra_pnginfo=None):
    if unique_id and extra_pnginfo and "workflow" in extra_pnginfo:
        workflow = extra_pnginfo["workflow"]
        node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id), None)
        if node:
            node["widgets_values"] = [text]
    return {"ui": {"text": [text]}, "result": (result,)}


def sha1_file(filepath):
    """Calculate the SHA-1 checksum of a file."""
    hash_sha1 = hashlib.sha1()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha1.update(chunk)
    return hash_sha1.hexdigest()


def get_filenames(dir_path):
    filenames = os.listdir(dir_path)
    filenames.sort()
    return filenames


def sha1_directory(dir_path):
    """Calculate the SHA-1 checksum of a directory."""
    hash_sha1 = hashlib.sha1()
    for filename in get_filenames(dir_path):
        filepath = os.path.join(dir_path, filename)
        if os.path.isfile(filepath):
            hash_sha1.update(sha1_file(filepath).encode("utf-8"))
    return hash_sha1.hexdigest()
