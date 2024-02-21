import functools
import hashlib
import inspect
import logging
import os

import inspect
from functools import wraps
from typing import NewType, get_args, get_origin
import torch
import logging
import sys
import inspect

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


class NumberInput(int):
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
    has_default = default != inspect.Parameter.empty
    if type_name in ["INT", "FLOAT"]:
        default_value = 0
        if default != inspect.Parameter.empty:
            default_value = default
            if isinstance(default_value, NumberInput):
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
        return (type_name, {"default": default_value, "display": "number"}), has_default
    elif type_name in ["STRING"]:
        default_value = default if default != inspect.Parameter.empty else ""
        if isinstance(default_value, Choice):
            return (default_value.choices,), False
        if isinstance(default_value, StringInput):
            return (
                type_name,
                {"default": default_value, "multiline": default_value.multiline},
            ), False
        return (type_name, {"default": default_value}), has_default
    return (type_name,), has_default


def infer_input_types_from_annotations(func, skip_first, debug=False):
    """
    Infer input types based on function annotations.
    """
    input_is_list = {}
    sig = inspect.signature(func)
    input_types = {}
    optional_input_types = {}

    params = list(sig.parameters.items())
    
    if debug:
        print("ALL PARAMS", params)
    
    if skip_first:
        print("SKIPPING FIRST PARAM ", params[0])
        params = params[1:]

    for param_name, param in params:
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
    
    if debug:
        print(return_annotation)
    
    if origin is tuple:
        for arg in return_args:
            if debug:
                logging.error(get_origin(arg))

            if get_origin(arg) == list:
                output_is_list.append(True)
                list_arg = get_args(arg)[0]
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
    elif return_annotation is not inspect.Parameter.empty:
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
    node_class,
    process_function,
    display_name,
    workflow_name,
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
        
    if debug:
        print("process_function:", process_function.__name__)

    # Initial class dictionary setup
    class_dict = {
        "INPUT_TYPES": classmethod(lambda cls: all_inputs),
        "CATEGORY": category,
        "RETURN_TYPES": return_types,
        "FUNCTION": cname + "_wrapper",
        "INPUT_IS_LIST": input_is_list,
        "OUTPUT_IS_LIST": output_is_list,
        "OUTPUT_NODE": is_output_node,
        "VALIDATE_INPUTS": validate_inputs,
        "IS_CHANGED": is_changed,
        cname + "_wrapper": process_function,
    }
    class_dict = {k: v for k, v in class_dict.items() if v is not None}

    assert (
        workflow_name not in NODE_CLASS_MAPPINGS
    ), f"Node class '{workflow_name} ({cname})' already exists!"
    assert (
        display_name not in NODE_DISPLAY_NAME_MAPPINGS.values()
    ), f"Display name '{display_name}' already exists!"
    assert(
        node_class not in NODE_CLASS_MAPPINGS.values()
    ), f"Only one method from '{node_class} can be used as a ComfyUI node.'"

    if node_class:
        for key, value in class_dict.items():
            setattr(node_class, key, value)
    else:
        node_class = type(workflow_name, (object,), class_dict)
    
    if debug:
        print(type(node_class), node_class)

    NODE_CLASS_MAPPINGS[workflow_name] = node_class
    NODE_DISPLAY_NAME_MAPPINGS[workflow_name] = display_name
    
    

def is_static_method(klass, attr):
    """Check if a method is a static method."""
    if klass is None:
        return False
    attr_value = inspect.getattr_static(klass, attr, None)
    is_static = isinstance(attr_value, staticmethod)
    return is_static


def is_class_method(klass, attr):
    if klass is None:
        return False
    attr_value = inspect.getattr_static(klass, attr, None)
    is_class_method = isinstance(attr_value, classmethod)
    return is_class_method


def get_node_class(func):
    split_name = func.__qualname__.split(".")

    if len(split_name) > 1:
        class_name = split_name[-2]
        node_class = globals().get(class_name, None)
        if node_class is None and hasattr(func, '__globals__'):
            node_class = func.__globals__.get(class_name, None)
        return node_class
    return None


def ComfyFunc(
    category="default",
    display_name=None,
    workflow_name=None,
    is_output_node=False,
    validate_inputs=None,
    is_changed=None,
    debug=False):
    def decorator(func):
        if debug:
            print("================= Decorating", func.__qualname__)
        
        node_class = get_node_class(func)
        
        is_static = is_static_method(node_class, func.__name__)
        is_cls_mth = is_class_method(node_class, func.__name__)
        is_member = node_class is not None and not is_static and not is_cls_mth
                
        required_inputs, optional_inputs, input_is_list_map = infer_input_types_from_annotations(func, is_member, debug)
        
        if debug:
            print(func.__name__, "Is static:", is_static, "Is member:", is_member, "Class method:", is_cls_mth)
            print("Required inputs:", required_inputs)
            print(required_inputs, optional_inputs, input_is_list_map)
        
        return_types, output_is_list = infer_return_types_from_annotations(func, debug)
        
        # There's not much point in a node that doesn't have any outputs
        # and isn't an output itself, so auto-promote in that case.
        force_output = len(return_types) == 0
        name_parts = [x.title() for x in func.__name__.split("_")]
        input_is_list = any(input_is_list_map.values())
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if debug:
                print(func.__name__, "wrapper called with", len(args), "args and", len(kwargs), "kwargs. Is cls mth:", is_cls_mth)
                for i, arg in enumerate(args):
                    print("arg", i, type(arg))
                for key, arg in kwargs.items():
                    print("kwarg", key, type(arg))
                
            # For some reason self still gets passed with class methods.
            if is_cls_mth:
                args = args[1:]
            
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

            result = func(*args, **kwargs)
            if not isinstance(result, tuple):
                return (result,)
            return result
        
        if node_class is None or is_static:
            wrapper = staticmethod(wrapper)
            
        if is_cls_mth:
            wrapper = classmethod(wrapper)

        create_comfy_node(
            func.__qualname__,
            category,
            node_class,
            wrapper,
            display_name if display_name else " ".join(name_parts),
            workflow_name if workflow_name else "".join(name_parts),
            required_inputs,
            optional_inputs,
            input_is_list,
            return_types,
            output_is_list,
            is_output_node=is_output_node or force_output,
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
