
import enum
import functools
import hashlib
import importlib
import inspect
import io
import json
import logging
import math
import os
import sys
import traceback
import typing
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import Callable, Dict, Union, get_args, get_origin

import nodes as comfyui_nodes
import numpy as np
import torch
from colorama import Fore
from PIL import Image

import easy_nodes
import easy_nodes.config_service as config_service
import easy_nodes.llm_debugging as llm_debugging

# Export the web directory so ComfyUI can pick up the JavaScript.
_web_path = os.path.join(os.path.dirname(__file__), "web")

if os.path.exists(_web_path):
    comfyui_nodes.EXTENSION_WEB_DIRS["ComfyUI-EasyNodes"] = _web_path
    logging.debug(f"Registered ComfyUI-EasyNodes web directory: '{_web_path}'")
else:
    logging.warning(f"ComfyUI-EasyNodes: Web directory not found at {_web_path}. Some features may not be available.")


class AutoDescriptionMode(Enum):
    NONE = "none"
    BRIEF = "brief"
    FULL = "full"
    

class CheckSeverityMode(Enum):
    OFF = "off"
    WARN = "warn"
    FATAL = "fatal"


@dataclass
class EasyNodesConfig:
    default_category: str
    auto_register: bool
    docstring_mode: AutoDescriptionMode
    verify_level: CheckSeverityMode
    auto_move_tensors: bool
    NODE_CLASS_MAPPINGS: dict
    NODE_DISPLAY_NAME_MAPPINGS: dict
    num_registered: int = 0
    get_node_mappings_called: bool = False


# Keep track of the config from the last init, because different custom_nodes modules 
# could possibly want different settings.
_current_config: EasyNodesConfig = None


# Changing the default of auto-register to false in 1.1, so catch if the user hasn't set it explicitly so we can give them a warning.
class AutoRegisterSentinel(enum.Enum):
    DEFAULT = enum.auto()


def initialize_easy_nodes(default_category: str = "EasyNodes", 
         auto_register: bool = AutoRegisterSentinel.DEFAULT, 
         docstring_mode: AutoDescriptionMode = AutoDescriptionMode.FULL, 
         verify_level: CheckSeverityMode = CheckSeverityMode.WARN,
         auto_move_tensors: bool = False):
    """
    Initializes the EasyNodes library with the specified configuration options.
    
    All nodes created after this call until the next call of init() will use the specified configuration options.

    Args:
        default_category (str, optional): The default category for nodes. Defaults to "EasyNodes".
        auto_register (bool, optional): Whether to automatically register nodes with ComfyUI (so you don't have to export). Defaults to False. Experimental.
        docstring_mode (AutoDescriptionMode, optional): The mode for generating node docstrings. Defaults to AutoDescriptionMode.FULL.
        verify_level (bool, optional): Whether to verify tensors for shape and data type according to ComfyUI type (MASK, IMAGE, etc). Runs on inputs and outputs. Defaults to False.
        auto_move_tensors (bool, optional): Whether to automatically move torch Tensors to the GPU before your function gets called, and then to the CPU on output. Defaults to False.
    """
    # If the user has already requested a prompt, that means auto-reload could conceivably re-import this module.
    # In that case, we should just return and not re-initialize.
    if _after_first_prompt:
        return
    
    global _current_config
    if _current_config:
        assert _current_config.num_registered > 0, "Re-initializing EasyNodes, but no Nodes have been registered since last initialization. This may indicate an issue."        
        assert _current_config.auto_register or not _current_config.NODE_CLASS_MAPPINGS, (
            f"Auto-registration was turned off by previous initializer, but {len(_current_config.NODE_CLASS_MAPPINGS)} nodes were not picked up.")

    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    
    auto_register_message = ""
    if auto_register is AutoRegisterSentinel.DEFAULT:
        auto_register_message = " NOTE: Auto-registration not set explicitly, running in mixed-mode. The default will change to False in a future version. If already calling get_node_mappings(), you can ignore this message (or pass auto-register explicitly to make it go away)."

    logging.info(f"Initializing EasyNodes. Auto-registration: {auto_register}{auto_register_message}")

    if auto_register is True:
        NODE_CLASS_MAPPINGS = comfyui_nodes.NODE_CLASS_MAPPINGS
        NODE_DISPLAY_NAME_MAPPINGS = comfyui_nodes.NODE_DISPLAY_NAME_MAPPINGS
    
    if auto_register is True or auto_register is AutoRegisterSentinel.DEFAULT:
        frame = sys._getframe(1).f_globals['__name__']
        _ensure_package_dicts_exist(frame)

    _current_config = EasyNodesConfig(default_category, auto_register, docstring_mode, verify_level, auto_move_tensors, 
                                      NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)


def get_node_mappings():
    assert _current_config is not None, "EasyNodes not initialized. Call easy_nodes.initialize_easy_nodes() before using ComfyNode."
    assert _current_config.num_registered > 0, "No nodes registered. Use the @ComfyNode() decorator to register nodes after calling easy_nodes.initialize_easy_nodes()."
    assert _current_config.auto_register is not True, "Auto-node registration is on. Call easy_nodes.initialize_easy_nodes(auto_register=False) if you want to export manually."
    assert not _current_config.get_node_mappings_called, "get_node_mappings() already called. This function should only be called once."
    _current_config.get_node_mappings_called = True
    return _current_config.NODE_CLASS_MAPPINGS, _current_config.NODE_DISPLAY_NAME_MAPPINGS


def _get_curr_config() -> EasyNodesConfig:
    if _current_config is None:
        logging.warning("easy_nodes.initialize_easy_nodes() should be called prior to any other EasyNodes activity. Initializing now with easy_nodes.initialize_easy_nodes() for backwards compatibility.")
        easy_nodes.initialize_easy_nodes()
    return _current_config


# Use as a default str value to show choices to the user.
class Choice(str):
    def __new__(cls, choices: list[str]):
        instance = super().__new__(cls, choices[0])
        instance.choices = choices
        return instance

    def __str__(self):
        return self.choices[0]


class StringInput(str):
    def __new__(cls, value, multiline=False, force_input=False, optional=False, hidden=False):
        instance = super().__new__(cls, value)
        instance.value = value
        instance.multiline = multiline
        instance.force_input = force_input
        instance.optional = optional
        instance.hidden = hidden
        return instance

    def to_dict(self):
        return {
            "default": self.value,
            "multiline": self.multiline,
            "display": "input",
            "forceInput": self.force_input,
        }


class NumberInput(float):
    def __new__(
        cls,
        default,
        min=None,
        max=None,
        step=None,
        round=None,
        display: str = "number",
        optional=False,
        hidden=False,
        force_input=False
    ):
        if min is not None and default < min:
            raise ValueError(f"Value {default} is less than the minimum allowed {min}.")
        if max is not None and default > max:
            raise ValueError(
                f"Value {default} is greater than the maximum allowed {max}."
            )
        instance = super().__new__(cls, default)
        instance.min = min
        instance.max = max
        instance.display = display
        instance.step = step
        instance.round = round
        instance.optional = optional
        instance.hidden = hidden
        instance.force_input = force_input
        return instance

    def to_dict(self):
        metadata = {
            "default": self,
            "display": self.display,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "round": self.round,
            "forceInput": self.force_input,
        }
        metadata = {k: v for k, v in metadata.items() if v is not None}
        return metadata

    def __repr__(self):
        return f"{super().__repr__()} (Min: {self.min}, Max: {self.max})"


_ANNOTATION_TO_COMFYUI_TYPE = {}
_SHOULD_AUTOCONVERT = {"str": True}
_DEFAULT_FORCE_INPUT = {}
_COMFYUI_TYPE_TO_ANNOTATION_CLS = {}

_after_first_prompt = False
_module_reload_times = {}
_module_dict = {}

_function_dict = {}
_function_checksums = {}
_function_update_times = {}

_curr_preview = {}
_curr_unique_id = None


class CustomVerifier:
    def __init__(self):
        raise NotImplementedError()
    
    def __call__(self, arg):
        raise NotImplementedError()


# This is the default validator that just ensures that one of them is directly descended
# from the other. This allows the semantic classes and the actual classes to be used interchangeably.
class SubclassVerifier(CustomVerifier):
    def __init__(self, cls: type):
        self.cls = cls
    
    def __call__(self, arg):
        assert issubclass(self.cls, type(arg)) or issubclass(type(arg), self.cls), (
            f"Expected a {self.cls.__name__}, but got received a value ({arg}) of type {type(arg).__name__}.")


class TypeVerifier(CustomVerifier):
    def __init__(self, allowed_types: list):
        self.allowed_types = allowed_types

    def __call__(self, value):
        for allowed_type in self.allowed_types:
            if isinstance(value, allowed_type):
                return
        raise ValueError(f"Expected one of {self.allowed_types}, got {type(value)}")


class AnythingVerifier(CustomVerifier):
    def __init__(self):
        pass
    def __call__(self, value):
        pass


@dataclass
class TensorVerifier(CustomVerifier):
    tensor_type_name: str
    allowed_shapes: list = None
    allowed_dims: list = None
    allowed_channels: list = None
    allowed_range: list = None
    
    def __call__(self, tensor):
        assert isinstance(tensor, torch.Tensor), f"Expected an {self.tensor_type_name}, got {type(tensor).__name__}"
        
        if self.allowed_range is not None:
            assert tensor.min() >= self.allowed_range[0] and tensor.max() <= self.allowed_range[1], f"{self.tensor_type_name} tensor must have values between {self.allowed_range[0]} and {self.allowed_range[1]}, got min {tensor.min()} and max {tensor.max()}"
        
        if self.allowed_shapes is not None:
            assert len(tensor.shape) in self.allowed_shapes, f"{self.tensor_type_name} tensor must have shape in {self.allowed_shapes}, got {tensor.shape}"
            
        if self.allowed_dims is not None:
            for i, dim in enumerate(self.allowed_dims):
                assert tensor.shape[i] <= dim, f"{self.tensor_type_name} tensor dimension {i} must be less than or equal to {dim}, got {tensor.shape[i]}"
                
        if self.allowed_channels is not None:
            assert tensor.shape[-1] in self.allowed_channels, f"{self.tensor_type_name} tensor must have the number of channels in {self.allowed_channels}, got {tensor.shape[-1]}"



_custom_verifiers: Dict[str, CustomVerifier] = {}


def _get_fully_qualified_name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def register_type(
    cls: type, 
    name: str = None, 
    should_autoconvert: bool = False, 
    is_auto_register: bool = False, 
    force_input: bool = False,
    verifier: CustomVerifier = None
):
    """Register a type for ComfyUI.

    Args:
        cls (type): The type to register.
        name (str): The name of the type.
        should_autoconvert (bool, optional): Whether the type should be automatically converted to the expected type before being passed to the wrapped function. Defaults to False.
        is_auto_register (bool, optional): Whether the type is automatically registered. Defaults to False.
        force_input (bool, optional): Whether the type should be forced as an input. Defaults to False.
    """
    if _after_first_prompt:
        return
    
    if name is None:
        name = cls.__name__
    
    key = _get_fully_qualified_name(cls)
    if not is_auto_register:
        assert key not in _ANNOTATION_TO_COMFYUI_TYPE, f"Type {cls} already registered."
        # assert name not in _COMFYUI_TYPE_TO_ANNOTATION, f"Type {name} already registered."

    if key in _ANNOTATION_TO_COMFYUI_TYPE:
        return
    
    # Assume the first type registered is the most general, and ignore later ones.
    # This means you can register multiple types of string-like object as STRING for
    # semantic purposes, but return values will just be checked if they're isinstance(v, str)
    if name not in _COMFYUI_TYPE_TO_ANNOTATION_CLS:
        _COMFYUI_TYPE_TO_ANNOTATION_CLS[name] = cls
        if verifier:
            _custom_verifiers[name] = verifier
        else:
            _custom_verifiers[name] = SubclassVerifier(cls)
    elif verifier:
        logging.warning(f"Custom verifier for {name} already registered. Ignoring new one.")

    _ANNOTATION_TO_COMFYUI_TYPE[key] = name
    _SHOULD_AUTOCONVERT[key] = should_autoconvert
    _DEFAULT_FORCE_INPUT[key] = force_input


# Made to match any and all other types.
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")


def _get_type_str(the_type: type) -> str:
    key = _get_fully_qualified_name(the_type)
    if key not in _ANNOTATION_TO_COMFYUI_TYPE and get_origin(the_type) is list:
        return _get_type_str(get_args(the_type)[0])

    if key not in _ANNOTATION_TO_COMFYUI_TYPE and the_type is not inspect._empty:
        logging.warning(
            f"Type '{the_type}' not registered with ComfyUI, treating as wildcard"
        )
        raise ValueError(f"Type '{the_type}' not registered with ComfyUI")

    type_str = _ANNOTATION_TO_COMFYUI_TYPE.get(key, any_type)
    return type_str


_gpu_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_cpu_device = torch.device("cpu")


def show_image(image: torch.Tensor, type: str = None):
    if type is None:
        retain_previews = config_service.get_config_value("easy_nodes.RetainPreviews", False)
        type = "output" if retain_previews else "temp"
    
    images = image
    for image in images:
        if len(image.shape) == 2:
            image = image.unsqueeze(-1)

        if image.shape[-1] == 1:
            image = torch.cat([image] * 3, axis=-1)

        image = image.cpu().numpy()

        image = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8))

        import folder_paths
        
        unique = hashlib.md5(image.tobytes()).hexdigest()[:8]

        filename = f"preview-{_curr_unique_id}_{unique}.png"
        
        # TODO: make configurable.
        subfolder = "ComfyUI-EasyNodes"
        full_output_path = Path(folder_paths.get_directory_by_type(type)) / subfolder / filename

        full_output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(str(full_output_path), compress_level=4)

        if "images" not in _curr_preview:
            _curr_preview["images"] = []
        _curr_preview["images"].append({"filename": filename, "subfolder": subfolder, "type": type})


def show_text(text: str):
    """Add a preview text to the ComfyUI node.

    Args:
        text (str): The text to display.
    """
    if "text" not in _curr_preview:
        _curr_preview["text"] = []
    _curr_preview["text"].append(text)


def _verify_values(config: EasyNodesConfig,
                   list_type: str, 
                   values: list[any], 
                   types: list[str], 
                   names: list[str], 
                   code_origin_loc: str, 
                   debug: bool=False):
    for i, val in enumerate(values):
        param_type = types[i]
        
        # It's a Choice.
        if isinstance(param_type, list):
            continue        
        
        if val is None:
            continue

        if debug:
            logging.info(f"Result {i} is {type(val)}, expected {types[i]}")
        def recursive_verify(verifier: callable, val: any):
            if isinstance(val, list):
                for v in val:
                    recursive_verify(verifier, v)
            else:
                verifier(val)

        # def verify(verifier: callable, val: any, return_name: str, severity: CheckSeverityMode):
    
        
        param_name = f"'{names[i]}'" if names else f"{list_type}_{i}"

        if param_type in _custom_verifiers:
            if config.verify_level in [CheckSeverityMode.WARN, CheckSeverityMode.FATAL]:
                try:
                    recursive_verify(_custom_verifiers[param_type], val)
                except Exception as e:
                    logging.error(_custom_verifiers)
                    logging.error(val)
                    error_str = f"Error verifying {list_type} {param_name}: {str(e)}\n{code_origin_loc}"
                    logging.warning(error_str)
                    if config.verify_level == CheckSeverityMode.FATAL:
                        raise ValueError(error_str) from None                        
        else:
            logging.warning(f"No verifier for {param_type}. Skipping verification.")


def _move_all_tensors_to_device(device: torch.device, tensor_or_tensors: Union[any, list]):
    if isinstance(tensor_or_tensors, torch.Tensor):
        return tensor_or_tensors.to(device)
    elif isinstance(tensor_or_tensors, list):
        return [_move_all_tensors_to_device(device, a) for a in tensor_or_tensors]
    return tensor_or_tensors


def _image_info(image: Union[torch.Tensor, np.ndarray]) -> str:
    if isinstance(image, torch.Tensor):
        if image.dtype in [torch.long, torch.int, torch.int32, torch.int64, torch.bool]:
            image = image.float()
        
        return (f"shape={image.shape} dtype={image.dtype} min={image.min()} max={image.max()}"
              + f" mean={image.mean()} sum={image.sum()} device={image.device}")
    elif isinstance(image, np.ndarray):
        return f"shape={image.shape}, dtype={image.dtype}, min={image.min()}, max={image.max()} mean={image.mean()} sum={image.sum()} "


class BufferHandler(logging.Handler):
    def __init__(self, buffer):
        logging.Handler.__init__(self)
        self.buffer = buffer
    
    def emit(self, record):
        msg = self.format(record)
        self.buffer.write(msg + '\n')


class Tee(object):
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    
    def flush(self):
        for f in self.files:
            f.flush()


def _compute_function_checksum(func_to_check):
    try:
        source_code = inspect.getsource(func_to_check)
    except Exception as e:
        logging.debug(f"Could not get source code for {func_to_check}: {e}")
        return 0
    return int(hashlib.sha256(source_code.encode('utf-8')).hexdigest(), 16)


def _register_function(func: callable, checksum, timestamp):
    if func.__qualname__ in _function_dict:
        assert _function_update_times[func.__qualname__] < timestamp, f"Function {func.__qualname__} already registered with later timestamp! {_function_update_times[func.__qualname__]} < {timestamp}"
        assert _function_checksums[func.__qualname__] != checksum, f"Function {func.__qualname__} already registered with same checksum! {_function_checksums[func.__qualname__]} == {checksum}"
    
    _function_dict[func.__qualname__] = func
    _function_checksums[func.__qualname__] = checksum
    _function_update_times[func.__qualname__] = timestamp


def _get_latest_version_of_module(module_name: str, debug: bool = False):
    if module_name not in _module_dict:
        _module_dict[module_name] = importlib.import_module(module_name)
    module = _module_dict[module_name]
    
    module_file = module.__file__
    
    # First reload the module if it needs to be reloaded.
    current_modified_time = os.path.getmtime(module_file)
    module_reload_time = _module_reload_times.get(module_name, 0)
    if current_modified_time > module_reload_time:
        time_diff = current_modified_time - module_reload_time
        logging.info(f"{Fore.LIGHTMAGENTA_EX}Reloading module {module_name} because file was edited. ({time_diff:.1f}s between versions){Fore.RESET}")
        # Set _has_prompt_been_requested so that any calls to ComfyFunc will get 
        # ignored rather than tripping the already-registered assert.
        global _after_first_prompt
        _after_first_prompt = True
        importlib.reload(module)
        _module_reload_times[module_name] = current_modified_time
    elif debug:
         logging.info(f"{module_name} up to date: {current_modified_time} vs {module_reload_time}")
    
    return module, current_modified_time


def _get_latest_version_of_func(func: callable, debug: bool = False):
    reload_modules = config_service.get_config_value("easy_nodes.ReloadOnEdit", False)
    if reload_modules and func.__module__:
        module, current_modified_time = _get_latest_version_of_module(func.__module__, debug)
        
        old_checksum = _function_checksums.get(func.__qualname__, 0)
        
        # Now pull the updated function from the module.
        last_function_update_time = _function_update_times.get(func.__qualname__, 0)
        if current_modified_time > last_function_update_time:
            time_diff = current_modified_time - last_function_update_time
            if hasattr(module, func.__name__):
                updated_func = getattr(module, func.__name__) 
                current_checksum = _compute_function_checksum(updated_func)
                if current_checksum != old_checksum:
                    logging.info(f"{Fore.LIGHTMAGENTA_EX}Updating {func.__qualname__} because function was modified. ({time_diff:.1f}s between versions){Fore.RESET}")
                    _register_function(updated_func, current_checksum, current_modified_time)
                elif debug:
                    logging.error(f"{func.__qualname__} up to date: {current_modified_time} vs {last_function_update_time}")
                    logging.error(inspect.getsource(_function_dict[func.__qualname__]))
    
    return _function_dict[func.__qualname__]


def _call_function_and_verify_result(config: EasyNodesConfig, func: callable, 
                                     args, kwargs, debug, input_desc, adjusted_return_types, 
                                     wrapped_name, return_names=None):
    try_count = 0
    llm_debugging_enabled = config_service.get_config_value("easy_nodes.llm_debugging", "Off") != "Off"
    max_tries = int(config_service.get_config_value("easy_nodes.max_tries", 1)) if llm_debugging_enabled == "AutoFix" else 1
    
    logging.debug(f"Running {func.__qualname__} with {max_tries} tries. {llm_debugging_enabled}")

    while try_count < max_tries:
        try_count += 1
        try:
            return_line_number = func.__code__.co_firstlineno
            node_logger = logging.getLogger()
            node_logger.setLevel(logging.INFO)
            buffer = io.StringIO()
            buffer_handler = BufferHandler(buffer)
            node_logger.addHandler(buffer_handler)
            buffer_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            sys.stdout = Tee(sys.stdout, buffer)

            _curr_preview.clear()
            result = func(*args, **kwargs)

            code_origin_loc = f"\n Source: {func.__qualname__} {func.__code__.co_filename}:{return_line_number}"
            num_expected_returns = len(adjusted_return_types)
            if num_expected_returns == 0:
                assert result is None, f"{wrapped_name}: Return value is not None, but no return type specified.\n{code_origin_loc}"
                return (None,)

            if not isinstance(result, tuple):
                result = (result,)
            assert len(result) == len(
                adjusted_return_types
            ), f"{wrapped_name}: Number of return values {len(result)} does not match number of return types {len(adjusted_return_types)}\n{code_origin_loc}"

            for i, ret in enumerate(result):
                if ret is None:
                    logging.warning(f"Result {i} is None")

            new_result = _move_all_tensors_to_device(_cpu_device, list(result)) if config.auto_move_tensors else list(result)
            _verify_values(config, "OUTPUT", result, adjusted_return_types, return_names, code_origin_loc, debug=debug)

            for i, ret in enumerate(new_result):
                new_result[i] = maybe_autoconvert(adjusted_return_types[i], ret)
            
            result = tuple(new_result)
            
            # If preview items were added, wrap the result.
            if _curr_preview:
                result = {"ui": _curr_preview.copy(), "result": result}
            return result

        except Exception as e:
            logging.error(f"Error while processing: {func}: {e}")
            if try_count == max_tries:
                # Calculate the number of interesting stack levels.
                _, _, tb = sys.exc_info()
                the_stack = traceback.extract_tb(tb)
                e.num_interesting_levels = len(the_stack) - 1
                logging.info(the_stack)
                
                formatted_stack = "\n".join(traceback.format_exception(type(e), e, tb))
                
                logging.warning(f"{formatted_stack}")
                
                raise e
            
            if llm_debugging_enabled:
                llm_debugging.process_exception_logic(func, e, input_desc, buffer)

        finally:
            node_logger.removeHandler(buffer_handler)
            sys.stdout = sys.__stdout__
            if buffer:
                buffer.close()

    assert False, "Should never reach this point"
    

def _ensure_package_dicts_exist(module_name: str):
    package_name = module_name.split('.')[-2]

    try:
        package = import_module(package_name)
        
        if not package.__file__.endswith("__init__.py"):
            raise ValueError(f"Package {package_name} is not a package. Cannot export.")

        if not hasattr(package, '__all__'):
            package.__all__ = []
            
        def add_if_not_there(dict_name):
            if dict_name not in package.__all__:
                package.__all__.append(dict_name)
            if not hasattr(package, dict_name):
                setattr(package, dict_name, {})
        
        add_if_not_there('NODE_CLASS_MAPPINGS')
        add_if_not_there('NODE_DISPLAY_NAME_MAPPINGS')
    except Exception as e:
        error_str = (f"Could not automatically find import package {package_name}. "
            + "Try initializing with easy_nodes.init(auto_register=False) and export manually in your __init__.py "
            + "with easy_nodes.get_node_mappings()")
        logging.error(error_str)
        raise e


def maybe_autoconvert(comfyui_type_name: str, arg: any):
    # Choices don't come with a registered type, they're just a list of strings.
    if isinstance(comfyui_type_name, list):
        return arg
    
    if _SHOULD_AUTOCONVERT.get(comfyui_type_name, False):
        comfyui_type = _COMFYUI_TYPE_TO_ANNOTATION_CLS[comfyui_type_name]
        if isinstance(arg, list):
            arg = [comfyui_type(el) for el in arg]
        else:
            arg = comfyui_type(arg)
    return arg


def ComfyNode(
    category: str = None,
    display_name: str = None,
    workflow_name: str = None,
    description: str = None,
    is_output_node: bool = False,
    return_types: list = None,
    return_names: list[str] = None,
    validate_inputs: Callable = None,
    is_changed: Callable = None,
    always_run: bool = False,
    debug: bool = False,
    color: str = None,
    bg_color: str = None,
    height: int = None,
    width: int = None,
):
    """
    Decorator function for creating ComfyUI nodes.

    Args:
        category (str): The category of the node.
        display_name (str): The display name of the node. If not provided, it will be generated from the function name.
        workflow_name (str): The workflow name of the node. If not provided, it will be generated from the function name.
        description (str): The description of the node. If not set, it will be generated from the function docstring.
        is_output_node (bool): Indicates whether the node is an output node and should be run regardless of if anything depends on it.
        return_types (list): A list of types to return. If not provided, it will be inferred from the function's annotations.
        return_names (list[str]): The names of the outputs. Must match the number of return types.
        validate_inputs (Callable): A function used to validate the inputs of the node.
        is_changed (Callable): A function used to determine if the node's inputs have changed.
        always_run (bool): Indicates whether the node should always run, regardless of whether its inputs have changed.
        debug (bool): Indicates whether to enable debug logging for this node.
        color (str): The color of the node.
        bg_color (str): The background color of the node.
        height (int): The default height of the node.
        width (int): The default width of the node.

    Returns:
        A callable used that can be used with a function to create a ComfyUI node.
    """
    curr_config = _get_curr_config()
    
    if not category:
        category = curr_config.default_category

    def decorator(func: callable):        
        if _after_first_prompt:
            # Sorry, we're closed for business.
            return func

        assert func.__qualname__ not in _function_dict, f"Function {func.__qualname__} already registered"

        if func.__qualname__ in _function_dict:
            return func

        modify_time = os.path.getmtime(func.__code__.co_filename) if os.path.exists(func.__code__.co_filename) else 0
        _module_reload_times[func.__module__] = modify_time
        _register_function(func, _compute_function_checksum(func), modify_time)
        
        filename = func.__code__.co_filename
        
        wrapped_name = func.__qualname__ + "_comfynode_wrapper"
        source_location = f"{filename}:{func.__code__.co_firstlineno}"
        code_origin_loc = f"\n Source: {func.__qualname__} {source_location}"
        original_is_changed = is_changed
        
        def wrapped_is_changed(*args, **kwargs):
            if always_run:
                logging.info(f"Always running {func.__qualname__}")
                return float("nan")
            
            unique_id = kwargs["unique_id"]
            updated_func = _get_latest_version_of_func(func, debug)
            current_checksum = _function_checksums[updated_func.__qualname__]
            if debug:
                logging.info(f"{func.__qualname__} {unique_id} is_changed: Checking if {original_is_changed} with args {args} and kwargs {kwargs.keys()}")
                for key in kwargs.keys():
                    logging.info(f"kwarg {key}: {type(kwargs[key])} {kwargs[key].shape if isinstance(kwargs[key], torch.Tensor) else ''}")
            
            try:
                if original_is_changed:
                    original_is_changed_params = inspect.signature(original_is_changed).parameters
                    filtered_kwargs = {key: value for key, value in kwargs.items() if key in original_is_changed_params}
                    original_num = original_is_changed(*args, **filtered_kwargs)
                    original_num = hash(original_num)
                else:
                    original_num = 0
                
                if math.isnan(original_num):
                    return float("nan")
            except Exception as e:
                logging.error(f"Error in is_changed function: {e} {func.__qualname__} {args} {kwargs.keys()}")
                raise e

            is_changed_val = current_checksum ^ original_num
            
            if debug:
                logging.info(f"{Fore.GREEN}{func.__qualname__}{Fore.RESET} {Fore.WHITE}{unique_id}{Fore.RESET} is_changed={Fore.LIGHTMAGENTA_EX}{is_changed_val}")
            return is_changed_val
        
        if debug:
            logger = logging.getLogger(wrapped_name)
            logger.info(
                "-------------------------------------------------------------------"
            )
            logger.info(f"Decorating {func.__qualname__}")

        node_class = _get_node_class(func)

        is_static = _is_static_method(node_class, func.__name__)
        is_cls_mth = _is_class_method(node_class, func.__name__)
        is_member = node_class is not None and not is_static and not is_cls_mth

        required_inputs, hidden_inputs, optional_inputs, input_is_list_map, input_type_map = (
            _infer_input_types_from_annotations(func, is_member, debug)
        )

        if debug:
            logger.info(f"{func.__name__} Is static: {is_static} Is member: {is_member} Class method: {is_cls_mth}")
            logger.info(f"Required inputs: {required_inputs} optional: {optional_inputs} input_is_list: {input_is_list_map} input_type_map: {input_type_map}")

        adjusted_return_types = []
        output_is_list = []
        if return_types is not None:
            adjusted_return_types, output_is_list = _infer_return_types_from_annotations(
                return_types, debug
            )
        else:
            adjusted_return_types, output_is_list = _infer_return_types_from_annotations(func, debug)

        if return_names:
            assert len(return_names) == len(
                adjusted_return_types
            ), f"Number of output names must match number of return types. Got {len(return_names)} names and {len(return_types)} return types."

        # There's not much point in a node that doesn't have any outputs
        # and isn't an output itself, so auto-promote in that case.
        force_output = len(adjusted_return_types) == 0
        name_parts = [x.title() for x in func.__name__.split("_")]
        input_is_list = any(input_is_list_map.values())
        
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if curr_config.auto_register == AutoRegisterSentinel.DEFAULT and curr_config.get_node_mappings_called is False:
                logging.warning("EasyNodes auto-registration not explicitly enabled, and easy_nodes.get_node_mappings() has not been called. "
                                + "In the future auto_register will default to False, so please set explicitly via easy_nodes.initialize_easy_nodes(auto_register=True), "
                                + "or use easy_nodes.get_node_mappings() to export to ComfyUI after all ComfyNodes have been created.")
            
            if debug:
                logger.info(
                    f"Calling {func.__name__} with {len(args)} args and {len(kwargs)} kwargs. Is class method: {is_cls_mth}"
                )
                for i, arg in enumerate(args):
                    logger.info(f"arg {i}: {type(arg)}")
                for key, arg in kwargs.items():
                    logger.info(f"kwarg {key}: {type(arg)}")

            all_inputs = {**required_inputs, **optional_inputs}

            input_desc = []
            keys = list(kwargs.keys())
            
            for key in keys:
                arg = kwargs[key]
                # Remove extra_pnginfo and unique_id from the kwargs if they weren't requested by the user.
                if key == "unique_id":
                    # logging.info(f"Setting unique_id to {arg}")
                    global _curr_unique_id
                    _curr_unique_id = arg
            
                if key not in param_names:
                    # logging.info(f"Removing extra kwarg {key}")
                    kwargs.pop(key)
                    continue
                
                arg = _move_all_tensors_to_device(_gpu_device, arg) if curr_config.auto_move_tensors else arg
                
                # TODO: Remove this special handling for mask once I remember what needed it.
                if (key in required_inputs and required_inputs[key][0] == "MASK"):
                    if isinstance(arg, torch.Tensor):
                        if len(arg.shape) == 2:
                            arg = arg.unsqueeze(0)
                    elif isinstance(arg, list):
                        for i, a in enumerate(arg):
                            if len(a.shape) == 2:
                                arg[i] = a.unsqueeze(0)

                # TODO: Move this into _call_function_and_verify_result 
                if key in all_inputs:
                    arg = maybe_autoconvert(all_inputs[key][0], arg)
                
                desc_name = _get_fully_qualified_name(type(arg))
                if isinstance(arg, torch.Tensor):
                    input_desc.append(f"{key} ({desc_name}): {_image_info(arg)}")
                else:
                    input_desc.append(f"{key} ({desc_name}): {arg}")
                
                kwargs[key] = arg
            
            # TODO: Move this into _call_function_and_verify_result
            input_names = [key for key in list(kwargs.keys()) if key not in hidden_inputs]
            input_values = [kwargs[key] for key in input_names]
            input_types = [all_inputs[key][0] for key in input_names]
            _verify_values(curr_config,
                         "INPUT", 
                         input_values,
                         input_types, 
                         input_names, 
                         code_origin_loc, debug=debug)

            # For some reason self still gets passed with class methods.
            if is_cls_mth:
                args = args[1:]

            # If the python function didn't annotate it as a list,
            # but INPUT_TYPES does, then we need to convert make it not a list.
            if input_is_list:
                for arg_name in kwargs.keys():
                    if debug:
                        logging.info(f"kwarg: {arg_name} {len(kwargs[arg_name])}")
                    if not input_is_list_map[arg_name]:
                        assert len(kwargs[arg_name]) == 1, f"Expected a single value for {arg_name}, but got {len(kwargs[arg_name])}"
                        kwargs[arg_name] = kwargs[arg_name][0]
            
            latest_func = _get_latest_version_of_func(func, debug)
            
            result = _call_function_and_verify_result(curr_config, latest_func, args, kwargs, debug, input_desc, adjusted_return_types, wrapped_name,
                                                      return_names=return_names)

            return result

        if node_class is None or is_static:
            wrapper = staticmethod(wrapper)

        if is_cls_mth:
            wrapper = classmethod(wrapper)

        the_description = description
        if the_description is None:
            the_description = ""
            if curr_config.docstring_mode is not AutoDescriptionMode.NONE and func.__doc__:
                the_description = func.__doc__.strip()
                if curr_config.docstring_mode == AutoDescriptionMode.BRIEF:
                    the_description = the_description.split("\n")[0]

        _create_comfy_node(
            wrapped_name,
            category,
            node_class,
            wrapper,
            display_name if display_name else " ".join(name_parts),
            workflow_name if workflow_name else "".join(name_parts),
            required_inputs,
            hidden_inputs,
            optional_inputs,
            input_is_list,
            adjusted_return_types,
            return_names,
            output_is_list,
            description=the_description,
            is_output_node=is_output_node or force_output,
            validate_inputs=validate_inputs,
            is_changed=wrapped_is_changed,
            color=color,
            bg_color=bg_color,
            height=height,
            width=width,
            debug=debug,
            source_location=source_location,
            easy_nodes_config=curr_config,
        )

        # Return the original function so it can still be used as normal (only ComfyUI sees the wrapper function).
        return func

    return decorator


def _annotate_input(
    param_name, annotation, default=inspect.Parameter.empty, debug=False
) -> tuple[tuple, bool, bool]:
    type_name = _get_type_str(annotation)
        
    if isinstance(default, Choice):
        return (default.choices,), False, False
    
    if debug:
        logging.info(f"{param_name} Default: {default} type: {type(default)} {isinstance(default, float)} {isinstance(default, NumberInput)}")
    
    if isinstance(default, str) and not isinstance(default, StringInput):
        default = StringInput(default)
    elif isinstance(default, (int, float)) and not isinstance(default, NumberInput):
        default = NumberInput(default)
    
    if isinstance(default, StringInput) or isinstance(default, NumberInput):
        return (type_name, default.to_dict()), default.optional, default.hidden

    metadata = {}
    if default is None:
        # If the user specified None explicitly, assume they're ok with it being optional.
        metadata["optional"] = True
        metadata["forceInput"] = True        
    elif default == inspect.Parameter.empty:
        # If they didn't give it a default value at all, then forceInput so that the UI
        # doesn't end up giving them a default that they may not want.
        metadata["forceInput"] = True
    else:
        metadata["default"] = default
    
    # This is the exception where they may have given it a default, but we still
    # want to force it as an input because changing that value will be rare.
    if _DEFAULT_FORCE_INPUT.get(_get_fully_qualified_name(annotation), False):
        metadata["forceInput"] = True

    return (type_name, metadata), default != inspect.Parameter.empty, False


def _infer_input_types_from_annotations(func, skip_first, debug=False):
    """
    Infer input types based on function annotations.
    """
    input_is_list = {}
    input_type_map = {}
    sig = inspect.signature(func)
    required_inputs = {}
    hidden_input_types = {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"}
    optional_input_types = {}

    params = list(sig.parameters.items())

    if debug:
        logging.info(f"ALL PARAMS {params}")

    if skip_first:
        if debug:
            logging.info(f"SKIPPING FIRST PARAM {params[0]}")
        params = params[1:]

    for param_name, param in params:
        origin = get_origin(param.annotation)
        input_is_list[param_name] = origin is list
        input_type_map[param_name] = param.annotation

        if debug:
            logging.info(f"Param default: {param.default}")

        the_param, is_optional, is_hidden = _annotate_input(param_name, param.annotation, param.default, debug)
        
        if param_name == "unique_id" or param_name == "extra_pnginfo":
            pass
        elif not is_optional:
            required_inputs[param_name] = the_param
        elif is_hidden:
            hidden_input_types[param_name] = the_param
        else:
            optional_input_types[param_name] = the_param
    return required_inputs, hidden_input_types, optional_input_types, input_is_list, input_type_map


def _infer_return_types_from_annotations(func_or_types, debug=False):
    """
    Infer whether each element in a function's return tuple is a list or a single item,
    handling direct list inputs as well as function annotations.
    """
    if isinstance(func_or_types, list):
        # Direct list of types provided
        return_args = func_or_types
        origin = tuple  # Assume tuple if directly provided with a list
    else:
        # Assuming it's a function, inspect its return annotation
        return_annotation = inspect.signature(func_or_types).return_annotation
        return_args = get_args(return_annotation)
        origin = get_origin(return_annotation)

        if debug:
            logging.info(f"return_annotation: '{return_annotation}'")
            logging.info(f"return_args: '{return_args}'")
            logging.info(f"origin: '{origin}'")
            logging.info(f"{type(return_annotation)}, {return_annotation}")

    types_mapped = []
    output_is_list = []

    if origin is tuple:
        for arg in return_args:
            if get_origin(arg) is list:
                output_is_list.append(True)
                list_arg = get_args(arg)[0]
                types_mapped.append(_get_type_str(list_arg))
            else:
                output_is_list.append(False)
                types_mapped.append(_get_type_str(arg))
    elif origin is list:
        if debug:
            logging.info(_get_type_str(return_annotation))
            logging.info(return_annotation)
            logging.info(return_args)
        types_mapped.append(_get_type_str(return_args[0]))
        output_is_list.append(origin is list)
    elif return_annotation is not inspect.Parameter.empty:
        types_mapped.append(_get_type_str(return_annotation))
        output_is_list.append(False)

    return_types_tuple = tuple(types_mapped)
    output_is_lists_tuple = tuple(output_is_list)
    if debug:
        logging.info(
            f"return_types_tuple: '{return_types_tuple}', output_is_lists_tuple: '{output_is_lists_tuple}'"
        )

    return return_types_tuple, output_is_lists_tuple


def hex_to_color(color: str) -> list[float]:
    col = color.strip('#').strip().upper()
    assert len(col) == 6, f"Color must be a hex color code, got {color}"
    assert all(c in "0123456789ABCDEF" for c in col), f"Color must be a hex color code, got {color}"
    color_rgb = [int(col[i : i + 2], 16) for i in [0, 2, 4]]
    return color_rgb


def _create_comfy_node(
    cname,
    category,
    node_class,
    process_function,
    display_name,
    workflow_name,
    required_inputs,
    hidden_inputs,
    optional_inputs,
    input_is_list,
    return_types,
    return_names,
    output_is_list,
    description=None,
    is_output_node=False,
    validate_inputs=None,
    is_changed=None,
    color=None,
    bg_color=None,
    height=None,
    width=None,
    source_location=None,
    debug=False,
    easy_nodes_config: EasyNodesConfig=None,
):
    all_inputs = {"required": required_inputs, "hidden": hidden_inputs, "optional": optional_inputs}
    
    node_info = {}
    if color is not None:
        color_rgb = hex_to_color(color)
        node_info["color"] = color
        if not bg_color:
            bg_color = "#" + "".join(f"{int(c * 0.6):02X}" for c in color_rgb)
            
    if bg_color is not None:
        _ = hex_to_color(bg_color)  # Check that it's a valid color
        node_info["bgColor"] = bg_color

    if height is not None:
        node_info["height"] = height

    if width is not None:
        node_info["width"] = width

    if source_location is not None:
        node_info["sourceLocation"] = source_location

    # Smuggle it in with the description. A bit hacky, but it works and I 
    # don't know of a better way to do it without modifying the ComfyUI code.
    if node_info:
        description = f"EasyNodesInfo={json.dumps(node_info)}\n" + description

    # Initial class dictionary setup
    class_dict = {
        "INPUT_TYPES": classmethod(lambda cls: all_inputs),
        "CATEGORY": category,
        "RETURN_TYPES": return_types,
        "FUNCTION": cname,
        "INPUT_IS_LIST": input_is_list,
        "OUTPUT_IS_LIST": output_is_list,
        "OUTPUT_NODE": is_output_node,
        "RETURN_NAMES": return_names,
        "VALIDATE_INPUTS": validate_inputs,
        "IS_CHANGED": is_changed,
        "DESCRIPTION": description,
        cname: process_function,
    }
    class_dict = {k: v for k, v in class_dict.items() if v is not None}

    if debug:
        logger = logging.getLogger(cname)
        for key, value in class_dict.items():
            logger.info(f"{key}: {value}")
            
    class_map = easy_nodes_config.NODE_CLASS_MAPPINGS
    display_map = easy_nodes_config.NODE_DISPLAY_NAME_MAPPINGS

    if not _after_first_prompt:
        all_workflow_names = set(class_map.keys()) | set(comfyui_nodes.NODE_CLASS_MAPPINGS.keys())
        all_display_names = set(display_map.values()) | set(comfyui_nodes.NODE_DISPLAY_NAME_MAPPINGS.values())
        all_node_classes = set(class_map.values()) | set(comfyui_nodes.NODE_CLASS_MAPPINGS.values())

        assert workflow_name not in all_workflow_names, f"Node class '{workflow_name} ({cname})' already exists!"
        assert display_name not in all_display_names, f"Display name '{display_name}' already exists!"
        assert node_class not in all_node_classes, f"Only one method from '{node_class}' can be used as a ComfyUI node."

    if node_class:
        for key, value in class_dict.items():
            setattr(node_class, key, value)
    else:
        node_class = type(workflow_name, (object,), class_dict)

    class_map[workflow_name] = node_class
    display_map[workflow_name] = display_name
    
    # Temporary for backwards compatibility.
    if easy_nodes_config.auto_register is AutoRegisterSentinel.DEFAULT:
        comfyui_nodes.NODE_CLASS_MAPPINGS[workflow_name] = node_class
        comfyui_nodes.NODE_DISPLAY_NAME_MAPPINGS[workflow_name] = display_name
    
    easy_nodes_config.num_registered += 1


def _is_static_method(cls, attr):
    """Check if a method is a static method."""
    if cls is None:
        return False
    attr_value = inspect.getattr_static(cls, attr, None)
    is_static = isinstance(attr_value, staticmethod)
    return is_static


def _is_class_method(cls, attr):
    if cls is None:
        return False
    attr_value = inspect.getattr_static(cls, attr, None)
    is_class_method = isinstance(attr_value, classmethod)
    return is_class_method


def _get_node_class(func):
    split_name = func.__qualname__.split(".")

    if len(split_name) > 1:
        class_name = split_name[-2]
        node_class = globals().get(class_name, None)
        if node_class is None and hasattr(func, "__globals__"):
            node_class = func.__globals__.get(class_name, None)
        return node_class
    return None


T = typing.TypeVar("T")


def create_field_setter_node(cls: type, category=None, debug=False) -> typing.Callable[..., T]:
    if category is None:
        category = _get_curr_config().default_category
    if debug:
        logging.info(f"Registering setter for class '{cls.__name__}'")
    key = _get_fully_qualified_name(cls)
    assert key in _ANNOTATION_TO_COMFYUI_TYPE, f"Type '{key}' not registered with ComfyUI, call register_type() and give it a name first."
    dynamic_function = _create_dynamic_setter(cls, debug=debug)
    ComfyNode(category, display_name=cls.__name__, workflow_name=cls.__name__, debug=debug)(
        dynamic_function)


def _create_dynamic_setter(cls: type, debug=False) -> typing.Callable[..., T]:
    obj = cls()
    func_name = cls.__name__
    setter_name = func_name + "_setter"
    
    properties = {}
    all_type_names = set([])

    # Collect properties and infer types from their current instantiated values.
    for attr_name in dir(obj):
        attr = getattr(obj, attr_name, None)
        if attr is not None and not callable(attr) and not attr_name.startswith("__"):
            if isinstance(attr, property) and attr.fset is not None:
                # Handle properties
                current_value = getattr(obj, attr_name, None)
                prop_type = type(current_value) if current_value is not None else typing.Any
                properties[attr_name] = (prop_type, current_value)

                if debug:
                    logging.info(
                        f"Property '{attr_name}' has type '{prop_type}' and value '{current_value}'"
                    )
            else:
                # Handle instance attributes
                current_value = getattr(obj, attr_name, None)
                prop_type = type(current_value) if current_value is not None else typing.Any
                properties[attr_name] = (prop_type, current_value)

                if debug:
                    logging.info(
                        f"Instance attribute '{attr_name}' has type '{prop_type}' and value '{current_value}'"
                    )

            # Automatically register the type and its subtypes, allowing duplicates
            register_type(
                prop_type, _get_fully_qualified_name(prop_type), is_auto_register=True
            )
            if hasattr(prop_type, "__args__"):
                for subtype in prop_type.__args__:
                    register_type(
                        subtype,
                        _get_fully_qualified_name(subtype),
                        is_auto_register=True,
                    )

            # Extract module name from the property type
            fully_qualled_name = _get_fully_qualified_name(prop_type)
            if "." in fully_qualled_name:
                all_type_names.add(fully_qualled_name)

    def get_default_value(prop_type, current_value):
        default_values = {
            int: f"NumberInput({current_value})",
            float: f"NumberInput({current_value}, -1000000, 10000000, 0.0001)",
            str: f"StringInput('{current_value}')",
            bool: f"{current_value}",
        }
        return default_values.get(prop_type, "None")

    func_params = []
    for prop, (prop_type, current_value) in properties.items():
        qualified_type_name = _get_fully_qualified_name(prop_type).replace('builtins.', '')
        default_value = get_default_value(prop_type, current_value)
        func_params.append(f"{prop}: {qualified_type_name}={default_value}")

    def_str = f"def {setter_name}("
    join_str = ",\n" + " " * len(def_str)
    func_params_str = join_str.join(func_params)

    # Generate import statements
    import_statements = [
        "import typing",
        "import importlib",
        "from easy_nodes import NumberInput, StringInput",
        # "import example.example_nodes",
    ]
    
    for module_name in all_type_names:
        if module_name.startswith("builtins."):
            continue
        package_name, type_name = module_name.rsplit(".", 1)
        import_statements.append(f"import {package_name}")
        
    # Alphabetize them and make unique
    import_statements = sorted(list(dict.fromkeys(import_statements)))

    func_body_lines = [
        f"cls = getattr(module, '{func_name}')",
        "new_obj = cls()",
    ]
    for prop in properties.keys():
        func_body_lines.append(f"if {prop} is not None: setattr(new_obj, '{prop}', {prop})")
    func_body_lines.append("return new_obj")
    func_body_lines = [f"    {line}" for line in func_body_lines]
    
    func_lines = import_statements + [f"module = importlib.import_module('{cls.__module__}')",
                                     f"{def_str}{func_params_str}) -> module.{cls.__name__}:"] + func_body_lines 
    func_code = "\n".join(func_lines)

    if debug:
        logging.info(f"Creating dynamic setter with code: '{func_code}'")

    globals_dict = {
        "typing": typing,
        "importlib": importlib,
        "NumberInput": NumberInput,
        "StringInput": StringInput,
        "module": importlib.import_module(cls.__module__),
    }
    locals_dict = {}

    # Update the global namespace with the module names
    # for module_name in module_names:
    #     globals_dict[module_name] = importlib.import_module(module_name)
    
    # Execute the function code
    exec(func_code, globals_dict, locals_dict)

    # Get the function object from the local namespace
    func = locals_dict[setter_name]

    return func
