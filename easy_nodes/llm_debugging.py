import difflib
import os
import shutil
import sys
import inspect
import logging
import re
import subprocess
import tempfile
import traceback

from colorama import Fore, Style
from openai import OpenAI

import easy_nodes.config_service as config_service


def create_openai_client() -> OpenAI:
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    if not openai_key:
        raise ValueError("OpenAI API key not found in OPENAI_API_KEY environment variable. "
                         + "Please set the API key to use LLM debugging.")
    return OpenAI(api_key=openai_key)


chatgpt_role_description = """
You are tasked as a Python debugging assistant, focusing solely on correcting provided function code. Your responses should exclusively consist of the amended function (including decorators), including any essential comments on modifications made. Ensure the function signature remains intact.

- Directly return one module-level element (function, class, method, module, etc.) per response. If an element does not need updating, do not return it.
- Limit responses to the corrected function, with changes clearly commented within.
- Incorporate new imports at the function's beginning if they're necessary for introduced logic or corrections.
- Assume all mentioned entities and functions are already defined and pre-imported in the function's context. Only import *new* elements. In such cases, prepend imports to the function body in place of the explicit comment marker referring to them.
- If a single-step fix isn't feasible, integrate debug logging to aid further troubleshooting, annotating these additions for potential later removal.
- Avoid markdown or other formatting in your response to ensure seamless file integration.
- Recognize that terms like Matrix, ImageTensor, DepthMap and MaskTensor refer to torch.Tensor annotations and don't necessitate additional imports or actions.
- Image tensors should be shape BxHxWx3, mask tensors should be BxHxW, and DepthMaps should be BxHxWx1, where B is the batch size, H is the height, and W is the width.
- Include any decorators that the function had attached.
- If a comment starts with "NOTE(GPT):", it is a note for you and should not be included in the final output.
- If you're not positive what the issue is, it's fine to add copious logging to help diagnose the problem. Output with logging.debug calls.
- If you can't provide any improvements at all, return a message starting with "! Unable to fix this:" and then provide a brief explanation of why it can't be fixed.
Acceptable reasons being that the problem exists outside the function's scope or that the function is being invoked incorrectly.
"""


def extract_function_or_class_name(source):
    """
    Extracts the name of the function or class from the given source code.
    Parameters:
    - source: The source code of the function or class.
    Returns:
    - The name of the function or class.
    """
    # Regular expression pattern to match function or class names
    pattern = r'(?:def|class)\s+(\w+)'

    # Search for the pattern in the source code
    match = re.search(pattern, source)

    if match:
        return match.group(1)
    else:
        return None


def module_to_file_path(module_name):
    """
    Converts a module name to its corresponding file path.

    Parameters:
    - module_name: The name of the module.

    Returns:
    - The file path of the module.
    """
    # Get the module object based on the module name
    module_obj = sys.modules.get(module_name)

    if module_obj is None:
        # If the module is not found, raise an error or return None
        raise ValueError(f"Module '{module_name}' not found.")

    # Get the file path of the module
    file_path = getattr(module_obj, '__file__', None)

    if file_path is None:
        # If the file path is not available, raise an error or return None
        raise ValueError(f"File path not found for module '{module_name}'.")

    # Normalize the file path
    file_path = os.path.abspath(file_path)

    # If the file path ends with '.pyc' or '.pyo', remove the 'c' or 'o' extension
    if file_path.endswith(('.pyc', '.pyo')):
        file_path = file_path[:-1]

    return file_path


def split_top_level_entries(code):
    """
    Splits the code into top-level entries (classes, functions, and statements).
    Parameters:
    - code: The code containing top-level entries.
    Returns:
    - A list of top-level entry code strings.
    """
    entries = []
    current_entry = []
    start_indices = []

    for i, line in enumerate(code.split('\n')):
        indent_level = len(line) - len(line.lstrip())
        line_empty = len(line.strip()) == 0

        # It's a new entry if the line has an indent level of 0 and is not empty or a comment,
        # and the previous line is not a decorator
        if (
            indent_level == 0
            and not line_empty
            and not line.strip().startswith('#')
            and (not current_entry or not current_entry[-1].strip().startswith('@'))
        ):
            if current_entry:
                entries.append('\n'.join(current_entry))
            current_entry = []
            start_indices.append(i)

        current_entry.append(line)

    if current_entry:
        entries.append('\n'.join(current_entry))

    return entries, start_indices


def replace_source_with_updates(entry_code: str, original_source: dict[str, list[str]]):
    entry_name = extract_function_or_class_name(entry_code)
    logging.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    logging.info(f"Replacing entry {entry_name}")
    logging.info("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # Find the original source code for the entry
    for module_name, source_items in original_source.items():
        for source_item in source_items:
            if f"def {entry_name}(" in source_item or f"class {entry_name}(" in source_item:
                original_entry_code = source_item
                original_module_name = module_name
    
    assert original_entry_code is not None, f"Original source code for {entry_name} not found"
    assert original_module_name is not None, f"Original module name for {entry_name} not found"

    # Replace the original entry code with the modified entry code
    modified_source = original_entry_code.replace(original_entry_code, entry_code)

    # Determine the file path based on the module name
    original_file_path = module_to_file_path(original_module_name)

    # Backup the full file first
    backup_file_path = create_backup_file(original_file_path)
    logging.info(f"Backed up {original_file_path} to '{backup_file_path}'")

    # Write the modified source code to a temp file in /tmp directory
    tmp_file_path = replace_function_in_file(original_file_path, modified_source)
    logging.info(f"Modified entry written to temporary file for review: {tmp_file_path}")

    # Create a diff between the original and modified source
    _, patch_file = tempfile.mkstemp(suffix=".patch")
    create_patch(original_file_path, tmp_file_path, patch_file)
    logging.info(f"Patch file created: {patch_file}")
    print_patch_with_color(patch_file)

    if config_service.get_config_value("easy_nodes.llm_debugging", "Off") == "AutoFix":
        logging.info("Applying the patch to the original file...")
        # Apply the patch to a copy of the original file to test changes
        _, patched_file = tempfile.mkstemp()
        apply_patch(original_file_path, patch_file, patched_file)

        verify_same(tmp_file_path, patched_file)
        
        # If the diff was applied correctly, consider updating the original file
        shutil.copy(patched_file, original_file_path)
        logging.info(f"Original file '{original_file_path}' updated with the patch.")
    else:
        logging.info("Skipping automatic patch application. Set 'easy_nodes.llm_debugging' to 'AutoFix' to apply the patch.")


def process_exception_logic(func, exception, input_desc, buffer):
    """
    Processes an exception by generating a patch using OpenAI's suggestions and applying it.

    Parameters:
    - func: The function that raised the exception.
    - exception: The exception instance.
    - input_desc: Description of the input for creating the chat prompt.
    - buffer: StringIO buffer capturing stdout and logging output.
    """
    buffer_content = buffer.getvalue()
    prompt, original_source = create_llm_prompt(func, input_desc, buffer_content, exception)

    # Prepare the chat prompt for OpenAI
    messages = [
        {"role": "system", "content": chatgpt_role_description},
        {"role": "user", "content": prompt}
    ]
    
    openai_client = create_openai_client()
    
    model_name = config_service.get_config_value("easy_nodes.llm_model", "gpt-4o")
    
    # Send the prompt to OpenAI and get the response
    # Assuming send_prompt_to_openai returns a structured response with the modified function code
    response = send_prompt_to_openai(
        client=openai_client,
        max_tokens=4096,
        model=model_name,
        messages=messages,
        verbose=True
    )

    # # Process the response to extract the modified source code
    function_code = response.choices[0].message.content
    # function_code = canned_response

    if function_code.strip().startswith("!"):
        logging.error(f"OpenAI was unable to provide a fix for the function: {function_code}")
        return

    function_code = remove_code_block_delimiters(function_code)
    logging.info(f"Modified function code:\n{function_code}")
    
    function_code = function_code[function_code.index("@ComfyFunc"):]

    # Split the function_code into top-level entries (classes and functions)
    top_level_entries, _ = split_top_level_entries(function_code)

    for entry_code in top_level_entries:
        replace_source_with_updates(entry_code, original_source)        


def create_patch(file_a_path, file_b_path, patch_file_path):
    """
    Creates a patch file that can be applied to file A to produce file B.

    Parameters:
    - file_a_path: Path to the original file (file A).
    - file_b_path: Path to the modified file (file B).
    - patch_file_path: Path where the patch file will be saved.
    """
    with open(file_a_path, 'r') as file_a:
        file_a_lines = [line.rstrip('\n') for line in file_a.readlines()]

    with open(file_b_path, 'r') as file_b:
        file_b_lines = [line.rstrip('\n') for line in file_b.readlines()]

    diff = difflib.unified_diff(
        file_a_lines, file_b_lines,
        fromfile=file_a_path, tofile=file_b_path,
        lineterm=''
    )

    with open(patch_file_path, 'w') as patch_file:
        for line in diff:
            patch_file.write(line + '\n')


def find_first_indented_line(the_source):
    """
    Finds the first indented line in the_source and returns a tuple containing
    the character position of the start of the line and the indentation level.
    """
    char_location = 0
    for line in the_source.split("\n"):
        if line.strip() and not line.strip().startswith("#"):
            indent_level = len(line) - len(line.lstrip())
            if indent_level > 0:
                return (char_location, indent_level)
        char_location += len(line) + 1
    return None


def print_patch_with_color(patch_file_path):
    """
    Prints the contents of a patch file with color highlighting using logging and colorama.

    Parameters:
    - patch_file_path: Path to the patch file containing changes.
    """
    # Open and read the patch file
    with open(patch_file_path, 'r') as patch_file:
        patch_lines = patch_file.readlines()

    # Iterate through each line in the patch file
    for line in patch_lines:
        if line.startswith('+'):
            logging.info(Fore.GREEN + line.rstrip())  # Green for additions
        elif line.startswith('-'):
            logging.info(Fore.RED + line.rstrip())    # Red for deletions
        elif line.startswith('@'):
            logging.info(Fore.CYAN + line.rstrip())   # Cyan for headers
        else:
            logging.info(line.rstrip())               # Default color for context and other lines


def apply_patch(original_file_path, patch_file_path, output_file_path):
    """
    Applies a patch file to an original file and writes the result to an output file.

    Parameters:
    - original_file_path: Path to the original file to which the patch will be applied.
    - patch_file_path: Path to the patch file containing changes.
    - output_file_path: Path where the modified file will be written.
    """
    cmd = ['patch', original_file_path, patch_file_path, '-o', output_file_path]
    subprocess.run(cmd, check=True)


def replace_function_in_file(file_path, modified_source):
    with open(file_path, 'r') as file:
        file_source = file.read()

    entries, chunk_starts = split_top_level_entries(file_source)
    what_to_replace = extract_function_or_class_name(modified_source)

    for i, entry in enumerate(entries):
        if extract_function_or_class_name(entry) == what_to_replace:
            entries[i] = modified_source + "\n"
            break
    else:
        assert False, f"Function or class {what_to_replace} not found in file {file_path}"

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
        new_text = "\n".join(entries)
        tmp_file.write(new_text)
        tmp_file_path = tmp_file.name

    return tmp_file_path
    

def send_prompt_to_openai(client: OpenAI, messages: list[dict], max_tokens: int, model: str, verbose: bool):
    # Validate messages format
    if not all(isinstance(message, dict) and 'role' in message and 'content' in message for message in messages):
        raise ValueError("All messages must be dictionaries with 'role' and 'content' keys")

    if verbose:
        print(Fore.BLUE + "Sending the following prompt to ChatGPT:" + Style.RESET_ALL)
        for message in messages:
            print(
                Fore.LIGHTBLACK_EX
                + f"{message['role'].title()}: {message['content']}"
                + Style.RESET_ALL
            )

    response = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens
    )
    return response


def get_source_from_exception_and_callable(exception, callable_obj, allowed_paths):
    source_dict = {}
    seen_names = set()
    
    tb = exception.__traceback__

    # Convert allowed paths to absolute paths
    allowed_paths = [os.path.abspath(path) for path in allowed_paths]

    

    def add_exception_marker(source, lineno):
        if 0 <= lineno < len(source):
            source[lineno] = source[lineno].rstrip() + " # <------- NOTE(GPT): Exception here\n"

    def process_frame(frame, lineno, is_exception=True):
        module_name = frame.f_globals.get('__name__')
        file_path = os.path.abspath(frame.f_code.co_filename)
        if any(file_path.startswith(path) for path in allowed_paths):
            if module_name not in source_dict:
                source_dict[module_name] = []
            try:
                # Check if the frame is associated with a class method
                class_name = frame.f_code.co_name.split('.')[0]
                if class_name != '<module>':
                    # Retrieve the source code for the entire class
                    class_obj = frame.f_locals.get('self').__class__
                    try:
                        source, _ = inspect.getsourcelines(class_obj)
                        # Find the start line number of the class
                        class_start_lineno = inspect.getsourcelines(class_obj)[1]
                        # Adjust the line number relative to the class start
                        class_lineno = lineno - class_start_lineno
                        if 0 <= class_lineno < len(source) and is_exception:
                            add_exception_marker(source, class_lineno)
                    except TypeError:
                        # Handle the case when class_obj is None
                        source, start_lineno = inspect.getsourcelines(frame)
                        if start_lineno <= lineno < start_lineno + len(source) and is_exception:
                            add_exception_marker(source, lineno - start_lineno)
                else:
                    # Retrieve the source code for the function or module level
                    source, start_lineno = inspect.getsourcelines(frame)
                    if start_lineno <= lineno < start_lineno + len(source) and is_exception:
                        add_exception_marker(source, lineno - start_lineno)
                if ''.join(source) not in source_dict[module_name]:
                    source_dict[module_name].append(''.join(source).strip())
                    seen_names.add(module_name + '.' + frame.f_code.co_name)
            except OSError:
                pass

    # Process exception traceback
    while tb is not None:
        frame = tb.tb_frame
        lineno = frame.f_lineno
        process_frame(frame, lineno)
        tb = tb.tb_next

    # Process callable if it's not already in the source_dict
    callable_module_name = callable_obj.__module__
    if callable_module_name not in source_dict:
        source_dict[callable_module_name] = []
    try:
        callable_source, _ = inspect.getsourcelines(callable_obj)
        callable_source_str = ''.join(callable_source).strip()
        
        global_name = callable_module_name + '.' + callable_obj.__name__
        logging.info(f"Global name: {global_name}")
        if global_name not in seen_names:            
            # logging.info(f"Retrieving source for {callable_obj.__name__} : {callable_source_str}")
            # logging.info(f"Keys: {source_dict.keys()}")
            if callable_source_str not in source_dict[callable_module_name]:
                source_dict[callable_module_name].append(callable_source_str)
        else:
            logging.info(f"Skipping {callable_obj.__name__} as it's already in the source_dict")
    except OSError:
        logging.error(f"Failed to retrieve source for {callable_obj.__name__}")
    
    logging.error(f"Seen names: {seen_names}")

    return source_dict


def create_llm_prompt(func, input_desc, buffer_content, e: Exception) -> tuple[str, dict[str, list[str]]]:
    """
    Creates a prompt for the ChatGPT based on function details, input descriptions, 
    execution logs, and the encountered exception.

    Args:
        func (Callable): The function that raised an exception during execution.
        input_desc (List[str]): Descriptions of the function's input parameters.
        buffer_content (str): Content captured from the execution's stdout and logging.
        e (Exception): The exception that was raised during function execution.

    Returns:
        List[str]: A list of strings composing the complete ChatGPT prompt.
    """    
    original_source = get_source_from_exception_and_callable(e, func, [os.path.dirname(func.__code__.co_filename)])
    combined_source = ""
    for k, v in original_source.items():
        logging.info(f"Original source for {k}")
        for item in v:
            combined_source += item + "\n\n\n"
        
    chat_gpt_prompt = [
        f"Details for function {func.__name__} in file {func.__code__.co_filename}:",
        "----------------------------------------------------",
        "Function argument name, type, value:",
        "    " + "\n    ".join(input_desc),
        "----------------------------------------------------",
        "Error:",
        f"```\n{str(e)}\n```",
        "----------------------------------------------------",
        "Stack trace:",
        f"```\n{traceback.format_exc()}\n```",
        "----------------------------------------------------",
        "Execution log:",
        f"```\n{buffer_content}```",
        "----------------------------------------------------",
        "Function source code:",
        f"```\n{combined_source}```",
    ]
    return "\n".join(chat_gpt_prompt), original_source


def create_backup_file(original_file_path):
    """
    Creates a backup of the given file with a numeric suffix (.bak.N),
    where N is a three-digit number starting with 000.

    Args:
        original_file_path (str): The path to the original file to back up.

    Returns:
        str: The path to the created backup file.
    """
    base_name = original_file_path + ".bak"
    suffix = 0
    backup_file_path = f"{base_name}.{suffix:03d}"

    # Increment the suffix if the backup file already exists
    while os.path.exists(backup_file_path):
        suffix += 1
        backup_file_path = f"{base_name}.{suffix:03d}"

    # Copy the original file to the new backup file path
    with open(original_file_path, "rb") as original_file:
        with open(backup_file_path, "wb") as backup_file:
            backup_file.write(original_file.read())

    logging.info(f"Backed up '{original_file_path}' to '{backup_file_path}'")
    return backup_file_path


def remove_code_block_delimiters(text):
    """
    Remove ```python at the start and ``` at the end of a code block.

    :param text: The text containing the code block delimiters.
    :return: The text with the code block delimiters removed.
    """
    # This regex pattern matches ```python at the start of the string and ``` at the end of the string.
    pattern = r'^```python\n|\n```$'
    cleaned_text = re.sub(pattern, '', text, flags=re.MULTILINE)
    return cleaned_text


def verify_same(file_a_path, file_b_path):
    # Now make sure that both files are identical
    with open(file_a_path, 'r') as new_file:
        updated_content_lines = new_file.readlines()

    with open(file_b_path, 'r') as tmp_file:
        patched_content_lines = tmp_file.readlines()

    # Use difflib to find differences
    differences = list(difflib.unified_diff(
        updated_content_lines, patched_content_lines,
        fromfile='updated_file', tofile='patched_file',
        lineterm=''
    ))

    if differences:
        for line in differences[:10]:
            logging.error(f"DIFFERENCE: {line}")
        assert False, f"The diff was not applied correctly between {file_a_path} and {file_b_path}"
    else:
        logging.info("Files are identical, no differences found.")
