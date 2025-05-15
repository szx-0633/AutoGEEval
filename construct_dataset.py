from call_language_model import call_language_model
from prompts import CONSTRUCT_ATOMIC
import os
import json
import re
import pandas as pd
import shutil


def generate_gee_atomic_test_code(operators_file, output_path, model_name="qwen-max-2025-01-25", model_provider="aliyun"):
    """
    Generates GEE atomic test code based on an Excel file of operators.

    Args:
        operators_file (str): Path to the Excel file containing operator information.
        output_path (str): Directory to save the generated test code files.
        model_name (str, optional): Name of the language model to use. Defaults to "qwen-max-2025-01-25".
        model_provider (str, optional): Provider of the language model. Defaults to "aliyun".
    """
    os.makedirs(output_path, exist_ok=True)

    operators_data = pd.read_excel(operators_file)
    operators_data = operators_data[["full_name","description","output_type","parameters"]]

    count = 0

    for index, row in operators_data.iterrows():
        # construct json information
        operator_name = row["full_name"]
        explanation = row["description"]
        return_type = row["output_type"]
        parameters = row["parameters"]

        operator_json = {
            "operator_name": operator_name,
            "explanation": explanation,
            "return_type": return_type,
            "parameters": parameters
        }

        out_file_path = os.path.join(output_path, f"{operator_name}.txt")

        # Skip if the output already exists
        if os.path.exists(out_file_path): # Used out_file_path directly
            print(f"Skipping {operator_name} as output file already exists.")
            count += 1
            continue

        output, token_usage, error = call_language_model(
            model_provider=model_provider,
            model_name=model_name,
            system_prompt="You are an expert in geospatial analysis with python and Google Earth Engine(GEE).",
            user_prompt=CONSTRUCT_ATOMIC+json.dumps(operator_json),
            temperature=0.1,
            max_tokens=8192,
            config_path=r"./llm_config.yaml")

        if error: # Added error handling
            print(f"Error processing {operator_name}: {error}")
            continue

        with open(out_file_path, "w", encoding='utf-8') as f:
            f.write(output)

        print(f"Processed {operator_name}")
        print(f"Token usage: {token_usage}")
        count += 1
        print(f"Total processed so far: {count}")


def process_raw_file(raw_file_path, code_path, config_path):
    """
    Processes raw .txt files to extract Python code and YAML test cases.

    Args:
        raw_file_path (str): Path to the directory containing raw .txt files.
        code_path (str): Path to save the extracted Python code files.
        config_path (str): Path to save the combined YAML file (including filename).
    """

    # Ensure save paths exist
    os.makedirs(code_path, exist_ok=True)
    # Initialize a string to store all YAML content
    all_yaml_content = ""
    processed_file_count = 0 # Renamed 'count' for clarity
    function_names = set()  # Used to store function names to avoid duplicates

    # Iterate through all .txt files
    for file_name in os.listdir(raw_file_path):
        if not file_name.endswith('.txt'):
            continue  # Skip non-.txt files

        file_path = os.path.join(raw_file_path, file_name)
        try: # Added try-except block for file operations
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")
            continue

        # Extract Python code block
        python_code_match = re.search(r'### Standard Code\s*```python\s*(.*?)\s*```', content, re.DOTALL)
        if python_code_match:
            python_code = python_code_match.group(1).strip()

            # Extract function name
            function_name_match = re.search(r'def\s+(\w+)\(', python_code)
            if function_name_match:
                function_name = function_name_match.group(1)
                # Check if function name already exists
                if function_name in function_names:
                    print(f"Warning: Function name '{function_name}' already exists. Skipping. File: {file_name}")
                    continue
                # Add function name to the set
                function_names.add(function_name)

                # Save Python code to a separate file
                code_file_path = os.path.join(code_path, f"{function_name}.txt")
                try:
                    with open(code_file_path, 'w', encoding='utf-8') as code_file:
                        code_file.write(python_code)
                except Exception as e:
                    print(f"Error writing code file for {function_name}: {e}")
                    continue # Skip to next file if writing fails

        # Extract YAML test cases
        yaml_match = re.search(r'### Test Cases\s*```yaml\s*(.*?)\s*```', content, re.DOTALL)
        if yaml_match:
            yaml_content = yaml_match.group(1).strip()
            all_yaml_content += yaml_content + "\n\n"  # Append YAML content to the string

        processed_file_count += 1

    # Save all YAML content to a file
    if all_yaml_content:
        try:
            with open(config_path, 'w', encoding='utf-8') as config_file:
                config_file.write(all_yaml_content.strip())  # Write combined YAML content
        except Exception as e:
            print(f"Error writing combined YAML file {config_path}: {e}")


    print(f"Processed {processed_file_count} files.")


def save_standard_tests(code_directory, instruction_directory, standard_code_directory):
    """
    Extracts content up to the second triple quote from code files to save as test instructions.
    Then moves the original file to the standard code directory.
    Assumes the function definition starts from the first line of the file.

    Args:
        code_directory (str): Directory path containing Python function files.
        instruction_directory (str): Output directory path for test instructions.
        standard_code_directory (str): Directory path for standard code.

    Returns:
        dict: A dictionary containing extraction information.
    """
    # Ensure output directories exist
    os.makedirs(instruction_directory, exist_ok=True)
    os.makedirs(standard_code_directory, exist_ok=True)

    # Get all code files
    try:
        files = os.listdir(code_directory)
    except FileNotFoundError:
        print(f"Error: Code directory {code_directory} not found.")
        return { # Return an error status
            'total_files': 0,
            'extracted_tests': 0,
            'errors': 1,
            'details': [{'error': f"Code directory {code_directory} not found."}]
        }


    # Dictionary to store extraction information
    extraction_info = {
        'total_files': len(files),
        'extracted_tests': 0,
        'errors': 0,
        'details': []
    }

    for file_name in files:
        if not file_name.endswith('.txt'):
            continue

        file_path = os.path.join(code_directory, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code_lines = f.readlines()

            # Count occurrences of triple quotes
            triple_quote_count = 0
            end_line_index = -1

            for i, line in enumerate(code_lines):
                # Count triple quotes in the current line
                quotes_in_line = line.count('"""')

                if quotes_in_line > 0:
                    triple_quote_count += quotes_in_line
                    # Check if the second triple quote (or more) is reached
                    if triple_quote_count >= 2:
                        end_line_index = i
                        break
            
            # If the second triple quote was found
            if end_line_index != -1:
                # Extract content from the first line to the line with the second triple quote
                test_content = ''.join(code_lines[0:end_line_index + 1])

                # Save to output file
                test_name = file_name.replace('.txt', '')
                output_file = os.path.join(instruction_directory, f"{test_name}_test_instruction.txt")
                with open(output_file, 'w', encoding='utf-8') as f_out:
                    f_out.write(test_content)

                extraction_info['extracted_tests'] += 1
                extraction_info['details'].append({
                    'file': file_name,
                    'output_file': output_file,
                    'lines_extracted': end_line_index + 1
                })
            else:
                # Second triple quote not found
                print(f"Second triple quote not found in file: {file_name}") and clarified
                extraction_info['errors'] += 1
                extraction_info['details'].append({
                    'file': file_name,
                    'error': "Second triple quote not found"
                })

            # Move original file to standard code directory
            new_path = os.path.join(standard_code_directory, file_name)
            shutil.move(file_path, new_path)

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
            extraction_info['errors'] += 1
            extraction_info['details'].append({
                'file': file_name,
                'error': str(e)
            })
    return extraction_info


def extract_and_match_functions(code_directory, yaml_path, output_yaml_path):
    """
    Extracts Python function names from a specified directory, finds corresponding
    test cases in a YAML configuration, and saves the matched configurations
    to a new YAML file.

    Parses YAML by text processing to preserve original formatting and tags.

    Args:
        code_directory (str): Directory path containing Python function files.
        yaml_path (str): Path to the original YAML configuration file.
        output_yaml_path (str): Path to the output (new) YAML file.

    Returns:
        bool: True if the operation was successful, False otherwise.
    """
    try:
        # Get all files in the code directory
        files = [f for f in os.listdir(code_directory) if os.path.isfile(os.path.join(code_directory, f))]

        # Extract function names from files
        function_names = set()
        function_name_pattern = r"def\s+(\w+)\("

        for file_name in files:
            file_path = os.path.join(code_directory, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code_content = f.read()
                    matches = re.findall(function_name_pattern, code_content)
                    function_names.update(matches)
            except Exception as e:
                print(f"Error reading file {file_name}: {e}")

        print(f"Function names extracted from code files: {function_names}")

        # Read all lines from the YAML file
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_lines = f.readlines()
        except FileNotFoundError:
            print(f"Error: YAML file {yaml_path} not found.")
            return False


        # Parse the YAML file, splitting it into function blocks
        function_blocks = {}
        current_block_lines = []
        current_function_name = None

        for line in yaml_lines:
            # Check if it's a new top-level entry (function definition)
            # A top-level entry is a line that doesn't start with a space or hyphen and contains a colon.
            stripped_line = line.strip()
            if stripped_line and not line.startswith(' ') and not line.startswith('-') and ':' in line:
                # If there's an existing current function, save its block
                if current_function_name is not None and current_block_lines:
                    function_blocks[current_function_name] = current_block_lines

                # Extract the new function name (part before the colon)
                current_function_name = stripped_line.split(':', 1)[0].strip()
                current_block_lines = [line]  # Start a new block with the current line
            elif current_function_name is not None:
                # Add the line to the current block
                current_block_lines.append(line)

        # Save the last function block
        if current_function_name is not None and current_block_lines:
            function_blocks[current_function_name] = current_block_lines

        # Create new YAML content, including only matched functions
        matched_blocks_content = []
        for func_name_key, block_lines in function_blocks.items():
            if func_name_key in function_names:
                matched_blocks_content.extend(block_lines)

        # Save matched blocks to the output file
        try:
            with open(output_yaml_path, 'w', encoding='utf-8') as f_out:
                f_out.writelines(matched_blocks_content)
        except Exception as e:
            print(f"Error writing output YAML file {output_yaml_path}: {e}")
            return False


        matched_count = sum(1 for func_key in function_blocks if func_key in function_names)
        print(f"Successfully matched configurations for {matched_count} functions. Saved to {output_yaml_path}")
        return True

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
        return False



if __name__== "__main__":

    # Generate raw data for atomic requirements
    operators_file_path = "./data/GEE_API_Atomic.xlsx"
    raw_output_path = "./test_code/autogen_atomic_raw"
    model_to_use = "qwen-max-2025-01-25"
    model_provider_name = "aliyun"
    generate_gee_atomic_test_code(operators_file_path, raw_output_path, model_provider=model_provider_name, model_name=model_to_use)

    # Process raw data, extract Python code and YAML test cases
    raw_files_dir = "./test_code/autogen_atomic_raw"
    extracted_code_dir = "./test_code/atomic_code"
    yaml_config_output_path = "./test_code/atomic_code/atomic_test_config.yaml"
    process_raw_file(raw_files_dir, extracted_code_dir, yaml_config_output_path)

    # Extract test instructions
    source_code_dir = "./test_code/atomic_code"
    test_instructions_dir = "./test_code/atomic_code/test_instructions"
    final_standard_code_dir = "./test_code/atomic_code/standard_code"
    save_standard_tests(source_code_dir, test_instructions_dir, final_standard_code_dir)

    # Extract function names and match YAML configurations
    std_code_path_for_yaml_match = "./test_code/atomic_code/standard_code"
    input_yaml_file_path = "./test_code/atomic_code/atomic_test_config.yaml"
    matched_yaml_output_path = "./test_code/atomic_code/matched_atomic_test_config.yaml"
    extract_and_match_functions(std_code_path_for_yaml_match, input_yaml_file_path, matched_yaml_output_path)

    print("Processing finished.")


