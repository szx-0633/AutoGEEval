import os
import shutil
import re
import yaml
import time
from typing import Optional, List, Dict, Tuple
import concurrent.futures
from tqdm import tqdm
from call_language_model import call_language_model


def extract_code_from_response(response_text: str) -> str:
    """
    Extracts Python code from the model's response.

    Args:
    response_text (str): The full response text from the model.

    Returns:
    str: The extracted Python code, or the original response if no code block is found.
    """
    # Try to match code blocks enclosed in triple backticks
    code_pattern = r"```python?(.*?)```"
    matches = re.findall(code_pattern, response_text, re.DOTALL)

    if matches:
        # Return the first matched code block, stripping leading/trailing whitespace
        return matches[0].strip()
    else:
        # If no code block is found, return the original response
        return response_text.strip()


def create_prompt_for_function_completion(test_file_content: str) -> str:
    """
    Creates a prompt suitable for function completion based on the test file content.

    Args:
    test_file_content (str): The content of the test file.

    Returns:
    str: The formatted prompt.
    """
    prompt = (
        "Please complete the following GEE Python API function. "
        "Return ONLY the complete function code without any explanations or additional text. "
        "Do not add any comments beyond what's already in the docstring. "
        "Here's the function:\n\n"
        f"{test_file_content}"
    )
    return prompt


def process_test_file(
        test_file_path: str,
        output_dir: str,
        model_provider: str,
        model_name: str,
        stream: bool,
        system_prompt: str,
        temperature: Optional[float] = 0.2,
        max_tokens: Optional[int] = 2048,
        config_path: str = './llm_config.yaml'
) -> Dict:
    """
    Processes a single test file, calls the model, and saves the result.

    Args:
    test_file_path (str): Path to the test file.
    output_dir (str): Output directory.
    model_provider (str): Model provider.
    model_name (str): Model name.
    stream (bool): Whether to use streaming.
    system_prompt (str): System prompt.
    temperature (float, optional): Temperature parameter.
    max_tokens (int, optional): Maximum number of tokens to generate.
    config_path (str): Path to the configuration file.

    Returns:
    Dict: Information about the processing result.
    """
    # Read the content of the test file
    with open(test_file_path, 'r', encoding='utf-8') as f:
        test_content = f.read()

    # Get the file name (without path and extension)
    file_name = os.path.basename(test_file_path)
    base_name = os.path.splitext(file_name)[0]

    output_file = os.path.join(output_dir, f"{base_name}_response.txt")

    if os.path.exists(output_file):
        print(f"File {output_file} already exists, skipping processing...")
        return {'test_file': test_file_path, 'error': None}

    # Create the prompt
    user_prompt = create_prompt_for_function_completion(test_content)

    # Call the model
    start_time = time.time()
    response_text, tokens_used, error_msg = call_language_model(
        model_provider=model_provider,
        model_name=model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        stream=stream,
        collect=True,
        temperature=temperature,
        max_tokens=max_tokens,
        files=None,
        config_path=config_path
    )
    elapsed_time = time.time() - start_time

    # Extract code
    code = extract_code_from_response(response_text)
    code = response_text

    # Save the result

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(code)

    # Return processing information
    result = {
        'test_file': test_file_path,
        'tokens_used': tokens_used,
        'elapsed_time': elapsed_time,
        'error': error_msg if error_msg else None
    }

    time.sleep(1)

    return result


def run_function_completion_tests(
        test_dir: str,
        models_config: List[Dict],
        type: str,
        stream: bool = False,
        system_prompt: str = "",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        config_path: str = './llm_config.yaml',
        parallel: bool = True,
        max_workers: int = 4,
        times: int = 5,
) -> Dict:
    """
    Runs function completion tests for all test files in the specified directory.

    Args:
    test_dir (str): Directory of test files.
    models_config (List[Dict]): List of model configurations, each containing provider and name.
    type (str): Type of requirement.
    system_prompt (str): System prompt.
    stream (bool): Whether to use streaming.
    temperature (float): Temperature parameter.
    max_tokens (int): Maximum number of tokens to generate.
    config_path (str): Path to the configuration file.
    parallel (bool): Whether to process in parallel.
    max_workers (int): Maximum number of parallel worker threads.
    times (int): Number of times to repeat the test.

    Returns:
    Dict: Test result statistics.
    """
    # Get all test files
    test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir)
                  if os.path.isfile(os.path.join(test_dir, f)) and f.endswith('.txt')]
    test_file_count = len(test_files)

    results = {}

    for time_run in range(1, times + 1):
        for model_config in models_config:
            model_provider = model_config['provider']
            model_name = model_config['name']
            model_name_simple = model_config['name_simple']

            print(f"\nProcessing model: {model_provider}/{model_name_simple}")

            # Create output directory
            output_dir = f"./generate_results/{model_name_simple}_{time_run}/{type}"
            os.makedirs(output_dir, exist_ok=True)
            output_file_count = len(os.listdir(output_dir))

            if output_file_count >= test_file_count:
                print(f"Model {model_provider}/{model_name_simple} has processed all files, skipping...")
                continue

            model_results = []

            if parallel:
                # Parallel processing
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_file = {
                        executor.submit(
                            process_test_file,
                            test_file,
                            output_dir,
                            model_provider,
                            model_name,
                            stream,
                            system_prompt,
                            temperature,
                            max_tokens,
                            config_path
                        ): test_file for test_file in test_files
                    }

                    for future in tqdm(concurrent.futures.as_completed(future_to_file),
                                       total=len(test_files),
                                       desc=f"Processing {model_name}"):
                        try:
                            result = future.result()
                            model_results.append(result)
                        except Exception as e:
                            test_file = future_to_file[future]
                            print(f"Error processing file {test_file}: {e}")
                            model_results.append({
                                'test_file': test_file,
                                'error': str(e)
                            })
            else:
                # Serial processing
                for test_file in tqdm(test_files, desc=f"Processing {model_name}"):
                    try:
                        result = process_test_file(
                            test_file,
                            output_dir,
                            model_provider,
                            model_name,
                            stream,
                            system_prompt,
                            temperature,
                            max_tokens,
                            config_path
                        )
                        model_results.append(result)
                    except Exception as e:
                        print(f"Error processing file {test_file}: {e}")
                        model_results.append({
                            'test_file': test_file,
                            'error': str(e)
                        })

            # Calculate statistics
            total_files = len(model_results)
            successful = len([r for r in model_results if r.get('error') is None])
            total_tokens = sum(r.get('tokens_used', 0) for r in model_results if r.get('tokens_used') is not None)
            avg_time = sum(r.get('elapsed_time', 0) for r in model_results if r.get('elapsed_time') is not None) / max(
                successful, 1)

            # Save result summary
            summary = {
                'model_provider': model_provider,
                'model_name': model_name,
                'total_files': total_files,
                'successful': successful,
                'failed': total_files - successful,
                'total_tokens_used': total_tokens,
                'average_time_per_file': avg_time,
                'detailed_results': model_results
            }

            # Save summary to file
            summary_file = os.path.join(output_dir, "summary.yaml")
            with open(summary_file, 'w', encoding='utf-8') as f:
                yaml.dump(summary, f, default_flow_style=False)

            results[f"{model_provider}/{model_name}"] = summary

            print(f"Finished processing {model_provider}/{model_name_simple} for the {time_run}st/nd/rd/th time:")
            print(f"  Total files: {total_files}")
            print(f"  Successful: {successful}, Failed: {total_files - successful}")
            print(f"  Total tokens used: {total_tokens}")
            print(f"  Average processing time: {avg_time:.2f} seconds")

    return results


def clean_file(model_name: str):
    """Cleans the generated files for a given model."""
    output_dir1 = f"./generate_results2/{model_name}/atomic/"

    # Check if the directory exists
    if not os.path.exists(output_dir1):
        print(f"Directory {output_dir1} does not exist, skipping cleaning for model {model_name}.")
        return

    for file in os.listdir(output_dir1):
        if file.endswith(".txt"):
            file_path = os.path.join(output_dir1, file) # Used os.path.join for robust path construction
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()

                # Process file content
                cleaned_content = clean_content(file_content)

                # Write back to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)

                # print(f"Cleaned file: {file}")
            except Exception as e:
                print(f"Error processing file {file}: {e}")


def clean_content(content: str) -> str:
    """Cleans the provided content by removing unnecessary parts like think blocks and main function calls."""
    lines = content.split('\n')
    cleaned_lines = []

    # Flag to indicate if currently inside a code block
    in_code_block = False
    # Flag to indicate if currently inside a think block
    in_think_block = False
    # Flag to indicate if the previous line was the end of a think block
    prev_was_think_end = False
    # Flag to indicate if currently inside an unnecessary main function
    in_main_function = False

    for line in lines:
        # Handle code block start marker
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            # Skip the code block marker line
            continue

        # Handle think block start
        if '<think>' in line:
            in_think_block = True
            continue

        # Handle think block end
        if '</think>' in line:
            in_think_block = False
            prev_was_think_end = True
            continue

        # Skip empty lines after a think block end
        if prev_was_think_end and line.strip() == '':
            prev_was_think_end = False
            continue
        else:
            prev_was_think_end = False

        # Identify and handle main function
        if 'if __name__ == "__main__":' in line:
            in_main_function = True
            continue

        # If not in a think block and not a code block marker, keep the line
        if not in_think_block and not in_main_function:
            cleaned_lines.append(line)

    # Further remove explanatory text
    cleaned_lines1 = []
    function_flag = False
    for line in cleaned_lines:
        if line.strip().startswith('def'):
            function_flag = True
        if function_flag:
            cleaned_lines1.append(line)

    return '\n'.join(cleaned_lines1)


def calculate_code_lines(model, task_type):
    """Calculates the average number of code lines for a given model and task type."""
    code_dir_1 = f"./generate_results2/{model}/{task_type}/"
    
    # Check if the directory exists
    if not os.path.exists(code_dir_1):
        print(f"Directory {code_dir_1} does not exist for model {model}, task type {task_type}. Returning 0 lines.")
        return 0
        
    files1 = os.listdir(code_dir_1)
    total_code_lines = 0
    file_count = 0
    for file in files1:
        if not file.endswith(".txt"):
            continue
        file_count += 1
        with open(os.path.join(code_dir_1, file), 'r', encoding='utf-8') as f: # Used os.path.join
            lines = f.readlines()
            code_lines = len(lines)
            total_code_lines += code_lines
    return total_code_lines / file_count if file_count > 0 else 0


if __name__ == "__main__":
    # System prompt
    system_prompt = (
        "You are an expert in geospatial analysis with python and Google Earth Engine(GEE)."
    )

    # Model configurations to test
    models_to_test = [
        {'provider': 'ollama', 'name': 'qwen2.5:3b', 'name_simple': 'qwen2.5_3b'},
        {'provider': 'ollama', 'name': 'qwen2.5:7b', 'name_simple': 'qwen2.5_7b'},
        {'provider': 'ollama', 'name': 'qwen2.5-coder:3b', 'name_simple': 'qwen2.5_coder_3b'},
        {'provider': 'ollama', 'name': 'qwen2.5-coder:7b', 'name_simple': 'qwen2.5_coder_7b'},
        {'provider': 'ollama', 'name': 'codellama:7b', 'name_simple': 'codellama_7b'},
        {'provider': 'aliyun', 'name': 'qwen2.5-32b-instruct', 'name_simple': 'qwen2.5_32b'},
        {'provider': 'aliyun', 'name': 'qwen2.5-coder-32b-instruct', 'name_simple': 'qwen2.5_coder_32b'},
        {'provider': 'aliyun', 'name': 'qwq-32b', 'name_simple': 'qwq_32b'},
        {'provider': 'deepseek', 'name': 'deepseek-chat', 'name_simple': 'deepseek_v3_250324'},
        {'provider': 'deepseek', 'name': 'deepseek-reasoner', 'name_simple': 'deepseek_r1'},
        {'provider': 'volcengine', 'name': 'deepseek-v3-241226', 'name_simple': 'deepseek_v3_241226'},
        {'provider': 'openai', 'name': 'gpt-4o-mini', 'name_simple': 'gpt_4o_mini'},
        {'provider': 'openai', 'name': 'gpt-4o', 'name_simple': 'gpt_4o'},
        {'provider': 'openai', 'name': 'o3-mini', 'name_simple': 'o3_mini'},
        {'provider': 'google', 'name': 'claude-3-7-sonnet-20250219', 'name_simple': 'claude_3.7_sonnet'},
        {'provider': 'anthropic', 'name': 'gemini-2.0-pro-exp-02-05', 'name_simple': 'gemini_2.0_pro'},
        {'provider': 'ollama', 'name': 'deepseek-coder-v2-16b', 'name_simple': 'deepseek_coder_v2_16b'},
        {'provider': 'ollama', 'name': 'geocode-gpt:latest', 'name_simple': 'geocode_gpt'},
    ]
    
    models = ["deepseek_r1", "codellama_7b", "qwen2.5_3b", "qwen2.5_7b", "qwen2.5_coder_3b", "qwen2.5_coder_7b", "qwen2.5_32b", "claude_3.7_sonnet",
              "qwen2.5_coder_32b", "qwq_32b", "deepseek_v3_241226", "deepseek_v3_250324", "gpt_4o_mini", "gpt_4o", "o3_mini", "gemini_2.0_pro",
              "deepseek_coder_v2_16b", "geocode_gpt"]

    # Run tests
    results1 = run_function_completion_tests(
        test_dir="dataset_complete/atomic_code/test_instructions",
        models_config=models_to_test,
        type="atomic",
        stream=False,
        system_prompt=system_prompt,
        temperature=0.2,
        max_tokens=4096,
        config_path='./llm_config.yaml',
        max_workers=8,
        parallel=True,
        times=5
    )
    
    # Output summary
    print("\nGeneration complete!")
    for model, summary in results1.items():
        print(f"\n{model}:")
        print(
            f"  Success rate: {summary['successful']}/{summary['total_files']} ({summary['successful'] / summary['total_files'] * 100:.1f}%)")
        print(f"  Total tokens used: {summary['total_tokens_used']}")
        print(f"  Average processing time: {summary['average_time_per_file']:.2f} seconds")

    # Clean generated code files
    models_to_clean = [
                "gpt_4o_1", "gpt_4o_2", "gpt_4o_3", "gpt_4o_4", "gpt_4o_5",
                "gpt_4o_mini_1", "gpt_4o_mini_2", "gpt_4o_mini_3", "gpt_4o_mini_4", "gpt_4o_mini_5",
                "o3_mini_1", "o3_mini_2", "o3_mini_3", "o3_mini_4", "o3_mini_5",
                "qwen2.5_3b_1", "qwen2.5_3b_2", "qwen2.5_3b_3", "qwen2.5_3b_4", "qwen2.5_3b_5",
                "qwen2.5_7b_1", "qwen2.5_7b_2", "qwen2.5_7b_3", "qwen2.5_7b_4", "qwen2.5_7b_5",
                "qwen2.5_coder_3b_1", "qwen2.5_coder_3b_2", "qwen2.5_coder_3b_3", "qwen2.5_coder_3b_4", "qwen2.5_coder_3b_5",
                "qwen2.5_coder_7b_1", "qwen2.5_coder_7b_2", "qwen2.5_coder_7b_3", "qwen2.5_coder_7b_4", "qwen2.5_coder_7b_5",
                "qwen2.5_32b_1", "qwen2.5_32b_2", "qwen2.5_32b_3", "qwen2.5_32b_4", "qwen2.5_32b_5",
                "qwen2.5_coder_32b_1", "qwen2.5_coder_32b_2", "qwen2.5_coder_32b_3", "qwen2.5_coder_32b_4", "qwen2.5_coder_32b_5",
                "codellama_7b_1", "codellama_7b_2", "codellama_7b_3", "codellama_7b_4", "codellama_7b_5",
                "claude_3_7_sonnet_1", "claude_3_7_sonnet_2", "claude_3_7_sonnet_3", "claude_3_7_sonnet_4", "claude_3_7_sonnet_5",
                "deepseek_coder_v2_16b_1", "deepseek_coder_v2_16b_2", "deepseek_coder_v2_16b_3", "deepseek_coder_v2_16b_4", "deepseek_coder_v2_16b_5",
                "deepseek_v3_241226_1", "deepseek_v3_241226_2", "deepseek_v3_241226_3", "deepseek_v3_241226_4", "deepseek_v3_241226_5",
                "deepseek_v3_250324_1", "deepseek_v3_250324_2", "deepseek_v3_250324_3", "deepseek_v3_250324_4", "deepseek_v3_250324_5",
                "qwq_32b_1", "qwq_32b_2", "qwq_32b_3", "qwq_32b_4", "qwq_32b_5",
                "geocode_gpt_1", "geocode_gpt_2", "geocode_gpt_3", "geocode_gpt_4", "geocode_gpt_5",
                "gemini_2.0_pro_1", "gemini_2.0_pro_2", "gemini_2.0_pro_3", "gemini_2.0_pro_4", "gemini_2.0_pro_5",
                "deepseek_r1_250120_1", "deepseek_r1_250120_2", "deepseek_r1_250120_3", "deepseek_r1_250120_4", "deepseek_r1_250120_5",
            ]
    for model_to_clean in models_to_clean:
        clean_file(model_to_clean)
        print("Cleaned all files for model: ", model_to_clean)
    for model_to_calculate_lines in models_to_clean:
        num_lines = calculate_code_lines(model_to_calculate_lines, "atomic")
        print(f"{model_to_calculate_lines},{num_lines}")

    print(1)
