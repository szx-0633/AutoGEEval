import yaml
import os
from collections import Counter
import pandas as pd


def process_test_report(yaml_path, processed_yaml_path):
    """
    Processes a YAML test report to classify errors and calculate statistics.

    Args:
        yaml_path (str): Path to the input YAML test report file.
        processed_yaml_path (str): Path to save the processed YAML report with error classifications.

    Returns:
        dict: A dictionary containing error statistics.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(processed_yaml_path), exist_ok=True)

    # Read the YAML file
    try:
        with open(yaml_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Input YAML file not found at {yaml_path}")
        return {
            'total_test_cases': 0,
            'total_errors': 0,
            'error_types': {},
            'error_percentages': {}
        }
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {yaml_path}: {e}")
        return {
            'total_test_cases': 0,
            'total_errors': 0,
            'error_types': {},
            'error_percentages': {}
        }


    # Counter for error types
    error_types_counter = Counter()
    total_test_cases = 0
    processed_tasks = {} # Store processed tasks to avoid modifying data directly during iteration

    # Process each task
    for task_name, task_info in data.items():
        processed_task_info = task_info.copy() # Work on a copy
        processed_task_info['test_cases'] = []
        # Check for file_errors (likely syntax errors before test execution)
        if task_info.get('file_errors', []):
            for test_case_data in task_info.get('test_cases', []):
                test_case = test_case_data.copy()
                if test_case.get('status') == 'failed': # Check if status exists
                    test_case['error_type'] = 'syntax error'
                    error_types_counter['syntax error'] += 1
                else:
                    test_case['error_type'] = None
                processed_task_info['test_cases'].append(test_case)
        else:
            # Process each test case if no file_errors
            for test_case_data in task_info.get('test_cases', []):
                test_case = test_case_data.copy()
                total_test_cases += 1

                if test_case.get('status') == 'passed': # Check if status exists
                    test_case['error_type'] = None
                    processed_task_info['test_cases'].append(test_case)
                    continue

                error_msg = test_case.get('error', '').lower()
                error_type = None # Initialize error_type

                # Determine error type based on error message keywords
                if 'error executing code' in error_msg:
                    error_type = 'syntax error'
                elif 'error in test case' in error_msg:
                    error_type = 'operator or dataset attribute or parameter error'
                elif 'error getting download url' in error_msg or 'error when checking' in error_msg:
                    error_type = 'invalid or wrong answer'
                else:
                    error_type = 'other error'

                test_case['error_type'] = error_type
                if error_type:  # Count only non-None error types
                    error_types_counter[error_type] += 1
                processed_task_info['test_cases'].append(test_case)
        processed_tasks[task_name] = processed_task_info

    # Calculate total errors and percentages
    total_errors = sum(error_types_counter.values())
    error_percentages = {error_type: (count / total_errors * 100 if total_errors > 0 else 0)
                         for error_type, count in error_types_counter.items()}

    # Add error statistics summary
    summary = {
        'total_test_cases': total_test_cases,
        'total_errors': total_errors,
        'error_types': dict(error_types_counter),
        'error_percentages': error_percentages
    }

    # Prepare data for the new YAML file
    result_data = {
        'tasks': processed_tasks, # Use the processed tasks
        'summary': summary
    }

    try:
        with open(processed_yaml_path, 'w', encoding='utf-8') as file:
            yaml.dump(result_data, file, default_flow_style=False, sort_keys=False) # Added sort_keys=False
    except IOError as e:
        print(f"Error writing processed YAML file {processed_yaml_path}: {e}")

    # Print statistical results
    print(f"Processing complete for: {yaml_path}")
    print(f"Total test cases: {total_test_cases}")
    print(f"Total errors: {total_errors}")
    print("Error type statistics:")
    for error_type_key, count_val in error_types_counter.items():
        print(f"  {error_type_key}: {count_val} ({error_percentages[error_type_key]:.2f}%)")

    # Return statistics for report generation
    return {
        'total_test_cases': total_test_cases,
        'total_errors': total_errors,
        'error_types': dict(error_types_counter),
        'error_percentages': error_percentages
    }


def process_models():
    """
    Processes test reports for a predefined list of models and task types.
    Aggregates statistics and generates summary reports.
    """
    models_to_process = [
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

    task_types_to_process = ["atomic"]

    # To store statistics for all models
    all_models_stats = {}
    # To store statistics per task type
    stats_per_task_type = {task_type: {} for task_type in task_types_to_process}

    for model_name in models_to_process:
        print(f"Processing model: {model_name}")
        current_model_stats = {'model': model_name}
        current_model_error_types = Counter()
        current_model_total_errors = 0

        for task_category in task_types_to_process:
            input_yaml_path = f"./generate_results/{model_name}/{task_category}_test_reports.yaml"
            output_yaml_path = f"./generate_results/{model_name}/{task_category}_processed_test_reports.yaml"

            try:
                report_stats = process_test_report(input_yaml_path, output_yaml_path)

                # Add statistics for each task type to the current model's stats
                current_model_stats[f'{task_category}_total_errors'] = report_stats['total_errors']

                # Add counts and percentages for various error types
                for error_category, count_value in report_stats['error_types'].items():
                    current_model_stats[f'{task_category}_{error_category}_count'] = count_value
                    current_model_stats[f'{task_category}_{error_category}_percent'] = report_stats['error_percentages'].get(error_category, 0)

                # Accumulate error types for the current model
                for error_category, count_value in report_stats['error_types'].items():
                    current_model_error_types[error_category] += count_value

                current_model_total_errors += report_stats['total_errors']

                # Store statistics for the task type
                stats_per_task_type[task_category][model_name] = {
                    'model': model_name,
                    'total_errors': report_stats['total_errors'],
                    'error_types': report_stats['error_types'],
                    'error_percentages': report_stats['error_percentages']
                }

            except Exception as e:
                print(f"Error processing task {task_category} for model {model_name}: {e}") and clarified
                current_model_stats[f'{task_category}_error'] = str(e)

        # Calculate overall error type percentages for the current model
        current_model_error_percentages = {err_type: (err_count / current_model_total_errors * 100 if current_model_total_errors > 0 else 0)
                                           for err_type, err_count in current_model_error_types.items()}

        current_model_stats['total_errors'] = current_model_total_errors
        current_model_stats['error_types'] = dict(current_model_error_types)
        current_model_stats['error_percentages'] = current_model_error_percentages

        all_models_stats[model_name] = current_model_stats

    # Generate statistical reports
    generate_reports(all_models_stats, stats_per_task_type, task_types_to_process)


def generate_reports(all_stats_data, task_type_specific_stats, task_categories):
    """Generates statistical reports and saves them as CSV files.""" and improved docstring
    # 1. Generate overall error statistics report
    overall_stats_list = []
    for model_name_key, model_data in all_stats_data.items():
        stat_entry = {
            'model': model_name_key,
            'total_errors': model_data['total_errors'],
        }
        for error_type_key, count_val in model_data['error_types'].items():
            stat_entry[f"{error_type_key}_count"] = count_val
        for error_type_key, percent_val in model_data['error_percentages'].items():
            stat_entry[f"{error_type_key}_percent"] = percent_val
        overall_stats_list.append(stat_entry)
    
    overall_df = pd.DataFrame(overall_stats_list)

    # Sort by total errors in descending order
    if not overall_df.empty and 'total_errors' in overall_df.columns:
        overall_df = overall_df.sort_values('total_errors', ascending=False)

    # Save the overall statistics report
    overall_report_file_path = "./generate_results/error_report_overall.csv"
    try:
        overall_df.to_csv(overall_report_file_path, index=False)
        print(f"Overall error statistics report saved to: {overall_report_file_path}")
    except IOError as e:
        print(f"Error saving overall report to {overall_report_file_path}: {e}")

    # 2. Generate error statistics report for each task type
    for task_cat in task_categories:
        task_specific_stats_list = []
        # Collect all possible error types for consistent column order
        all_error_keys_for_task = set()
        if task_cat in task_type_specific_stats:
            for model_data_val in task_type_specific_stats[task_cat].values():
                all_error_keys_for_task.update(model_data_val['error_types'].keys())
        
        sorted_error_keys = sorted(list(all_error_keys_for_task))

        if task_cat in task_type_specific_stats:
            for model_name_key, model_data_val in task_type_specific_stats[task_cat].items():
                stat_entry = {
                    'model': model_name_key,
                    'total_errors': model_data_val['total_errors'],
                }
                for error_key in sorted_error_keys: # Use sorted keys
                    stat_entry[f"{error_key}_count"] = model_data_val['error_types'].get(error_key, 0)
                    stat_entry[f"{error_key}_percent"] = model_data_val['error_percentages'].get(error_key, 0)
                task_specific_stats_list.append(stat_entry)

        task_df = pd.DataFrame(task_specific_stats_list)

        # Sort by total errors in descending order
        if not task_df.empty and 'total_errors' in task_df.columns:
            task_df = task_df.sort_values('total_errors', ascending=False)

        # Save the task type statistics report
        task_report_file_path = f"./generate_results/error_report_{task_cat}.csv"
        try:
            task_df.to_csv(task_report_file_path, index=False)
            print(f"Error statistics report for task {task_cat} saved to: {task_report_file_path}") and clarified
        except IOError as e:
            print(f"Error saving report for task {task_cat} to {task_report_file_path}: {e}")


# If this script is run directly
if __name__ == "__main__":
    process_models()
