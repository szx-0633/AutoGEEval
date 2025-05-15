import os
import json
import yaml
from pathlib import Path
from collections import defaultdict
import pandas as pd


def process_test_reports(reports_dir, output_yaml_path, output_stats_path, configs):
    """
    Processes test reports from a directory, merges them into a YAML file,
    and generates statistical information.

    Args:
        reports_dir (str): Path to the directory containing test reports.
        output_yaml_path (str): Path for the output merged YAML file.
        output_stats_path (str): Path for the output statistics JSON file.
        configs (dict): Original test case configurations.
    """
    # Dictionary to store all test reports, keyed by function_name
    reports_by_function = {}

    # Statistics
    stats = {
        "total_tasks": 0,
        "total_test_cases": 0,
        "passed_tasks": 0,
        "passed_test_cases": 0
    }

    if not os.path.isdir(reports_dir):
        print(f"Error: Reports directory not found: {reports_dir}")
        return stats # Return empty/initial stats

    # Iterate over all JSON files in the directory
    for file_path in Path(reports_dir).glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            function_name = report["function_name"]
            file_errors = report.get("file_errors", [])

            if len(file_errors) > 0:
                # If file_errors exist, load test cases for this task from the configuration and mark all as failed
                test_cases_config = configs.get(function_name, [])

                # Create a list of failed test cases
                failed_test_cases = []
                for i, _ in enumerate(test_cases_config):
                    failed_test_case = {
                        "error": file_errors[0],  # Use the first error from file_errors
                        "retry_count": 0,
                        "status": "failed",
                        "test_case_id": i + 1
                    }
                    failed_test_cases.append(failed_test_case)

                # Update test case information in the report
                report["test_cases"] = failed_test_cases
                report["total_test_cases"] = len(failed_test_cases)
                report["failed_test_cases"] = len(failed_test_cases)
                report["passed_test_cases"] = 0
                report["skipped_test_cases"] = 0
                report["status"] = "failed"

                # Update statistics
                stats["total_tasks"] += 1
                stats["total_test_cases"] += len(failed_test_cases)

            else:
                # Update statistics
                stats["total_tasks"] += 1
                stats["total_test_cases"] += report["total_test_cases"]
                if report["status"] == "passed":
                    stats["passed_tasks"] += 1
                stats["passed_test_cases"] += report["passed_test_cases"]

            # Add the report to the dictionary, keyed by function_name
            reports_by_function[function_name] = report

        except FileNotFoundError:
            print(f"Error: Report file not found: {file_path}")
        except json.JSONDecodeError:
            print(f"Could not parse JSON file: {file_path}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {file_path}: {e}")

    # Calculate pass rates
    stats["task_pass_rate"] = stats["passed_tasks"] / stats["total_tasks"] if stats["total_tasks"] > 0 else 0
    stats["test_case_pass_rate"] = stats["passed_test_cases"] / stats["total_test_cases"] if stats[
                                                                                                 "total_test_cases"] > 0 else 0

    # Format pass rates as percentages
    stats["task_pass_rate_percentage"] = f"{stats['task_pass_rate']:.2%}"
    stats["test_case_pass_rate_percentage"] = f"{stats['test_case_pass_rate']:.2%}"

    try:
        # Ensure output directories exist
        os.makedirs(os.path.dirname(output_yaml_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_stats_path), exist_ok=True)

        # Save the report dictionary to a YAML file
        with open(output_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(reports_by_function, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        # Save statistics to a JSON file
        with open(output_stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"Merged {stats['total_tasks']} test reports into {output_yaml_path}")
        print(f"Statistics saved to {output_stats_path}")
    except IOError as e:
        print(f"Error writing output files: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during output: {e}")

    return stats


def process_test_reports_only_first(reports_dir, output_yaml_path, output_stats_path, configs):
    """
    Processes test reports, keeping only the first test case for each, 
    merges them into a YAML file, and generates statistics.

    Args:
        reports_dir (str): Path to the directory containing test reports.
        output_yaml_path (str): Path for the output merged YAML file.
        output_stats_path (str): Path for the output statistics JSON file.
        configs (dict): Original test case configurations.
    """
    # Dictionary to store all test reports, keyed by function_name
    reports_by_function = {}

    # Statistics
    stats = {
        "total_tasks": 0,
        "total_test_cases": 0, # Will always be 1 per task in this function's logic
        "passed_tasks": 0,
        "passed_test_cases": 0
    }

    if not os.path.isdir(reports_dir):
        print(f"Error: Reports directory not found: {reports_dir}")
        return stats

    # Iterate over all JSON files in the directory
    for file_path in Path(reports_dir).glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            function_name = report["function_name"]
            file_errors = report.get("file_errors", [])

            if len(file_errors) > 0:
                # If file_errors exist, load test cases for this task from the configuration and mark all as failed
                test_cases_config = configs.get(function_name, [])

                # Create a list of failed test cases (only the first one matters here)
                failed_test_cases = []
                if test_cases_config: # Ensure there's at least one config to create a failed case from
                    failed_test_case = {
                        "error": file_errors[0],  # Use the first error from file_errors
                        "retry_count": 0,
                        "status": "failed",
                        "test_case_id": 1 # Since we only care about the first
                    }
                    failed_test_cases.append(failed_test_case)
                
                # Update report's test cases to reflect this single failed case
                report["test_cases"] = failed_test_cases 
            else:
                # Keep only the first test case if no file_errors
                if report.get("test_cases"):
                    report["test_cases"] = report["test_cases"][:1]
                else:
                    report["test_cases"] = []


            stats["total_tasks"] += 1
            
            if not report["test_cases"]: # If, after filtering, no test cases remain (e.g. original was empty or file_error with no config)
                # print(f"Report for {function_name} has no test cases after processing.")
                report["status"] = "failed" # Consider it a failed task
                report["passed_test_cases"] = 0
                report["failed_test_cases"] = 1 # Assume one conceptual test case for the task
                report["skipped_test_cases"] = 0
                report["total_test_cases"] = 1
                stats["total_test_cases"] += 1 # Count this conceptual test case
                reports_by_function[function_name] = report
                continue

            # Now, total_test_cases for this report is 1
            stats["total_test_cases"] += 1 
            current_test_case = report["test_cases"][0]

            if current_test_case["status"] == "passed":
                report["status"] = "passed"
                report["passed_test_cases"] = 1
                stats["passed_tasks"] += 1
                stats["passed_test_cases"] += 1
            else:
                report["status"] = "failed"
                report["passed_test_cases"] = 0
            
            report["failed_test_cases"] = 1 - report["passed_test_cases"]
            report["skipped_test_cases"] = 0
            report["total_test_cases"] = 1 # Explicitly set for the report

            # Add the report to the dictionary, keyed by function_name
            reports_by_function[function_name] = report

        except FileNotFoundError:
            print(f"Error: Report file not found: {file_path}")
        except json.JSONDecodeError:
            print(f"Could not parse JSON file: {file_path}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {file_path}: {e}")
    
    # Calculate pass rates
    stats["task_pass_rate"] = stats["passed_tasks"] / stats["total_tasks"] if stats["total_tasks"] > 0 else 0
    stats["test_case_pass_rate"] = stats["passed_test_cases"] / stats["total_test_cases"] if stats[
                                                                                                 "total_test_cases"] > 0 else 0

    # Format pass rates as percentages
    stats["task_pass_rate_percentage"] = f"{stats['task_pass_rate']:.2%}"
    stats["test_case_pass_rate_percentage"] = f"{stats['test_case_pass_rate']:.2%}"
    
    try:
        # Ensure output directories exist
        os.makedirs(os.path.dirname(output_yaml_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_stats_path), exist_ok=True)

        # Save the report dictionary to a YAML file
        with open(output_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(reports_by_function, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        # Save statistics to a JSON file
        with open(output_stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"Merged {stats['total_tasks']} test reports (first case only) into {output_yaml_path}")
        print(f"Statistics (first case only) saved to {output_stats_path}")
    except IOError as e:
        print(f"Error writing output files: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during output: {e}")

    return stats


def python_constructor(loader, node):
    """
    Custom constructor to parse Python code in !python tags.
    In this context, it's a no-op, returning None.
    """
    return None  # If no function, return None


def process_models(models, task_types):
    """
    Processes test reports for different models and task types.

    Args:
        models (list): A list of model names.
        task_types (list): A list of task types (e.g., "atomic", "combined").
    """
    yaml.add_constructor('!python', python_constructor) # Handles !python tags by ignoring them

    # Example list of models
    # models = ["gpt_4o", "gpt_4o_mini", ...]
    # Example list of task types
    # task_types = ["atomic", "combined", "theme"]

    # Create data structures to store results
    task_pass_rates = defaultdict(dict)
    test_case_pass_rates = defaultdict(dict)
    summary_data = []

    for model in models:
        for task_type in task_types:
            # Construct paths using os.path.join for robustness
            base_results_dir = os.path.join(".", "generate_results2", model)
            reports_directory = os.path.join(base_results_dir, f"{task_type}_output", "reports")
            merged_yaml_path = os.path.join(base_results_dir, f"{task_type}_test_reports.yaml")
            statistics_path = os.path.join(base_results_dir, f"{task_type}_test_statistics.json")
            
            config_file_path = os.path.join(".", "dataset_complete", f"{task_type}_code", f"{task_type}_test_config.yaml")

            config = None
            try:
                with open(config_file_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) # Use safe_load for security
            except FileNotFoundError:
                print(f"Error: Config file not found: {config_file_path}")
                continue # Skip this model/task_type combination
            except yaml.YAMLError as e:
                print(f"Error parsing YAML config file {config_file_path}: {e}")
                continue
            except Exception as e:
                print(f"An unexpected error occurred while reading config {config_file_path}: {e}")
                continue
            
            if not os.path.isdir(reports_directory):
                print(f"Warning: Reports directory not found, skipping: {reports_directory}")
                # Add entry to indicate missing data or skip
                task_pass_rates[model][task_type] = "N/A (No reports)"
                test_case_pass_rates[model][task_type] = "N/A (No reports)"
                summary_data.append([
                    model, task_type, 0, 0, "N/A (No reports)", "N/A (No reports)"
                ])
                continue

            # Ensure output directories for this model/task_type exist
            try:
                os.makedirs(os.path.dirname(merged_yaml_path), exist_ok=True)
                os.makedirs(os.path.dirname(statistics_path), exist_ok=True)
            except OSError as e:
                print(f"Error creating directories for {model}/{task_type}: {e}")
                continue


            # stats = process_test_reports(reports_directory, merged_yaml_path, statistics_path, config)
            stats = process_test_reports_only_first(reports_directory, merged_yaml_path, statistics_path, config)
            
            print("Summary statistics:")
            print(f"Total tasks: {stats.get('total_tasks', 'N/A')}")
            print(f"Total test cases: {stats.get('total_test_cases', 'N/A')}")
            print(f"Task pass rate: {stats.get('task_pass_rate_percentage', 'N/A')}")
            print(f"Test case pass rate: {stats.get('test_case_pass_rate_percentage', 'N/A')}")
            print(f"Finished processing model: {model}, type: {task_type}\n")

            # Save pass rate data
            task_pass_rates[model][task_type] = f"{stats.get('task_pass_rate_percentage', 'N/A')}"
            test_case_pass_rates[model][task_type] = f"{stats.get('test_case_pass_rate_percentage', 'N/A')}"

            # Save detailed data
            summary_data.append([
                model,
                task_type,
                stats.get("total_tasks", 0),
                stats.get("total_test_cases", 0),
                f"{stats.get('task_pass_rate_percentage', 'N/A')}",
                f"{stats.get('test_case_pass_rate_percentage', 'N/A')}"
            ])

    # Output task pass rate report
    print("\n===== Task Pass Rate Report =====")
    print_csv_table(task_pass_rates, task_types, "Model")

    # Output test case pass rate report
    print("\n===== Test Case Pass Rate Report =====")
    print_csv_table(test_case_pass_rates, task_types, "Model")

    # Save detailed report to CSV file
    csv_summary_report_path = os.path.join(".", "generate_results2", "test_summary_report.csv")
    summary_headers = ["Model", "Task Type", "Total Tasks", "Total Test Cases", "Task Pass Rate (%)", "Test Case Pass Rate (%)"]
    save_csv_file(summary_data, summary_headers, csv_summary_report_path)
    print(f"\nDetailed report saved to: {csv_summary_report_path}")


def print_csv_table(data_dict, columns, header_name):
    """Prints a table in CSV format to the console."""
    # Print header
    header = f"{header_name}," + ",".join(columns)
    print(header)

    # Print each row
    for key_item in sorted(data_dict.keys()): # Sort by model name for consistent output
        row_values = [str(key_item)]
        for col in columns:
            row_values.append(str(data_dict[key_item].get(col, "N/A")))
        print(",".join(row_values))


def save_csv_file(data_rows, headers, file_path):
    """Saves data to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8", newline='') as f: # Add newline='' for csv
            # Write header
            f.write(",".join(headers) + "\n")
            # Write data rows
            for row in data_rows:
                f.write(",".join(map(str, row)) + "\n")
    except IOError as e:
        print(f"Error saving CSV file {file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving CSV {file_path}: {e}")


def delete_unwanted_files(models, task_types, base_dir="generate_results"):
    """
    Deletes specified unwanted files for given models and task types.

    Args:
        models (list): List of model names.
        task_types (list): List of task types.
        base_dir (str): Base directory where model results are stored (e.g., "generate_results" or "generate_results2").
    """
    # Example list of models
    # models = ["gpt_4o", ...]

    for model in models:
        for task_type in task_types:
            # Delete test report files (test_summary.json)
            unwanted_reports_path = os.path.join(".", base_dir, model, f"{task_type}_output", "test_summary.json")
            try:
                if os.path.exists(unwanted_reports_path):
                    os.remove(unwanted_reports_path)
                    print(f"Deleted file: {unwanted_reports_path}")
            except OSError as e:
                print(f"Error deleting file {unwanted_reports_path}: {e}")

            # Delete merged YAML files (*_test_reports.yaml)
            unwanted_yaml_path = os.path.join(".", base_dir, model, f"{task_type}_test_reports.yaml")
            try:
                if os.path.exists(unwanted_yaml_path):
                    os.remove(unwanted_yaml_path)
                    print(f"Deleted file: {unwanted_yaml_path}")
            except OSError as e:
                print(f"Error deleting file {unwanted_yaml_path}: {e}")


def calculate_pass_at_k(model_basename, task_type, k_attempts, base_results_dir="generate_results2"):
    """
    Calculates pass@k for tasks and test cases.

    Args:
        model_basename (str): The base name of the model (e.g., "gpt_4o").
        task_type (str): Task type (e.g., "atomic", "combined", "theme").
        k_attempts (int): The 'k' value for pass@k (number of attempts).
        base_results_dir (str): Base directory for results (e.g., "generate_results2").

    Returns:
        dict: A dictionary containing task and test case pass rates and counts.
    """
    report_file_name = f"{task_type}_test_reports.yaml"

    # To store the pass status of each task and test case across k attempts
    tasks_passed_in_k = {}  # {task_name: True if passed in any of k attempts}
    test_cases_passed_in_k = {}  # {(task_name, test_case_id): True if passed in any of k attempts}
    
    all_tasks_identifiers = set() # To count total unique tasks encountered
    all_test_cases_identifiers = set() # To count total unique test cases encountered

    processed_any_report = False

    # Iterate through the k attempts/reports
    for i in range(1, k_attempts + 1):
        # Construct path for each attempt, e.g., model_1, model_2
        current_attempt_model_dir = f"{model_basename}_{i}"
        report_path = os.path.join(base_results_dir, current_attempt_model_dir, report_file_name)

        if not os.path.exists(report_path):
            print(f"Warning: Report file not found for attempt {i}: {report_path}")
            continue
        
        processed_any_report = True
        report_data = None
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = yaml.safe_load(f)
        except FileNotFoundError: # Should be caught by os.path.exists, but good practice
            print(f"Error: Report file disappeared before opening: {report_path}")
            continue
        except yaml.YAMLError as e:
            print(f"Error parsing YAML report {report_path}: {e}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred reading report {report_path}: {e}")
            continue

        if not report_data: # If yaml.safe_load returns None or empty
            print(f"Warning: Report data is empty for {report_path}")
            continue

        # Update the pass status of tasks and test cases
        for task_name, task_info in report_data.items():
            all_tasks_identifiers.add(task_name)
            # Initialize task status (if not already initialized for this k-evaluation)
            if task_name not in tasks_passed_in_k:
                tasks_passed_in_k[task_name] = False

            # If task status is 'passed' in this attempt, mark as passed for pass@k
            if task_info.get('status') == 'passed':
                tasks_passed_in_k[task_name] = True

            # Update test case pass status
            for test_case in task_info.get('test_cases', []):
                test_case_id = test_case.get('test_case_id')
                if test_case_id is None: # Skip if test_case_id is missing
                    continue
                case_key = (task_name, test_case_id)
                all_test_cases_identifiers.add(case_key)

                # Initialize test case status (if not already initialized for this k-evaluation)
                if case_key not in test_cases_passed_in_k:
                    test_cases_passed_in_k[case_key] = False

                # If test case passed in this attempt, mark as passed for pass@k
                if test_case.get('status') == 'passed':
                    test_cases_passed_in_k[case_key] = True
    
    # If no report was processed (e.g., all k report files were missing)
    if not processed_any_report:
        return {
            "task_pass_rate": 0,
            "test_case_pass_rate": 0,
            "tasks_passed_count": 0,
            "total_tasks": 0,
            "test_cases_passed_count": 0,
            "total_test_cases": 0,
            "processed_any_report": False
        }

    # Calculate pass rates based on unique identifiers found across all reports
    total_tasks = len(all_tasks_identifiers)
    tasks_passed_count = sum(1 for task_name in all_tasks_identifiers if tasks_passed_in_k.get(task_name, False))

    total_test_cases = len(all_test_cases_identifiers)
    test_cases_passed_count = sum(1 for case_key in all_test_cases_identifiers if test_cases_passed_in_k.get(case_key, False))

    task_pass_rate = tasks_passed_count / total_tasks if total_tasks > 0 else 0
    test_case_pass_rate = test_cases_passed_count / total_test_cases if total_test_cases > 0 else 0

    return {
        "task_pass_rate": task_pass_rate,
        "test_case_pass_rate": test_case_pass_rate,
        "tasks_passed_count": tasks_passed_count,
        "total_tasks": total_tasks,
        "test_cases_passed_count": test_cases_passed_count,
        "total_test_cases": total_test_cases,
        "processed_any_report": True
    }


def calculate_pass_at_k_only_one(model_basename, task_type, k_attempts, base_results_dir="generate_results2", expected_total_tasks=None):
    """
    Calculates pass@k when each task effectively has only one test case considered.
    The pass rate can be divided by `expected_total_tasks` if provided and valid.

    Args:
        model_basename (str): The base name of the model.
        task_type (str): Task type.
        k_attempts (int): The 'k' value.
        base_results_dir (str): Base directory for results.
        expected_total_tasks (int, optional): The expected total number of unique tasks.
                                             If provided, pass rate is tasks_passed_count / expected_total_tasks.
                                             Otherwise, it's tasks_passed_count / actual_total_tasks_found.
    Returns:
        dict: Dictionary containing pass rates and counts.
    """
    report_file_name = f"{task_type}_test_reports.yaml"

    # Store whether each unique task passed in any of the k attempts
    tasks_passed_in_k = {}  # {task_name: True if passed in any of k attempts}
    all_tasks_identifiers = set() # To count total unique tasks encountered
    processed_any_report = False

    # Iterate through k run results/attempts
    for i in range(1, k_attempts + 1):
        current_attempt_model_dir = f"{model_basename}_{i}"
        report_path = os.path.join(base_results_dir, current_attempt_model_dir, report_file_name)

        if not os.path.exists(report_path):
            print(f"Warning: Report file not found for attempt {i}: {report_path}")
            continue
        
        processed_any_report = True
        report_data = None
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Error: Report file disappeared: {report_path}")
            continue
        except yaml.YAMLError as e:
            print(f"Error parsing YAML report {report_path}: {e}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred reading report {report_path}: {e}")
            continue
        
        if not report_data:
            print(f"Warning: Report data is empty for {report_path}")
            continue

        # Update task pass status
        for task_name, task_info in report_data.items():
            all_tasks_identifiers.add(task_name)
            test_cases = task_info.get('test_cases', [])
            
            # If this task is already marked as passed from a previous attempt, skip
            if tasks_passed_in_k.get(task_name, False):
                continue

            # Initialize if not seen before in this k-evaluation
            if task_name not in tasks_passed_in_k:
                tasks_passed_in_k[task_name] = False

            if not test_cases: # No test cases means it didn't pass
                # tasks_passed_in_k[task_name] remains False
                continue

            # Since each task is considered to have only one effective test case for this function's logic
            # (as per process_test_reports_only_first)
            first_test_case = test_cases[0]
            status = first_test_case.get('status', 'unknown')

            if status == 'passed':
                tasks_passed_in_k[task_name] = True
    
    if not processed_any_report:
        return {
            "task_pass_rate": 0,
            "test_case_pass_rate": 0, # Same as task pass rate in this "only_one" context
            "tasks_passed_count": 0,
            "total_tasks": 0,
            "test_cases_passed_count": 0,
            "total_test_cases": 0,
            "processed_any_report": False
        }

    # Total unique tasks encountered across all k reports for this model and task type
    actual_total_tasks_found = len(all_tasks_identifiers)
    tasks_passed_count = sum(1 for task_name in all_tasks_identifiers if tasks_passed_in_k.get(task_name, False))

    # Determine the denominator for the pass rate
    denominator = actual_total_tasks_found
    if expected_total_tasks is not None and isinstance(expected_total_tasks, int) and expected_total_tasks > 0:
        denominator = expected_total_tasks
        if actual_total_tasks_found > expected_total_tasks:
            print(f"Warning: Found {actual_total_tasks_found} tasks for {model_basename}/{task_type}, but expected {expected_total_tasks}. Using expected for rate.")
        elif actual_total_tasks_found < expected_total_tasks and actual_total_tasks_found > 0 : # only warn if some tasks were found but less than expected
             print(f"Warning: Found only {actual_total_tasks_found} tasks for {model_basename}/{task_type}, expected {expected_total_tasks}. Using expected for rate.")


    pass_rate = tasks_passed_count / denominator if denominator > 0 else 0

    return {
        "task_pass_rate": pass_rate,
        "test_case_pass_rate": pass_rate, # In "only_one" context, task pass rate is effectively test case pass rate
        "tasks_passed_count": tasks_passed_count,
        "total_tasks_found": actual_total_tasks_found, # Actual unique tasks found
        "total_tasks_denominator": denominator, # Denominator used for rate calculation
        "test_cases_passed_count": tasks_passed_count, # Same as tasks passed
        "total_test_cases_found": actual_total_tasks_found, # Same as tasks found
        "processed_any_report": True
    }


if __name__ == "__main__":
    # Define models and task types to process
    # These lists might be adjusted based on actual directory names (e.g., without _1, _2 suffixes for process_models)
    # For calculate_pass_at_k, model_basenames are used.
    
    # Models for process_models (expects directories like "gemini_2.0_pro_1", etc.)
    models_for_processing = [
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
    
    all_model_prefixes = [
        "gemini_2.0_pro", "gpt_4o", "gpt_4o_mini", "claude_3_7_sonnet", 
        "deepseek_r1_250120", "deepseek_v3_241226", "deepseek_v3_250324",
        "o3_mini", "codellama_7b", "qwen2.5_3b", "qwen2.5_7b", "qwen2.5_32b",
        "qwen2.5_coder_3b", "qwen2.5_coder_7b", "qwen2.5_coder_32b", "qwq_32b", 
        "deepseek_coder_v2_16b", "geocode_gpt"
    ]
    
    models_for_processing_generated = []
    for prefix in all_model_prefixes:
        max_attempts = 5 
        for i in range(1, max_attempts + 1):
            models_for_processing_generated.append(f"{prefix}_{i}")

    model_basenames_for_pass_k = all_model_prefixes

    task_types_to_process = ["atomic"]

    # --- Step 1: Process individual reports (using _only_first logic) ---
    # This generates the individual YAML reports per model_attempt and task_type
    process_models(models_for_processing_generated, task_types_to_process)


    # --- Step 2: Calculate Pass@k using the reports generated in Step 1 ---
    results_base_dir = "generate_results2"
    expected_tasks_atomic = 1325 

    expected_tasks_map = {
        "atomic": expected_tasks_atomic,
    }


    for task_type in task_types_to_process:
        # Initialize a dictionary to hold pass@k data for DataFrame
        pass_k_table_data = {k_val: {} for k_val in range(1, 6)} # For k=1 to k=5

        current_expected_total_tasks = expected_tasks_map.get(task_type)
        if current_expected_total_tasks is None:
            print(f"Warning: Expected total tasks for type '{task_type}' not defined. Pass rate denominator will be actual tasks found.")


        for model_base in model_basenames_for_pass_k:
            max_k_for_model = 5
            
            for k_val in range(1, 6):
                if k_val > max_k_for_model:
                    pass_k_table_data[k_val][model_base] = "N/A (k > max attempts)"
                    continue

                # Use calculate_pass_at_k_only_one as per original script's structure
                pass_k_stats = calculate_pass_at_k_only_one(
                    model_base, 
                    task_type, 
                    k_val, 
                    base_results_dir=results_base_dir,
                    expected_total_tasks=current_expected_total_tasks
                )
                
                # Store the task_pass_rate (which is also test_case_pass_rate in this context)
                # Format as percentage string for the table
                pass_k_table_data[k_val][model_base] = f"{pass_k_stats['task_pass_rate']:.2%}" if pass_k_stats["processed_any_report"] else "N/A (No reports)"
                
                print(f"Processed pass@{k_val} for model: {model_base}, task_type: {task_type}. Rate: {pass_k_table_data[k_val][model_base]}")

        # Convert pass@k data to DataFrame for this task_type
        df_pass_k = pd.DataFrame(pass_k_table_data)
        df_pass_k.index.name = "Model" # Set index name
        df_pass_k = df_pass_k.rename(columns={k: f"Pass@{k}" for k in df_pass_k.columns}) # Rename columns to Pass@1, Pass@2 etc.


        print(f"\n=== Pass@k Table - Task Type: {task_type} ===")
        print(df_pass_k)

        # Save the DataFrame to CSV
        pass_k_csv_path = os.path.join(results_base_dir, f"pass_at_k_summary_{task_type}.csv")
        try:
            os.makedirs(os.path.dirname(pass_k_csv_path), exist_ok=True)
            df_pass_k.to_csv(pass_k_csv_path)
            print(f"Pass@k table for {task_type} saved to: {pass_k_csv_path}")
        except IOError as e:
            print(f"Error saving Pass@k CSV for {task_type} to {pass_k_csv_path}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while saving Pass@k CSV for {task_type}: {e}")

    print("\nProcessing finished.")