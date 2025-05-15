import os
import re
import numpy as np
import yaml
import ee
import geemap
import json
import requests
import rasterio
import shutil
import time
import types
from shapely.geometry import shape

GEE_PROJECT_NAME = YOUR_GEE_PROJECT_NAME  # Replace with your GEE project name

def download_file(url, destination, max_retries=3, retry_delay=30):
    """
    Download a file with retry mechanism.
    :param url: Download URL
    :param destination: Destination file path
    :param max_retries: Maximum number of retries
    :param retry_delay: Delay between retries (seconds)
    :return: (success_flag, error_message_or_None)
    """
    for retry in range(max_retries + 1):
        try:
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            response = requests.get(url, stream=True, timeout=100)
            response.raise_for_status()  # Raise an exception for HTTP errors

            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return True, None
        except Exception as e:
            error_str = str(e)

            # Check for network errors
            network_error = any(err in error_str for err in [
                "HTTPSConnectionPool",
                "500 Server Error",
                "Internal Server Error",
                "Client Error",
                "Bad Request for url"
                "Timeout"
            ])

            if network_error and retry < max_retries:
                wait_time = retry_delay
                print(
                    f"⚠️ Network error downloading file ({retry + 1}/{max_retries}), waiting {wait_time} seconds to retry...")
                time.sleep(wait_time)
                continue
            else:
                if retry == max_retries:
                    return False, f"Max retries exceeded. Last error: {error_str}"
                else:
                    return False, f"Error downloading file: {error_str}"


def python_constructor(loader, node):
    """
    Custom constructor for parsing Python code in !python tags.
    """
    python_code = loader.construct_scalar(node)  # Get the code content
    local_vars = {}  # Define local variable space
    global_vars = globals().copy()  # Copy global variables
    try:
        # Dynamically execute the code
        exec(compile(python_code, "<yaml>", "exec"), global_vars, local_vars)
        # Check for function return
        for var_name, var_value in local_vars.items():
            if isinstance(var_value, types.FunctionType):
                return var_value  # Return the function object
        return None  # Return None if no function is found
    except Exception as e:
        print(f"Error executing Python code: {e}")
        return None


def get_params_data(params_data):
    """
    Process parameter data and execute any Python functions.
    """
    processed_params = {}
    for key, value in params_data.items():
        if callable(value):  # If the value is a function
            try:
                result = value()  # Execute the function
                if isinstance(result, str):
                    result = result.replace("<ee-project>", GEE_PROJECT_NAME)
                processed_params[key] = result  # Dynamically execute the function and get the result
            except Exception as e:
                print(f"Error executing function for key '{key}': {e}")
                processed_params[key] = None
        else:
            if isinstance(value, str):
                value = value.replace("<ee-project>", GEE_PROJECT_NAME)
            processed_params[key] = value  # Keep the ordinary value
    return processed_params


def get_download_url_with_retry(result, region, max_retries=3, retry_delay=30):
    """
    Attempts to get a download URL for a GEE result with a specified number of retries.

    :param result: The GEE object (e.g., Image, FeatureCollection) to get a download URL for.
    :param region: The geometry defining the region for the download.
    :param max_retries: Maximum number of retry attempts.
    :param retry_delay: Delay in seconds between retries.
    :return: A tuple (success_flag, response_or_error_message).
             success_flag is True if URL is obtained, False otherwise.
             response_or_error_message is the URL string or an error message.
    """
    for retry in range(max_retries + 1):
        try:
            result_url = result.getDownloadURL({
                'region': region,
                'crs': 'EPSG:4326',
                'scale': 30,
                'format': 'GeoTIFF'
            })
            return True, result_url
        except Exception as e:
            error_str = str(e)

            # Handle special case for large regions
            if ("Total request size" in error_str) or ("Pixel grid dimensions" in error_str):
                try:
                    centroid = region.centroid(maxError=1)
                    smaller_region = centroid.buffer(1000).bounds()
                    smaller_result = result.clip(smaller_region)
                    result_url = smaller_result.getDownloadURL({
                        'region': smaller_region,
                        'crs': 'EPSG:4326',
                        'scale': 30,
                        'format': 'GeoTIFF'
                    })
                    return True, result_url
                except Exception as inner_e:
                    error_str = f"Error after region reduction: {str(inner_e)}"

            # Handle network errors
            network_error = any(err in error_str for err in [
                "HTTPSConnectionPool",
                "500 Server Error",
                "Internal Server Error",
                "Connection refused",
                "Connection reset",
                "Timeout",
                "Too Many Requests"
            ])

            if network_error and retry < max_retries:
                wait_time = retry_delay
                print(
                    f"⚠️ Network error getting download URL ({retry + 1}/{max_retries}), waiting {wait_time} seconds to retry...")
                time.sleep(wait_time)
                continue
            else:
                if retry == max_retries:
                    return False, f"Max retries exceeded. Last error: {error_str}"
                else:
                    return False, f"Error getting download URL: {error_str}"


def test_single_file(file_path, config, output_directory, ref_answer_dir, max_retries=3, retry_delay=60):
    """
    Tests a single code file against a GEE test configuration.

    :param file_path: Path to the code file to test.
    :param config: Dictionary containing the test configuration.
    :param output_directory: Directory to save test outputs.
    :param ref_answer_dir: Directory containing reference answers.
    :param max_retries: Maximum retries for operations like downloads.
    :param retry_delay: Delay between retries.
    :return: A dictionary with statistics for the tested file.
    """
    file_name = os.path.basename(file_path)
    print(f"\n{'=' * 50}")
    print(f"Testing file: {file_name}")
    print(f"{'=' * 50}")

    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    # Add initialization code
    init_code = (f"import ee\nimport geemap\n"
                 f"geemap.set_proxy(port=7890)\nee.Initialize(project='{GEE_PROJECT_NAME}')\n")
    code_added = init_code + code

    # File-level test statistics
    file_stats = {
        "file_name": file_name,
        "function_name": None,
        "total_test_cases": 0,
        "passed_test_cases": 0,
        "failed_test_cases": 0,
        "skipped_test_cases": 0,
        "file_errors": [],  # File-level errors
        "test_cases": [],  # Detailed results of each test case
        "status": "skipped"  # Initial status is skipped
    }

    # Attempt to parse and execute the file code
    file_execution_success = False
    local_vars = {}
    function_name = None

    # Create test assets for asset tasks
    if "asset" in file_name.lower():
        init_test_assets()

    for retry in range(max_retries + 1):
        try:
            # Use regex to extract function name
            function_name_pattern = r"def (\w+)\("  # Match def followed by function name and parentheses
            function_matches = re.findall(function_name_pattern, code_added)

            if not function_matches:
                error_msg = "No function definition found in the file"
                file_stats["file_errors"].append(error_msg)
                print(error_msg)
                break

            function_name = function_matches[0]
            file_stats["function_name"] = function_name

            # Execute the code
            exec(code_added, globals(), local_vars)
            file_execution_success = True

            break

        except Exception as e:
            error_str = str(e)
            network_error = any(err in error_str for err in [
                "HTTPSConnectionPool",
                "500 Server Error",
                "Internal Server Error",
                "Client Error",
                "Bad Request for url"
                "Timeout"
            ])

            if network_error and retry < max_retries:
                print(
                    f"⚠️ Network error during code execution ({retry + 1}/{max_retries}), waiting {retry_delay} seconds to retry...")
                file_stats["file_errors"].append(
                    f"Network error during code execution (attempt {retry + 1}): {error_str}")
                time.sleep(retry_delay)
                continue
            else:
                if network_error:
                    error_msg = f"Max retries exceeded during code execution. Last error: {error_str}"
                else:
                    error_msg = f"Error executing code: {error_str}"

                print(f"❌ {error_msg}")
                file_stats["file_errors"].append(error_msg)
                break

    # If file code execution fails, skip the file
    if not file_execution_success:
        file_stats["status"] = "failed"
        print(f"❌ File execution failed, skipping test cases")
        return file_stats

    # Get all test data for the function
    test_cases = config.get(function_name, [])
    print(f"Testing function: {function_name}")

    if len(test_cases) == 0:
        error_msg = f"No test cases found for {function_name}"
        file_stats["file_errors"].append(error_msg)
        print(print(f"❌ No test cases found for {function_name}"))
        file_stats["status"] = "skipped"
        return file_stats

    file_stats["total_test_cases"] = len(test_cases)

    # Execute each test case
    for i, test_case in enumerate(test_cases):
        test_case_result = {
            "test_case_id": i + 1,
            "status": "skipped",
            "error": None,
            "retry_count": 0
        }

        print(f"Running test case {i + 1}/{len(test_cases)}...")

        # Retry mechanism for test cases
        for retry in range(max_retries + 1):
            test_case_result["retry_count"] = retry

            try:
                params = test_case['params']
                params_data = params.copy()
                params_data = get_params_data(params_data)

                # Pass the processed parameter data to the function
                result = local_vars[function_name](**params_data)

                flag, message = check_result(result, test_case, ref_answer_dir, output_directory)

                if flag == False:
                    message = str(message)
                    print(f"❌ Test case {i + 1} failed: {message}")
                    test_case_result["status"] = "failed"
                    test_case_result["error"] = message
                    file_stats["failed_test_cases"] += 1
                    break
                else:
                    print(f"✅ Test case {i + 1} passed!")
                    test_case_result["status"] = "passed"
                    file_stats["passed_test_cases"] += 1
                    break

            except Exception as e:
                error_str = str(e)
                network_error = any(err in error_str for err in [
                    "HTTPSConnectionPool",
                    "500 Server Error",
                    "Internal Server Error",
                    "Client Error",
                    "Bad Request for url",
                    "Timeout"
                ])

                if network_error and retry < max_retries:
                    print(
                        f"⚠️ Network error in test case {i + 1} ({retry + 1}/{max_retries}), waiting {retry_delay} seconds to retry...")
                    time.sleep(retry_delay)
                    continue
                elif error_str.strip() == "Ok" or error_str.strip() == "None":
                    # This is a bug in GEE, which sometimes raises "Ok" or "None" when the task(export) is completed
                    print(f"✅ Test case {i + 1} passed!")
                    test_case_result["status"] = "passed"
                    file_stats["passed_test_cases"] += 1
                    break
                else:
                    if network_error:
                        error_msg = f"Max retries exceeded. Last error: {error_str}"
                    else:
                        error_msg = f"Error in test case: {error_str}"

                    print(f"❌ Test case {i + 1} failed: {error_msg}")
                    test_case_result["status"] = "failed"
                    test_case_result["error"] = error_msg
                    file_stats["failed_test_cases"] += 1
                    break

        # If the test case is skipped
        if test_case_result["status"] == "skipped":
            file_stats["skipped_test_cases"] += 1

        # Add test case result to file statistics
        file_stats["test_cases"].append(test_case_result)

    # Delete test assets if the file name contains "asset" (case-insensitive)
    if "asset" in file_name.lower():
        delete_test_assets()

    # Determine the overall status of the file based on test case results
    if file_stats["failed_test_cases"] == 0 and file_stats["passed_test_cases"] > 0:
        file_stats["status"] = "passed"
        print(f"\n✅ All test cases passed for {file_name}")
    elif file_stats["passed_test_cases"] == 0:
        file_stats["status"] = "failed"
        file_stats["status"] = "partial"  # 部分测试用例通过
        print(
            f"\n⚠️ {file_stats['passed_test_cases']}/{file_stats['total_test_cases']} test cases passed for {file_name}")
    else:
        file_stats["status"] = "partial"  # 部分测试用例通过
        print(
            f"\n⚠️ {file_stats['passed_test_cases']}/{file_stats['total_test_cases']} test cases passed for {file_name}")

    return file_stats


def init_test_assets(max_retries=3, retry_delay=30):
    """
    Initializes test assets in Google Earth Engine.
    Creates necessary folders and uploads test data if they don't exist.

    :param max_retries: Maximum number of retries for GEE operations.
    :param retry_delay: Delay in seconds between retries.
    """
    def create_test_asset(test_folder, asset_name):
        """Helper function to create a specific test asset."""
        image = ee.Image.constant(1).rename('constant_image').set({'test': True})
        full_path = f'{test_folder}/{asset_name}'

        task = ee.batch.Export.image.toAsset(
            image=image,
            description=asset_name,
            assetId=full_path,
            scale=1000,
            maxPixels=1e9,
            region=ee.Geometry.Rectangle([120, 30, 121, 31], 'EPSG:4326', False)
        )
        task.start()
        return full_path, task

    def wait_for_tasks(timeout=60, interval=2):
        """Helper function to wait for GEE tasks to complete."""
        start_time = time.time()
        while True:
            tasks = ee.data.listOperations()
            active_tasks = [t for t in tasks if t['metadata']['state'] in ['QUEUED', 'RUNNING']]
            if not active_tasks:
                return
            if time.time() - start_time > timeout:
                print("Error when creating test assets: timeout!\n")
                return
            time.sleep(interval)

    def delete_all_assets(folder_path):
        """Helper function to delete all assets within a specified GEE folder."""
        try:
            assets = ee.data.listAssets(folder_path)
            for asset in assets.get('assets', []):
                ee.data.deleteAsset(asset["name"])
        except Exception as e:
            print(f"⚠️ Error deleting assets from {folder_path}: {str(e)}")

    folder_path = f"projects/{GEE_PROJECT_NAME}/assets/test-assets"
    folder_path2 = f"projects/{GEE_PROJECT_NAME}/assets/test-assets-to-list"

    for attempt in range(max_retries + 1):  # Includes the first attempt and up to max_retries retries
        try:
            # 清理旧资产
            delete_all_assets(folder_path)
            delete_all_assets(folder_path2)

            # 创建文件夹
            try:
                ee.data.createFolder(folder_path)
            except Exception as e:
                pass
            try:
                ee.data.createFolder(folder_path2)
            except Exception as e:
                pass

            # 创建图像资产
            for i in range(3):
                create_test_asset(folder_path, f'test_image_{i}')
            create_test_asset(folder_path2, 'test_image_3')

            # 创建表格资产
            table_asset_id = 'projects/ee-szx/assets/test-assets/test_table_asset'
            fc = ee.FeatureCollection([
                ee.Feature(ee.Geometry.Point([-122.22599, 37.77045]), {'name': 'Point A', 'id': 1}),
                ee.Feature(ee.Geometry.Point([-118.24368, 34.05223]), {'name': 'Point B', 'id': 2}),
                ee.Feature(ee.Geometry.Point([-115.1398, 36.1699]), {'name': 'Point C', 'id': 3})
            ])
            task = ee.batch.Export.table.toAsset(
                collection=fc,
                description='test_table_export',
                assetId=table_asset_id,
            )
            task.start()

            wait_for_tasks()

            # 检查资产数量是否达标
            assets = ee.data.listAssets(folder_path)
            if len(assets.get('assets', [])) >= 4:
                print("✅ All test assets created successfully.")
                return
            else:
                print("❌ Not enough assets were created. Retrying...")
                raise Exception("Not enough assets created.")

        except Exception as e:
            error_str = str(e)
            network_error = any(err in error_str for err in [
                "HTTPSConnectionPool",
                "500 Server Error",
                "Internal Server Error",
                "Client Error",
                "Bad Request for url",
                "Timeout",
                "googleapiclient.errors.HttpError"
            ])

            if (network_error or "Not enough assets created." in error_str) and attempt < max_retries:
                print(f"⚠️ Error occurred: {error_str}")
                print(f"🔄 Retrying... ({attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"❌ Failed to initialize test assets after {attempt} attempts: {error_str}")
                return


def delete_test_assets():
    """
    Deletes predefined test assets and folders from Google Earth Engine.
    This is typically used for cleanup after tests.
    """
    folder_path = f"projects/{GEE_PROJECT_NAME}/assets/test-assets"
    folder_path2 = f"projects/{GEE_PROJECT_NAME}/assets/test-assets-to-list"

    try:     
        assets = ee.data.listAssets(folder_path)
        for asset in assets['assets']:
            ee.data.deleteAsset(asset["name"])
        assets2 = ee.data.listAssets(folder_path2)
        for asset in assets2['assets']:
            ee.data.deleteAsset(asset["name"])
        test_folder_path = f"projects/{GEE_PROJECT_NAME}/assets/test-assets/test_folder"
        ee.data.deleteAsset(test_folder_path)
    except Exception as e:
        pass # Suppress exceptions during deletion, as assets might not exist

    return


def run_code_from_txt(code_directory, yaml_path, output_directory, type, reference_directory="./test_code", max_retries=3, retry_delay=60):
    """
    Reads code files from a specified directory, executes tests using YAML configuration,
    and saves the results to the output directory.

    :param code_directory: Path to the directory containing code files.
    :param yaml_path: Path to the YAML configuration file.
    :param output_directory: Path to the directory for output results.
    :param type: Test type: atomic, combined, or theme.
    :param reference_directory: Path to the directory for reference answers.
    :param max_retries: Maximum number of retries for network errors.
    :param retry_delay: Delay time between retries (seconds).
    :return: None
    """
    # Register YAML constructor for custom Python objects
    yaml.add_constructor('!python', python_constructor)

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Create subdirectories for passed, failed tests, output results, and reports
    passed_dir = os.path.join(output_directory, "passed")
    failed_dir = os.path.join(output_directory, "failed")
    output_dir = os.path.join(output_directory, "output_results")
    report_dir = os.path.join(output_directory, "reports")
    ref_dir = os.path.join(reference_directory, f"{type}_code/ref_answer")

    os.makedirs(passed_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # Get all .txt code files from the specified directory
    files = [f for f in os.listdir(code_directory) if f.endswith('.txt')]

    # Read YAML configuration
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    # Global test statistics
    total_files = len(files)
    passed_files = 0
    failed_files = 0
    skipped_files = 0

    print(f"Found {total_files} files to test")

    # Test each file
    for file in files:
        file_path = os.path.join(code_directory, file)

        # Test a single file
        file_stats = test_single_file(
            file_path,
            config,
            output_dir,
            ref_dir,
            max_retries,
            retry_delay
        )

        # Move file based on test result and update statistics
        if file_stats["status"] == "passed":
            passed_files += 1
            shutil.move(file_path, os.path.join(passed_dir, file))
        elif file_stats["status"] == "failed" or file_stats["status"] == "partial":
            failed_files += 1
            shutil.move(file_path, os.path.join(failed_dir, file))
        else:
            skipped_files += 1 # Increment for skipped or other statuses

        # Save test report for the current file
        report_path = os.path.join(report_dir, f"{file.replace('.txt', '')}_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(file_stats, f, indent=2, ensure_ascii=False)

    # Generate overall test summary report
    summary_report = {
        "total_files": total_files,
        "passed_files": passed_files,
        "failed_files": failed_files,
        "skipped_files": skipped_files,
        "pass_rate": f"{(passed_files / total_files * 100):.2f}%" if total_files > 0 else "0%"
    }

    summary_path = os.path.join(output_directory, "test_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Total files: {total_files}")
    print(f"Passed files: {passed_files}")
    print(f"Failed files: {failed_files}")
    print(f"Skipped files: {skipped_files}")
    print(f"Pass rate: {summary_report['pass_rate']}")
    print("Test report saved to: " + summary_path)
    print("=" * 50)


def check_result(result, test_case, ref_answer_dir, output_directory):
    """
    Checks the result of a GEE operation against an expected answer.
    Handles various GEE data types and saves output/comparison.

    :param result: The GEE object or Python native type result from the executed code.
    :param test_case: Dictionary containing test case details (e.g., out_type, expected_answer).
    :param ref_answer_dir: Directory containing reference answer files.
    :param output_directory: Directory to save the actual output from the GEE operation.
    :return: A tuple (flag, message).
             flag is True if the result matches the expected answer, False otherwise.
             message provides details on assertion errors or other issues.
    """
    # Process results based on output type
    out_type = test_case['out_type']
    expected_answer = test_case['expected_answer']
    answer_path = os.path.join(ref_answer_dir, expected_answer)
    os.makedirs(os.path.dirname(answer_path), exist_ok=True)

    function_case_name = test_case['expected_answer'].split('.')[0]
    result_filename = f"{function_case_name}_output"
    result_path = os.path.join(output_directory, result_filename)

    # WARNING: ONLY USE IT WHEN DEBUGGING
    # result_path = answer_path

    message = None # Initialize message for potential errors

    if out_type == "ee.Image":
        # Process image type results: compare if the image array is correct
        try:
            region = result.geometry()
            # If region is not defined or unbounded, use a default small region
            if (not region.getInfo()) or (region.isUnbounded().getInfo()):
                region = ee.Geometry.BBox(-1, -1, 1, 1)
                result = result.clip(region)
        except Exception:
            # Default region if geometry access fails
            region = ee.Geometry.BBox(-1, -1, 1, 1)
            result = result.clip(region)

        info = result.getInfo()
        bands = len(info['bands'])

        if bands == 0:
            print("Warning: Empty image result")
            result_array = np.array([-1]) # Use a placeholder for empty images
        else:
            # Get the download URL for the image
            get_flag, response = get_download_url_with_retry(result, region)
            if not get_flag:
                return False, response # Return error if URL retrieval fails
            result_url = response

            # Download GeoTIFF file
            temp_tif = os.path.join(output_directory, f"temp_image.tif")
            download_flag , error = download_file(result_url, temp_tif)
            if not download_flag:
                return False, error # Return error if download fails

            # Read raster data and convert to NumPy array
            result_raster = rasterio.open(temp_tif).read()
            result_numpy = np.array(result_raster).transpose(1, 2, 0) # Transpose to (height, width, bands)
            result_array = np.round(result_numpy, 3) # Round to 3 decimal places

            # Delete temporary TIF file
            if os.path.exists(temp_tif):
                os.remove(temp_tif)

        # Save as NPY file
        if not result_path.endswith('.npy'):
            result_path = f"{result_path}.npy"
        if not answer_path.endswith('.npy'):
            answer_path = f"{answer_path}.npy"
        np.save(result_path, result_array)

        # Check answer
        answer_array = np.load(answer_path, allow_pickle=True)
        # print("Expected:", answer_array) # For debugging
        # print("Got:", result_array)      # For debugging
        try :
            np.testing.assert_array_almost_equal(result_array, answer_array, decimal=3)
            flag = True
        except AssertionError as e:
            flag = False
            message = e
        except Exception as e:
            flag = False
            message = e

    elif out_type == "ee.ArrayImage": # This type does not exist in GEE, used only for result checking
        # Ensure file extension is .npy
        if not result_path.endswith('.npy'):
            result_path = f"{result_path}.npy"
        if not answer_path.endswith('.npy'):
            answer_path = f"{answer_path}.npy"

        point = ee.Geometry.Point([120.05, 30.05]) # Define a sample point
        # Attempt to sample the array image using common property names
        result_array0 = result.sample(point, 500).first().get('array').getInfo()
        result_array = np.array(result_array0)
        if result_array0 is None: # Fallback property names
            result_array0 = result.sample(point, 500).first().get('U').getInfo()
            result_array = np.array(result_array0)
        if result_array0 is None:
            result_array0 = result.sample(point, 500).first().get('Q').getInfo()
            result_array = np.array(result_array0)
        if result_array0 is None:
            result_array0 = result.sample(point, 500).first().get('L').getInfo()
            result_array = np.array(result_array0)
        if result_array0 is None:
            result_array0 = result.sample(point, 500).first().get('identity').getInfo()
            result_array = np.array(result_array0)
        if result_array0 is None:
            result_array0 = result.sample(point, 500).first().get('constant').getInfo()
            result_array = np.array(result_array0)

        # Save as NPY file
        np.save(result_path, result_array)

        # Check answer
        answer_array = np.load(answer_path, allow_pickle=True)
        # print("Expected:", answer_array) # For debugging
        # print("Got:", result_array)      # For debugging
        try:
            np.testing.assert_array_almost_equal(result_array, answer_array, decimal=3)
            flag = True
        except AssertionError as e:
            flag = False
            message = e
        except Exception as e:
            flag = False
            message = e

    elif out_type == "ee.ImageCollection":
        # Check if the ImageCollection is empty
        result_size = result.size().getInfo()
        if result_size == 0:
            print("Warning: Empty ImageCollection result")
            result_array = np.array([-1]) # Placeholder for empty collection
            if not result_path.endswith('.npy'): # Ensure correct extension
                result_path = f"{answer_path}.npy" # Fallback to answer path structure if needed
            np.save(result_path, result_array)
        else:
            # Process ImageCollection results: compare if the first image is correct
            result = result.first() # Get the first image for comparison
            try:
                region = result.geometry()
                # If region is not defined or unbounded, use a default small region
                if (not region.getInfo()) or (region.isUnbounded().getInfo()):
                    region = ee.Geometry.BBox(-1, -1, 1, 1)
                    result = result.clip(region)
            except Exception:
                # Default region if geometry access fails
                region = ee.Geometry.BBox(-1, -1, 1, 1)
                result = result.clip(region)

            # Get the download URL for the image
            get_flag, response = get_download_url_with_retry(result, region)
            if not get_flag:
                return False, response
            result_url = response

            # Download GeoTIFF file
            temp_tif = os.path.join(output_directory, f"temp_image.tif")
            download_flag, error = download_file(result_url, temp_tif)
            if not download_flag:
                return False, error

            # Read raster data and convert to NumPy array
            result_raster = rasterio.open(temp_tif).read()
            result_numpy = np.array(result_raster).transpose(1, 2, 0)
            result_array = np.round(result_numpy, 3)

            # Save as NPY file
            if not result_path.endswith('.npy'):
                result_path = f"{result_path}.npy"
            if not answer_path.endswith('.npy'):
                answer_path = f"{answer_path}.npy"
            np.save(result_path, result_array)

            # Delete temporary TIF file
            if os.path.exists(temp_tif):
                os.remove(temp_tif)

        # Check answer
        answer_array = np.load(answer_path, allow_pickle=True)
        # print("Expected:", answer_array) # For debugging
        # print("Got:", result_array)      # For debugging
        try :
            np.testing.assert_array_almost_equal(result_array, answer_array, decimal=3)
            flag = True
        except AssertionError as e:
            flag = False
            message = e
        except Exception as e:
            flag = False
            message = e

    elif out_type == "ee.Geometry":
        # Ensure file extension is .geojson
        if not result_path.endswith('.geojson'):
            result_path = f"{result_path}.geojson"
        if not answer_path.endswith('.geojson'):
            answer_path = f"{answer_path}.geojson"

        result_info = result.getInfo() # Get GeoJSON representation
        result_geojson_str = json.dumps(result_info) # Convert to string

        # Save as GeoJSON file
        with open(result_path, encoding='utf-8', mode='w') as f:
            f.write(result_geojson_str)

        # Check answer
        with open(answer_path, 'r', encoding='utf-8') as f:
            answer_geojson_dict = json.load(f)

        answer_geometry = shape(answer_geojson_dict) # Convert dict to shapely geometry
        result_geometry = shape(result_info)       # Convert dict to shapely geometry
        # print("Expected:", answer_geojson_dict) # For debugging
        # print("Got:", result_info)             # For debugging
        if answer_geometry.equals(result_geometry): # Compare geometries
            flag = True
        else:
            flag = False

    elif out_type == "ee.FeatureCollection" or out_type == "ee.Feature":
        # Ensure file extension is .geojson
        if not result_path.endswith('.geojson'):
            result_path = f"{result_path}.geojson"
        if not answer_path.endswith('.geojson'):
            answer_path = f"{answer_path}.geojson"

        # For Feature/FeatureCollection, compare their geometries
        result_info = result.geometry().getInfo()
        result_geojson_str = json.dumps(result_info)

        # Save as GeoJSON file
        with open(result_path, encoding='utf-8', mode='w') as f:
            f.write(result_geojson_str)

        # Check answer
        with open(answer_path, 'r', encoding='utf-8') as f:
            answer_geojson_dict = json.load(f)
        answer_geometry = shape(answer_geojson_dict)
        result_geometry = shape(result_info)
        # print("Expected:", answer_geojson_dict) # For debugging
        # print("Got:", result_info)             # For debugging
        if answer_geometry.equals(result_geometry):
            flag = True
        else:
            flag = False

    elif out_type == "ee.String":
        # Ensure file extension is .txt
        if not result_path.endswith('.txt'):
            result_path = f"{result_path}.txt"
        if not answer_path.endswith('.txt'):
            answer_path = f"{answer_path}.txt"

        result_str = str(result.getInfo())

        # Save as TXT file
        with open(result_path, encoding='utf-8', mode='w') as f:
            f.write(result_str)

        # Check answer
        with open(answer_path, encoding='utf-8', mode='r') as f:
            answer_str = f.read()
        # print("Expected:", answer_str) # For debugging
        # print("Got:", result_str)      # For debugging
        if result_str == answer_str:
            flag = True
        else:
            flag = False

    elif out_type == "ee.Number":
        # Ensure file extension is .txt
        if not result_path.endswith('.txt'):
            result_path = f"{result_path}.txt"
        if not answer_path.endswith('.txt'):
            answer_path = f"{answer_path}.txt"

        result_num = result.getInfo() # GEE Number to Python float/int

        # Save as TXT file
        with open(result_path, encoding='utf-8', mode='w') as f:
            f.write(str(result_num))

        # Check answer
        with open(answer_path, encoding='utf-8', mode='r') as f:
            answer_str = f.read()
        answer_num = float(answer_str) # Convert read string to float for comparison
        # print("Expected:", answer_num) # For debugging
        # print("Got:", result_num)      # For debugging
        # Using math.isclose for float comparison is generally better,
        # but here direct comparison is used as per original code.
        if result_num == answer_num:
            flag = True
        else:
            flag = False

    elif (out_type == "ee.Dictionary" or out_type == "ee.Reducer" or out_type == "ee.Blob"
          or out_type == "ee.Filter" or out_type == "ee.Classifier" or out_type == "ee.ErrorMargin"
          or out_type == "ee.Clusterer" or out_type == "ee.Kernel"
          or out_type == "ee.PixelType" or out_type == "ee.Join"):
        # Ensure file extension is .txt (for JSON storage)
        if not result_path.endswith('.txt'):
            result_path = f"{result_path}.txt"
        if not answer_path.endswith('.txt'):
            answer_path = f"{answer_path}.txt"

        result_dict = result.getInfo() # GEE object to Python dictionary

        # Save as TXT file (JSON formatted)
        with open(result_path, encoding='utf-8', mode='w') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2) # Save with indentation

        # Check answer
        with open(answer_path, encoding='utf-8', mode='r') as f:
            answer_dict = json.load(f)
        # print("Expected:", answer_dict) # For debugging
        # print("Got:", result_dict)      # For debugging

        # Ignore the 'updateTime' field, its value is related to program runtime and not a deterministic element in the output
        if 'updateTime' in result_dict:
            del result_dict['updateTime']
        if 'updateTime' in answer_dict:
            del answer_dict['updateTime']

        if result_dict == answer_dict: # Direct dictionary comparison
            flag = True
        else:
            flag = False

    elif out_type == "ee.Projection":
        # Ensure file extension is .txt
        if not result_path.endswith('.txt'):
            result_path = f"{result_path}.txt"
        if not answer_path.endswith('.txt'):
            answer_path = f"{answer_path}.txt"

        result_proj_info = result.getInfo() # Get projection info (usually a dict)
        result_proj_str = str(result_proj_info) # Convert to string for saving

        # Save as TXT file
        with open(result_path, encoding='utf-8', mode='w') as f:
            f.write(result_proj_str)

        # Check answer
        with open(answer_path, encoding='utf-8', mode='r') as f:
            answer_proj_str = f.read()
        # print("Expected:", answer_proj_str) # For debugging
        # print("Got:", result_proj_str)      # For debugging
        if result_proj_str == answer_proj_str:
            flag = True
        else:
            flag = False

    elif out_type == "ee.Date":
        # Ensure file extension is .txt
        if not result_path.endswith('.txt'):
            result_path = f"{result_path}.txt"
        if not answer_path.endswith('.txt'):
            answer_path = f"{answer_path}.txt"

        result_dict = result.getInfo() # GEE Date to dict {'type': 'Date', 'value': milliseconds_timestamp}
        result_date_ms = int(result_dict['value']) # Extract timestamp

        # Save as TXT file (timestamp as string)
        with open(result_path, encoding='utf-8', mode='w') as f:
            f.write(str(result_date_ms))

        # Check answer
        with open(answer_path, encoding='utf-8', mode='r') as f:
            answer_date_ms = int(f.read())
        # print("Expected:", answer_date_ms) # For debugging
        # print("Got:", result_date_ms)      # For debugging
        if result_date_ms == answer_date_ms:
            flag = True
        else:
            flag = False

    elif out_type == "ee.DateRange":
        # Ensure file extension is .txt
        if not result_path.endswith('.txt'):
            result_path = f"{result_path}.txt"
        if not answer_path.endswith('.txt'):
            answer_path = f"{answer_path}.txt"

        if result.isUnbounded().getInfo(): # Handle unbounded date ranges
            result_dates_ms = [0] # Placeholder or specific value for unbounded
        else:
            # GEE DateRange.getInfo() returns a dict like {'type': 'DateRange', 'dates': [start_ms, end_ms]}
            result_dict_info = result.getInfo()
            result_dates_ms = result_dict_info["dates"]

        # Save as TXT file (JSON list of timestamps)
        with open(result_path, encoding='utf-8', mode='w') as f:
            json.dump(result_dates_ms, f)

        # Check answer
        with open(answer_path, encoding='utf-8', mode='r') as f:
            answer_dates_ms = json.load(f)
        # print("Expected:", answer_dates_ms) # For debugging
        # print("Got:", result_dates_ms)      # For debugging
        if result_dates_ms == answer_dates_ms:
            flag = True
        else:
            flag = False

    elif out_type == "ee.List":
        # Ensure file extension is .txt (for JSON storage)
        if not result_path.endswith('.txt'):
            result_path = f"{result_path}.txt"
        if not answer_path.endswith('.txt'):
            answer_path = f"{answer_path}.txt"

        result_list = result.getInfo() # GEE List to Python list

        # Save as TXT file (JSON formatted list)
        with open(result_path, encoding='utf-8', mode='w') as f:
            json.dump(result_list, f, ensure_ascii=False, indent=2)

        # Check answer
        with open(answer_path, encoding='utf-8', mode='r') as f:
            answer_list = json.load(f)
        # print("Expected:", answer_list) # For debugging
        # print("Got:", result_list)      # For debugging
        if result_list == answer_list:
            flag = True
        else:
            flag = False

    elif out_type == "ee.Array" or out_type == "ee.ConfusionMatrix":
        # Ensure file extension is .npy
        if not result_path.endswith('.npy'):
            result_path = f"{result_path}.npy"
        if not answer_path.endswith('.npy'):
            answer_path = f"{answer_path}.npy"

        result_array = np.array(result.getInfo()) # GEE Array/ConfusionMatrix to NumPy array

        # Save as NPY file
        np.save(result_path, result_array)

        # Check answer
        answer_array = np.load(answer_path, allow_pickle=True)
        # print("Expected:", answer_array) # For debugging
        # print("Got:", result_array)      # For debugging
        try:
            np.testing.assert_array_almost_equal(result_array, answer_array, decimal=3)
            flag = True
        except AssertionError as e:
            flag = False
            message = e
        except Exception as e:
            flag = False
            message = e

    elif out_type == "ee.Element": # Generic GEE Element
        # Ensure file extension is .txt
        if not result_path.endswith('.txt'):
            result_path = f"{result_path}.txt"
        if not answer_path.endswith('.txt'):
            answer_path = f"{answer_path}.txt"

        result_info_str = str(result.getInfo()) # Get info and convert to string

        # Save as TXT file
        with open(result_path, encoding='utf-8', mode='w') as f:
            f.write(result_info_str)

        # Check answer
        with open(answer_path, encoding='utf-8', mode='r') as f:
            answer_str = f.read()
        # print("Expected:", answer_str) # For debugging
        # print("Got:", result_info_str) # For debugging
        if result_info_str == answer_str:
            flag = True
        else:
            flag = False

    elif out_type == "int" or out_type == "float": # Native Python types
        # Ensure file extension is .txt
        if not result_path.endswith('.txt'):
            result_path = f"{result_path}.txt"
        if not answer_path.endswith('.txt'):
            answer_path = f"{answer_path}.txt"

        # Save as TXT file
        with open(result_path, encoding='utf-8', mode='w') as f:
            f.write(str(result)) # Result is already a Python number

        # Check answer
        with open(answer_path, encoding='utf-8', mode='r') as f:
            answer_num_str = f.read()
        # Convert based on original type for more precise comparison if needed,
        # but float conversion is used here as per original.
        answer_num = float(answer_num_str)
        # print("Expected:", answer_num) # For debugging
        # print("Got:", result)         # For debugging
        if result == answer_num:
            flag = True
        else:
            flag = False

    elif out_type == "bool" or out_type == "str": # Native Python types
        # Ensure file extension is .txt
        if not result_path.endswith('.txt'):
            result_path = f"{result_path}.txt"
        if not answer_path.endswith('.txt'):
            answer_path = f"{answer_path}.txt"

        # Save as TXT file
        result_str_native = str(result) # Result is already a Python bool/str
        with open(result_path, encoding='utf-8', mode='w') as f:
            f.write(result_str_native)

        # Check answer
        with open(answer_path, encoding='utf-8', mode='r') as f:
            answer_str_file = f.read()
        # print("Expected:", answer_str_file) # For debugging
        # print("Got:", result_str_native)    # For debugging
        if result_str_native == answer_str_file:
            flag = True
        else:
            flag = False

    elif out_type == "list": # Native Python list
        # Ensure file extension is .txt (for JSON storage)
        if not result_path.endswith('.txt'):
            result_path = f"{result_path}.txt"
        if not answer_path.endswith('.txt'):
            answer_path = f"{answer_path}.txt"

        # Save as TXT file (JSON formatted list)
        with open(result_path, encoding='utf-8', mode='w') as f:
            json.dump(result, f, ensure_ascii=False, indent=2) # Result is already a Python list

        # Check answer
        with open(answer_path, encoding='utf-8', mode='r') as f:
            answer_list_file = json.load(f)
        # print("Expected:", answer_list_file) # For debugging
        # print("Got:", result)               # For debugging
        if result == answer_list_file:
            flag = True
        else:
            flag = False

    elif out_type == "dict": # Native Python dict
        # Ensure file extension is .txt (for JSON storage)
        if not result_path.endswith('.txt'):
            result_path = f"{result_path}.txt"
        if not answer_path.endswith('.txt'):
            answer_path = f"{answer_path}.txt"

        # Save as TXT file (JSON formatted dict)
        with open(result_path, encoding='utf-8', mode='w') as f:
            json.dump(result, f, ensure_ascii=False, indent=2) # Result is already a Python dict

        # Check answer
        with open(answer_path, encoding='utf-8', mode='r') as f:
            answer_dict_file = json.load(f)
        # print("Expected:", answer_dict_file) # For debugging
        # print("Got:", result)                # For debugging

        # Ignore 'updateTime' field if present in native dicts as well
        if 'updateTime' in result:
            del result['updateTime']
        if 'updateTime' in answer_dict_file:
            del answer_dict_file['updateTime']

        if result == answer_dict_file:
            flag = True
        else:
            flag = False

    else:
        print(f"Unsupported output type: {out_type}")
        return False, f"Unsupported output type: {out_type}"

    # Construct error message if check failed
    error_detail = str(message) if message else "Result mismatch"
    return flag, "Error when checking: " + error_detail if not flag else None


def check_model_result(model_names):
    """
    Runs tests for a list of specified model names.
    Assumes a directory structure for generated results and datasets.

    :param model_names: A list of strings, where each string is a model name.
    """
    for model_name in model_names:
        print(f"Starting test for {model_name}...")
        run_code_from_txt(f"./generate_results/{model_name}/atomic", r"./dataset_complete/atomic_code/atomic_test_config.yaml",
                          f"./generate_results/{model_name}/atomic_output", "atomic", reference_directory="./dataset_complete")
        print(f"{model_name} test completed!\n")


if __name__ == '__main__':
    # input("This program involves GEE operations. Please confirm your network environment and press Enter to continue!")
    geemap.set_proxy(port=7890) # Set proxy for geemap if needed
    ee.Authenticate() # Authenticate with Earth Engine
    ee.Initialize(project=GEE_PROJECT_NAME) # Initialize Earth Engine with a project, remember to set the project name
    models = [
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
    check_model_result(models)

    print(1)
