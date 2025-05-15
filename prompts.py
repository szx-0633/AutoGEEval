CONSTRUCT_ATOMIC = (
    '''
    ## Task Description
    You need to generate standard test code and configuration file entries for a given Google Earth Engine (GEE) Python API operator.
    Each operator will have two parts: the standard code and the test cases in the configuration file.

    ### Input
    1. **Operator Name**: Name of the operator
    2. **Explanation**: The explanation of the operator about what it does
    3. **Parameter List**: List of parameters with their types and descriptions. For example, `image` (ee.Image): The input image
    4. **Return Type**: The return type of the operator

    ### Output
    1. **Standard Code**: Define a function that uses the given operator and returns the result.
    The function name should be (Data Type+ operator name + Task). For example, `ee.Image.NormalizedDifference`->`imageNormalizedDifferenceTask`.
    2. **Test Cases in Configuration File**: Include multiple test cases, each with parameters, expected answer path, and output type.

    ### GEE objects in params
    1.If the parameter is an GEE object(e.g. ee.Image, ee.Number, etc), use the following format in the configuration file to return the object with python:
    param_name: !python |
        def get_ee_object():
            import ee
            ee.Initialize()
            # then get and return the wanted object
    2.Notice that some operators may require specific GEE objects as input. e.g. 'ee.Array.CholoskyDecomposition' requires a positive definite ee.Array matrix.
    
    ### Output Type
    1. The output type can be one of the following:
    GEE objects:
    "ee.Image", "ee.FeatureCollection", "ee.Number", "ee.List", "ee.Dictionary", "ee.Geometry", "ee.Array", "ee.ImageArray"
    Python objects:
    "str", "int", "float", "bool", "list", "dict", "NoneType"
    2. You can use other types if needed.
    
    ### Expected answer
    1. The value of the "expected_answer" field in the configuration file MUST be the path to the file containing the expected output.
    2. The file name should be (function name + "_testcase" + testcase_number), file type should be .npy for images and arrays,
     .geojson for geometry or feature objects, .txt for other types.
    
    ### Example
    #### Example Input
    - **Operator Name**: `normalizedDifference`
    - **Function Explanation**: Compute the normalized difference between the given two bands
    - **Parameter List**:
      - `image` (ee.Image): The input image
      - `band1` (str): The name of the first band
      - `band2` (str): The name of the second band
    - **Return Type**: `ee.Image`

    #### Example Output
    ##### Standard Code
    ```python
    def imageCannyEdgeDetectorTask(image: ee.Image, threshold: float, sigma: float = 1.0) -> ee.Image:
    """Applies the Canny edge detection algorithm to an image. """
        canny_edge = ee.Algorithms.CannyEdgeDetector(image, threshold, sigma)
        return canny_edge
    ```
    ##### Test Cases
    ```yaml
    imageCannyEdgeDetectorTask:
    - params:
        image: !python |
          def get_image():
            import ee
            ee.Initialize()
            dataset = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                           .filterBounds(ee.Geometry.Point([120, 30]))
                           .filterDate('2024-01-01', '2024-12-31'))
            img = dataset.first()
            region = ee.Geometry.Rectangle([120.05, 30.05, 120.1, 30.1])
            clipped_img = img.clip(region)
            return clipped_img
        sigma: 1.5
        threshold: 0.3
      expected_answer: imageCannyEdgeDetectorTask_testcase1.npy
      out_type: ee.Image
    ```
    
    ### Note
    1. The function should just include ONE operator and return the result. They are used for automatic testing.
    2. If the output is a GEE object, do NOT perform getInfo() function. Just return the object.
    3. Use the given operator for your answer, do NOT use other methods or operators to solve the task.
    4. Any import statements, initialization statements or example usages are NOT needed.
    5. Do NOT add any explanation.

    ### Operator Information
    Here is the operator information:
    
    ''')
