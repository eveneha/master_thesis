import onnxruntime as ort 
import numpy as np 
import time 

quantized_model_path = "tcn_quantized_model_torch_export.onnx"

# Example: Create some dummy input data (replace with your actual data loading)
# Check your model's expected input shape and type!
# Quantized models often still accept FP32 inputs.
input_shape = (1, 1, 1000,1)  # Example for an image model (batch, channels, height, width)
input_type = np.float32
dummy_input_data = np.random.randn(*input_shape).astype(input_type)

# --- Load the ONNX Runtime Session ---
print(f"Loading quantized model from: {quantized_model_path}")

# Standard session options
sess_options = ort.SessionOptions()

# Optional: Enable ORT optimizations (often on by default for CPU)
# sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Optional: Explicitly set Execution Provider (CPU is default and supports INT8 well)
# providers = ['CPUExecutionProvider'] # Or ['CUDAExecutionProvider'], ['TensorrtExecutionProvider'] if applicable and model is compatible

try:
    # Create the inference session
    session = ort.InferenceSession(quantized_model_path, sess_options) #, providers=providers)
    print("ONNX Runtime session created successfully.")
    print(f"Using Execution Provider(s): {session.get_providers()}")

    # Get model input details
    input_name = session.get_inputs()[0].name
    input_shape_model = session.get_inputs()[0].shape
    input_type_model = session.get_inputs()[0].type

    print(f"Model expects input name: '{input_name}', shape: {input_shape_model}, type: {input_type_model}")
    # Simple check - might need adjustment based on dynamic axes ('None')
    # assert list(input_shape) == input_shape_model, f"Input shape mismatch: Provided {input_shape}, Model expects {input_shape_model}"
    # assert np.dtype(input_type).name in input_type_model, f"Input type mismatch: Provided {input_type}, Model expects {input_type_model}"


    # Get model output details
    output_names = [output.name for output in session.get_outputs()]
    print(f"Model output name(s): {output_names}")

    # --- Run Inference ---
    print("Running inference...")
    start_time = time.perf_counter()

    # Prepare inputs as a dictionary
    inputs = {input_name: dummy_input_data}

    # Run the model
    outputs = session.run(output_names, inputs) # Pass None for output_names to get all outputs

    end_time = time.perf_counter()
    print(f"Inference completed in {end_time - start_time:.4f} seconds.")

    # --- Process Outputs ---
    print(f"Received {len(outputs)} output(s).")
    # Example: Print shape of the first output
    if outputs:
        print(f"Shape of first output ('{output_names[0]}'): {outputs[0].shape}")
        # Add your post-processing logic here
        # E.g., apply softmax, find max index, etc.

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()