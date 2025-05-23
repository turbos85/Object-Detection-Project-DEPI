# --------------------------------------------------------------------------
# app.py for Hugging Face Space: YOLOv8 Object Detection (BDD100K)
# --------------------------------------------------------------------------

# 1. Imports
import gradio as gr
from ultralytics import YOLO
import os
import cv2
import numpy as np
import torch
import time
import glob
from PIL import Image # PIL is often useful for image handling

# 2. Configuration & Setup
MODEL_PATH = "best_yolov8n_sz640_bdd100k_subset.pt" # Relative path within the HF repo
CONFIDENCE_THRESHOLD = 0.30 # Adjust detection confidence threshold
IOU_THRESHOLD = 0.45        # Adjust Non-Max Suppression IoU threshold

print("--- Initializing Detection Service ---")

# Determine compute device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the YOLOv8 model
model = None
model_load_error = None
if os.path.exists(MODEL_PATH):
    try:
        print(f"Loading model from: {MODEL_PATH}...")
        model = YOLO(MODEL_PATH)
        # Optional: Explicitly move model - YOLO might handle this, but good practice
        # model.to(device)
        print("Model loaded successfully.")
        # Perform a quick test inference to warm up / check compatibility
        _ = model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False, device=device)
        print("Model test inference successful.")
    except Exception as e:
        model_load_error = f"Error loading model: {e}"
        print(model_load_error)
else:
    model_load_error = f"Error: Model file not found at {MODEL_PATH}"
    print(model_load_error)

# Define the class names (ensure this order EXACTLY matches your training config)
class_names = [
    'car', 'truck', 'bus', 'motor', 'bike', 'person',
    'rider', 'traffic light', 'traffic sign'
]
if model and hasattr(model, 'names'):
     # Prefer model's names if available and they seem correct
     if isinstance(model.names, dict) and len(model.names) == len(class_names):
         model_class_names_list = [model.names[i] for i in sorted(model.names.keys())]
         # Basic check if names look plausible (optional)
         if all(isinstance(name, str) for name in model_class_names_list):
              print("Using class names defined within the model file.")
              class_names = model_class_names_list
         else:
              print("Warning: Model names attribute seems invalid, using hardcoded list.")
     else:
          print("Warning: Model names attribute mismatch or missing, using hardcoded list.")
print(f"Using class names: {class_names}")


# 3. Inference Functions
def detect_objects_image(input_image: np.ndarray):
    """
    Performs object detection on a single input image.
    Args: input_image (np.ndarray): Input image (NumPy BGR).
    Returns: np.ndarray: Annotated image (NumPy RGB).
    """
    if model is None or input_image is None:
        print("Image Detection Error: Model not loaded or no input image.")
        # Return a placeholder or original image if input exists
        return cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB) if input_image is not None else None

    print(f"Processing image with shape: {input_image.shape}")
    start_time = time.time()

    try:
        # Run prediction (model expects BGR for numpy arrays by default)
        results = model.predict(
            source=input_image,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=device,
            verbose=False # Keep console clean
        )

        if not results or results[0].boxes is None:
             print("No objects detected.")
             annotated_image_bgr = input_image # Return original if no boxes
        else:
            # Use the plot() method to get annotated image (returns BGR)
            annotated_image_bgr = results[0].plot(conf=True) # Show conf scores on boxes

        # Convert BGR to RGB for Gradio display
        annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)

        end_time = time.time()
        print(f"Image processing finished in {end_time - start_time:.2f} seconds.")
        return annotated_image_rgb

    except Exception as e:
        print(f"An error occurred during image prediction: {e}")
        # Return original image in RGB format in case of error
        return cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB) if input_image is not None else None


def detect_objects_video(input_video_path: str):
    """
    Performs object detection on an input video file. Saves the output.
    Args: input_video_path (str): Path to the input video.
    Returns: str or None: Path to the processed video or None on failure.
    """
    if model is None or not input_video_path or not os.path.exists(input_video_path):
        print(f"Video Detection Error: Model not loaded or invalid video path: {input_video_path}")
        return None

    print(f"Processing video file: {input_video_path}")
    start_time = time.time()
    output_video_path = None
    last_reported_path = None

    try:
        # Predict and save the video directly
        # Note: Filename might be slightly altered if it has tricky characters.
        results_generator = model.predict(
            source=input_video_path,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=device,
            save=True,       # Save the output video
            stream=True,     # Process as stream for efficiency
            verbose=False    # Keep console clean
        )

        print("Starting video frame processing...")
        processed_frames = 0
        for i, results in enumerate(results_generator): # Must iterate to process
            processed_frames += 1
            if hasattr(results, 'save_dir') and results.save_dir:
                last_reported_path = results.save_dir # Keep track of where YOLO is saving

        print(f"Finished processing {processed_frames} frames.")
        end_time = time.time()

        # --- Find the saved video file ---
        if last_reported_path:
             input_filename = os.path.basename(input_video_path)
             # Common extensions YOLO might save as
             possible_extensions = ['.mp4', '.avi', '.mov', '.mkv']
             # Check original filename first
             potential_path = os.path.join(last_reported_path, input_filename)
             if os.path.exists(potential_path):
                 output_video_path = potential_path
             else:
                 # If exact match fails, search for *any* video file in the save dir
                 print(f"Warning: Expected output file '{input_filename}' not found in '{last_reported_path}'. Searching directory...")
                 found_files = []
                 for ext in possible_extensions:
                     found_files.extend(glob.glob(os.path.join(last_reported_path, f"*{ext}")))

                 if found_files:
                     # Sort by modification time? Or just take the first? Assume first for now.
                     output_video_path = found_files[0]
                     print(f"Using detected output file: {output_video_path}")
                 else:
                     print(f"Error: No video files found in the save directory: {last_reported_path}")
        else:
             print("Error: Could not determine save directory from YOLO results.")

        # --- Final Check ---
        if output_video_path and os.path.exists(output_video_path):
             print(f"Video processing finished in {end_time - start_time:.2f} seconds.")
             return output_video_path
        else:
             print(f"Error: Video processing completed, but failed to find the output video file.")
             return None

    except Exception as e:
        import traceback
        print(f"FATAL ERROR during video prediction: {e}")
        print(traceback.format_exc())
        return None


# Wrapper for Gradio video output with status updates
def video_wrapper(video_path):
    if not video_path: return None, "No video file provided."
    status = f"Processing: {os.path.basename(video_path)}... (This may take time)"
    # Return status immediately while processing starts
    yield None, status # Yield initial status update
    result_path = detect_objects_video(video_path)
    if result_path:
        final_status = f"Processing complete. Output video ready."
        yield result_path, final_status
    else:
        final_status = "Video processing failed. Check logs or try again."
        yield None, final_status


# 4. Gradio Interface Definition
interface_title = "Autonomous Vehicle Object Detection: YOLOv8n on BDD100K" # Choose your title

interface_description = """
Detect common objects in driving scenarios using a YOLOv8n model fine-tuned on the BDD100K dataset.
Upload an image or video file. Detected objects (cars, trucks, buses, motorcycles, bicycles, pedestrians, riders, traffic lights, traffic signs) will be highlighted with bounding boxes.
Processing video can take significant time depending on length and hardware (runs on CPU by default on free tier Spaces).
"""

with gr.Blocks(theme=gr.themes.Glass(), title=interface_title) as demo:
    gr.Markdown(f"# {interface_title}")
    gr.Markdown(interface_description)

    with gr.Tabs():
        # --- Image Detection Tab ---
        with gr.TabItem("Image Detection"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="numpy", label="Upload Image")
                    image_submit_btn = gr.Button("Detect Objects in Image", variant="primary")
                with gr.Column(scale=1):
                    image_output = gr.Image(type="numpy", label="Detection Results")

            image_submit_btn.click(
                fn=detect_objects_image,
                inputs=image_input,
                outputs=image_output
            )
            gr.Examples( # Provide accessible URLs for examples
                examples=[
                    ["https://ultralytics.com/images/bus.jpg"],
                    ["https://ultralytics.com/images/zidane.jpg"]
                ],
                inputs=image_input,
                outputs=image_output,
                fn=detect_objects_image,
                cache_examples=True # Cache results for faster example loading
            )

        # --- Video Detection Tab ---
        with gr.TabItem("Video Detection"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Upload Video")
                    video_submit_btn = gr.Button("Detect Objects in Video", variant="primary")
                with gr.Column(scale=1):
                    video_output = gr.Video(label="Detection Results")
                    video_status = gr.Textbox(label="Status", interactive=False)

            video_submit_btn.click(
                fn=video_wrapper, # Use the wrapper
                inputs=video_input,
                outputs=[video_output, video_status]
            )
            # Optional: Add video examples if you have short, accessible URLs
            # gr.Examples(...)


# 5. Launch the App
if __name__ == "__main__":
    if model is None:
        print("FATAL: Model failed to load. Launching error interface.")
        # Fallback interface showing the error
        with gr.Blocks(theme=gr.themes.Base()) as error_demo:
             gr.Markdown(f"""
             # Application Error
             **Failed to load the YOLOv8 model.**
             Check the Space logs for details. Error message:
             ```
             {model_load_error}
             ```
             Model path expected: `{MODEL_PATH}`
             """)
        error_demo.launch()
    else:
        print("Launching Gradio application...")
        demo.launch() # Default launch for HF Spaces (no server_name/port needed)