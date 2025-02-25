from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import uuid
import base64
import os
import logging
import cv2
import glob
import time
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"]  
)

logging.basicConfig(level=logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

CACHED_IMAGE_PATH = None 

class GenerateVideoRequest(BaseModel):
    input: dict

def wait_for_file(file_path, timeout=10):
    """Waits for a file to be fully written before serving it."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 1000:  # Ensure non-empty file
            return True
        time.sleep(0.5) 
    return False

@app.post("/generate-video/")
async def generate_video(request: GenerateVideoRequest):
    global CACHED_IMAGE_PATH

    try:
        input_data = request.input
        audio_data_base64 = input_data.get("file")
        new_image_data_base64 = input_data.get("image")

        if not audio_data_base64:
            return JSONResponse(content={"error": "Audio is required"}, status_code=400)

        temp_uuid = str(uuid.uuid4())
        audio_path = f"/dev/shm/{temp_uuid}.wav"
        output_video_path = f"/dev/shm/{temp_uuid}.mp4"

        with open(audio_path, "wb") as audio_file:
            audio_file.write(base64.b64decode(audio_data_base64))

        if new_image_data_base64:
            image_path = "/dev/shm/cached_image.jpg"
            with open(image_path, "wb") as image_file:
                image_file.write(base64.b64decode(new_image_data_base64))

            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"Failed to load image at {image_path}")
                return JSONResponse(content={"error": "Invalid image file"}, status_code=400)
            
            CACHED_IMAGE_PATH = image_path
            logging.info("Updated cached image.")

        if not CACHED_IMAGE_PATH:
            return JSONResponse(content={"error": "No image available. Please upload one."}, status_code=400)

        command = [
            "python", "inference.py",
            "--driven_audio", audio_path,
            "--source_image", CACHED_IMAGE_PATH,
            "--result_dir", "/dev/shm",
            "--preprocess", "full",
            "--output_path", output_video_path 
        ]
        logging.info(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True)

        if not wait_for_file(output_video_path):
            logging.error("Output video file was not properly generated.")
            return JSONResponse(content={"error": "Video file is invalid or missing."}, status_code=500)

        file_size = os.path.getsize(output_video_path)
        logging.info(f"Serving video file: {output_video_path} ({file_size} bytes)")

        response = FileResponse(
            output_video_path,
            media_type="video/mp4",
            filename="generated_video.mp4"
        )

        def cleanup():
            for file_path in [audio_path, output_video_path]:
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        app.add_event_handler("shutdown", cleanup)  

        return response

    except subprocess.CalledProcessError as e:
        logging.error(f"SadTalker failed: {e}")
        return JSONResponse(content={"error": f"SadTalker failed: {str(e)}"}, status_code=500)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
