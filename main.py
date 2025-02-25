from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import uuid
import io
import base64
import os
import logging
import cv2
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

CACHED_IMAGE_PATH = None  # Stores the most recent image path

class GenerateVideoRequest(BaseModel):
    input: dict

@app.post("/generate-video/")
async def generate_video(request: GenerateVideoRequest):
    global CACHED_IMAGE_PATH

    try:
        input_data = request.input
        audio_data_base64 = input_data.get("file")
        new_image_data_base64 = input_data.get("image")

        if not audio_data_base64:
            return JSONResponse(content={"error": "Audio is required"}, status_code=400)

        # Decode and save the audio
        audio_data = base64.b64decode(audio_data_base64)
        temp_uuid = str(uuid.uuid4())
        audio_path = f"/dev/shm/{temp_uuid}.wav"
        output_video_path = f"/dev/shm/{temp_uuid}.mp4"

        with open(audio_path, "wb") as audio_file:
            audio_file.write(audio_data)

        # Check if a new image is provided
        if new_image_data_base64:
            image_data = base64.b64decode(new_image_data_base64)
            image_path = f"/dev/shm/cached_image.jpg"
            with open(image_path, "wb") as image_file:
                image_file.write(image_data)

            # Validate image
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"Failed to load image at {image_path}")
                return JSONResponse(content={"error": "Invalid image file"}, status_code=400)

            CACHED_IMAGE_PATH = image_path  # Update the cached image
            logging.info("Updated cached image.")

        # Use the most recent cached image if no new image is provided
        if not CACHED_IMAGE_PATH:
            return JSONResponse(content={"error": "No image available. Please upload one."}, status_code=400)

        # Run the video generation script
        command = [
            "python", "inference.py",
            "--driven_audio", audio_path,
            "--source_image", CACHED_IMAGE_PATH,
            "--result_dir", "/dev/shm",
            "--preprocess", "full",
            "--enhancer", "gfpgan"
        ]
        logging.info(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True)

        # Find the generated video file (handling possible filename variations)
        generated_files = [f for f in os.listdir("/dev/shm") if f.endswith(".mp4")]
        if not generated_files:
            return JSONResponse(content={"error": "Generated video not found."}, status_code=500)

        latest_video_path = max(generated_files, key=lambda f: os.path.getctime(os.path.join("/dev/shm", f)))
        video_full_path = os.path.join("/dev/shm", latest_video_path)

        # Read video for response
        with open(video_full_path, "rb") as video_file:
            video_blob = io.BytesIO(video_file.read())
            video_blob.seek(0)

        # Cleanup temporary files
        for file_path in [audio_path, video_full_path]:
            if os.path.exists(file_path):
                os.remove(file_path)

        return StreamingResponse(video_blob, media_type="video/mp4")

    except subprocess.CalledProcessError as e:
        logging.error(f"SadTalker failed: {e}")
        return JSONResponse(content={"error": f"SadTalker failed: {str(e)}"}, status_code=500)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
