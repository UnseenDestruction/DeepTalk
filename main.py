from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import uuid
import io
import base64
import os
import logging
from pydantic import BaseModel
from PIL import Image
import cv2

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

@app.post("/generate-video/")
async def generate_video(request: GenerateVideoRequest):
    global CACHED_IMAGE_PATH

    try:
        input_data = request.input
        audio_data_base64 = input_data.get("file")
        new_image_data_base64 = input_data.get("image")

        if not audio_data_base64:
            return JSONResponse(content={"error": "Audio is required"}, status_code=400)

        audio_data = base64.b64decode(audio_data_base64)

        temp_uuid = str(uuid.uuid4())
        audio_path = f"/dev/shm/{temp_uuid}.wav"
        output_video_path = f"/dev/shm/{temp_uuid}.mp4"

        with open(audio_path, "wb") as audio_file:
            audio_file.write(audio_data)

        if new_image_data_base64:
            image_data = base64.b64decode(new_image_data_base64)
            image_path = "/dev/shm/cached_image.jpg"
            with open(image_path, "wb") as image_file:
                image_file.write(image_data)
            CACHED_IMAGE_PATH = image_path
            logging.info("Updated cached image.")

        if not CACHED_IMAGE_PATH or not os.path.exists(CACHED_IMAGE_PATH) or os.path.getsize(CACHED_IMAGE_PATH) == 0:
            return JSONResponse(content={"error": "Invalid or missing source image."}, status_code=400)

        # Validate image with OpenCV
        image_cv = cv2.imread(CACHED_IMAGE_PATH)
        if image_cv is None:
            return JSONResponse(content={"error": "Failed to load image with OpenCV."}, status_code=400)

        # Validate image with PIL
        try:
            img = Image.open(CACHED_IMAGE_PATH)
            img.verify()  # Check if it's a valid image file
        except Exception as e:
            return JSONResponse(content={"error": f"Invalid image format: {e}"}, status_code=400)

        logging.info("Image is valid and ready for processing.")

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

        with open(output_video_path, "rb") as video_file:
            video_blob = io.BytesIO(video_file.read())
            video_blob.seek(0)

        for file_path in [audio_path, output_video_path]:
            if os.path.exists(file_path):
                os.remove(file_path)

        return StreamingResponse(video_blob, media_type="video/mp4")

    except subprocess.CalledProcessError as e:
        logging.error(f"SadTalker failed: {e}")
        return JSONResponse(content={"error": f"SadTalker failed: {str(e)}"}, status_code=500)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)