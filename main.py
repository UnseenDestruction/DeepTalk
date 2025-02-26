from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import uuid
import base64
import os
import logging
import glob
import asyncio
from pydantic import BaseModel
from pathlib import Path

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

async def run_inference(audio_path, image_path, output_dir):
    """Runs the inference script asynchronously."""
    command = [
        "python", "inference.py",
        "--driven_audio", audio_path,
        "--source_image", image_path,
        "--result_dir", output_dir,
        "--preprocess", "full",
        "--still",
    ]
    logging.info(f"Running: {' '.join(command)}")
    
    process = await asyncio.create_subprocess_exec(*command)
    await process.communicate()  # Wait for process completion

def get_latest_video(directory):
    """Gets the latest generated video file."""
    files = sorted(Path(directory).glob("*.mp4"), key=os.path.getctime, reverse=True)
    return str(files[0]) if files else None

@app.post("/generate-video/")
async def generate_video(request: GenerateVideoRequest):
    global CACHED_IMAGE_PATH
    input_data = request.input
    audio_data_base64 = input_data.get("file")
    new_image_data_base64 = input_data.get("image")
    temp_uuid = str(uuid.uuid4())
    audio_path = f"/dev/shm/{temp_uuid}.wav"
    output_dir = f"/dev/shm/{temp_uuid}"  
    os.makedirs(output_dir, exist_ok=True)

    # Write audio file
    with open(audio_path, "wb") as audio_file:
        audio_file.write(base64.b64decode(audio_data_base64))

    # Handle image caching
    if new_image_data_base64:
        CACHED_IMAGE_PATH = f"{output_dir}/cached_image.jpg"
        with open(CACHED_IMAGE_PATH, "wb") as image_file:
            image_file.write(base64.b64decode(new_image_data_base64))
        
    if not CACHED_IMAGE_PATH:
        return JSONResponse(content={"error": "No image available."}, status_code=400)
    
    # Run inference asynchronously
    await run_inference(audio_path, CACHED_IMAGE_PATH, output_dir)
    output_video = get_latest_video(output_dir)

    if not output_video:
        return JSONResponse(content={"error": "Failed to generate video."}, status_code=500)

    async def video_stream():
        with open(output_video, "rb") as f:
            while chunk := f.read(8192):  
                yield chunk
    
    return StreamingResponse(video_stream(), media_type="video/mp4")
