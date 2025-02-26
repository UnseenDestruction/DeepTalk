from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import uuid
import base64
import os
import glob
import asyncio
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CACHED_IMAGE_PATH = None  

class GenerateVideoRequest(BaseModel):
    input: dict

async def run_inference_async(audio_path, image_path, output_dir):
    command = [
        "python", "inference.py",
        "--driven_audio", audio_path,
        "--source_image", image_path,
        "--result_dir", output_dir,
        "--preprocess", "full",
        "--still"
    ]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    await process.communicate()

def get_latest_video(directory):
    files = sorted(glob.glob(f"{directory}/*.mp4"), key=os.path.getctime, reverse=True)
    return files[0] if files else None

@app.post("/generate-video/")
async def generate_video(request: GenerateVideoRequest, background_tasks: BackgroundTasks):
    global CACHED_IMAGE_PATH
    input_data = request.input
    audio_data_base64 = input_data.get("file")
    new_image_data_base64 = input_data.get("image")
    temp_uuid = str(uuid.uuid4())
    audio_path = f"/dev/shm/{temp_uuid}.wav"
    output_dir = f"/dev/shm/{temp_uuid}"  
    os.makedirs(output_dir, exist_ok=True)

    audio_bytes = base64.b64decode(audio_data_base64)
    with open(audio_path, "wb") as audio_file:
        audio_file.write(audio_bytes)

    if new_image_data_base64:
        CACHED_IMAGE_PATH = f"{output_dir}/cached_image.jpg"
        image_bytes = base64.b64decode(new_image_data_base64)
        with open(CACHED_IMAGE_PATH, "wb") as image_file:
            image_file.write(image_bytes)
        
    if not CACHED_IMAGE_PATH:
        return JSONResponse(content={"error": "No image available."}, status_code=400)
    
    start_time = time.time()
    
    # Run inference in background
    await run_inference_async(audio_path, CACHED_IMAGE_PATH, output_dir)

    output_video = get_latest_video(output_dir)
    if not output_video:
        return JSONResponse(content={"error": "Failed to generate video."}, status_code=500)

    response_time = time.time() - start_time
    print(f"Response Time: {response_time:.2f} seconds")  # Debugging

    async def video_stream():
        with open(output_video, "rb") as f:
            while chunk := f.read(8192):  
                yield chunk
    
    return StreamingResponse(video_stream(), media_type="video/mp4")
