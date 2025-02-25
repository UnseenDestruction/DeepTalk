from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import uuid
import io
import base64
import os
import logging
import cv2
import aiofiles
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
CHUNK_SIZE = 1024 * 1024  # 1MB per chunk for streaming

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
            image_path = f"/dev/shm/cached_image.jpg"
            with open(image_path, "wb") as image_file:
                image_file.write(image_data)

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
            "--enhancer", "gfpgan"
        ]
        logging.info(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True)

        generated_files = [f for f in os.listdir("/dev/shm") if f.endswith(".mp4")]
        if not generated_files:
            return JSONResponse(content={"error": "Generated video not found."}, status_code=500)

        latest_video_path = max(generated_files, key=lambda f: os.path.getctime(os.path.join("/dev/shm", f)))
        video_full_path = os.path.join("/dev/shm", latest_video_path)

        return JSONResponse(content={"video_url": f"/stream/{latest_video_path}"})

    except subprocess.CalledProcessError as e:
        logging.error(f"SadTalker failed: {e}")
        return JSONResponse(content={"error": f"SadTalker failed: {str(e)}"}, status_code=500)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


async def video_streamer(file_path: str, start: int, end: int):
    """Streams video file in chunks to reduce memory usage."""
    async with aiofiles.open(file_path, mode="rb") as file:
        await file.seek(start)
        remaining = end - start
        while remaining > 0:
            chunk = await file.read(min(CHUNK_SIZE, remaining))
            if not chunk:
                break
            yield chunk
            remaining -= len(chunk)

@app.get("/stream/{filename}")
async def stream_video(filename: str, request: Request):
    """Handles streaming of video files with partial content support."""
    file_path = f"/dev/shm/{filename}"

    if not os.path.exists(file_path):
        return Response(status_code=404, content="File not found.")

    file_size = os.path.getsize(file_path)
    range_header = request.headers.get("range")

    if range_header:
        # Handle "Range" requests for seeking
        range_value = range_header.replace("bytes=", "").strip()
        start, end = range_value.split("-")
        start = int(start) if start else 0
        end = int(end) if end else file_size - 1
        end = min(end, file_size - 1)

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(end - start + 1),
            "Content-Type": "video/mp4",
        }

        return StreamingResponse(video_streamer(file_path, start, end + 1), headers=headers, status_code=206)

    # Default: Stream full video
    headers = {
        "Content-Length": str(file_size),
        "Content-Type": "video/mp4",
    }
    return StreamingResponse(video_streamer(file_path, 0, file_size), headers=headers)
