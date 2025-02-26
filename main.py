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
import asyncio
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

async def run_subprocess(command):
    process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await process.communicate()
    logging.info(f"Stdout: {stdout.decode()}")
    logging.error(f"Stderr: {stderr.decode()}")

def wait_for_file(file_path, timeout=5):
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 1000:
            return True
        time.sleep(0.2)
    return False

def get_latest_video():
    video_files = sorted(glob.glob("/dev/shm/**/*.mp4", recursive=True), key=os.path.getctime, reverse=True)
    return video_files[0] if video_files else None

def convert_to_h264(input_path, output_path):
    try:
        command = [
            "ffmpeg", "-y", "-i", input_path,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", 
            "-c:a", "aac", "-b:a", "128k", 
            output_path  
        ]
        subprocess.run(command, check=True)
        logging.info(f"✅ Converted video to H.264: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ FFmpeg conversion failed: {e}")
        return None
    
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

        with open(audio_path, "wb") as audio_file:
            audio_file.write(base64.b64decode(audio_data_base64))

        if new_image_data_base64:
            image_path = f"/dev/shm/{temp_uuid}.jpg"
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
            "--facerender", "pirender",
            "--device", "mps",
            "--enhancer", "gfpgan"
        ]
        logging.info(f"Running command: {' '.join(command)}")
        await run_subprocess(command)

        video_output_path = get_latest_video()
        if not video_output_path or not wait_for_file(video_output_path):
            logging.error("Output video file was not properly generated.")
            return JSONResponse(content={"error": "Video generation failed"}, status_code=500)

        converted_video_path = f"/dev/shm/{temp_uuid}_h264.mp4"
        converted_path = convert_to_h264(video_output_path, converted_video_path)

        if not converted_path:
            return JSONResponse(content={"error": "Failed to convert video to H.264"}, status_code=500)

        file_size = os.path.getsize(converted_path)
        logging.info(f"Serving video file: {converted_path} ({file_size} bytes)")

        response = FileResponse(
            converted_path,
            media_type="video/mp4",
            filename="generated_video.mp4"
        )

        @response.call_on_close
        def cleanup():
            for file_path in [audio_path, video_output_path, converted_path]:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logging.info(f"Deleted: {file_path}")

        return response

    except subprocess.CalledProcessError as e:
        logging.error(f"SadTalker failed: {e}")
        return JSONResponse(content={"error": f"SadTalker failed: {str(e)}"}, status_code=500)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)