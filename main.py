from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import subprocess
import uuid
import base64
import os
import logging
import cv2
import glob
import aiofiles
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

logging.basicConfig(level=logging.INFO)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CACHED_IMAGE = None  

async def convert_to_h264(input_path, output_path):
    """Converts the video to H.264 format."""
    try:
        command = [
            "ffmpeg", "-y", "-i", input_path,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            output_path
        ]
        process = await asyncio.create_subprocess_exec(*command)
        await process.communicate()
        return output_path if process.returncode == 0 else None
    except Exception as e:
        logging.error(f"FFmpeg error: {e}")
        return None

@app.websocket("/generate-video/")
async def websocket_generate_video(websocket: WebSocket):
    await websocket.accept()
    global CACHED_IMAGE

    try:
        data = await websocket.receive_json()
        audio_data_base64 = data.get("file")
        new_image_data_base64 = data.get("image")

        if not audio_data_base64:
            await websocket.send_json({"error": "Audio is required"})
            await websocket.close()
            return

        temp_uuid = str(uuid.uuid4())
        audio_path = f"/dev/shm/{temp_uuid}.wav"

        async with aiofiles.open(audio_path, "wb") as audio_file:
            await audio_file.write(base64.b64decode(audio_data_base64))

        if new_image_data_base64:
            image_data = base64.b64decode(new_image_data_base64)
            image_array = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)

            if image_array is None:
                await websocket.send_json({"error": "Invalid image file"})
                await websocket.close()
                return

            CACHED_IMAGE = image_array
            logging.info("Updated cached image.")

        if CACHED_IMAGE is None:
            await websocket.send_json({"error": "No image available. Please upload one."})
            await websocket.close()
            return

        await websocket.send_json({"status": "Processing started"})

        command = [
            "python", "inference.py",
            "--driven_audio", audio_path,
            "--source_image", "/dev/shm/cached_image.jpg",
            "--result_dir", "/dev/shm",
            "--preprocess", "full",
           
        ]
        logging.info(f"Running command: {' '.join(command)}")

        process = await asyncio.create_subprocess_exec(*command)
        await process.communicate()

        if process.returncode != 0:
            await websocket.send_json({"error": "Video generation failed"})
            await websocket.close()
            return

        video_files = sorted(glob.glob("/dev/shm/**/*.mp4", recursive=True), key=os.path.getctime, reverse=True)

        if not video_files:
            await websocket.send_json({"error": "No video output found"})
            await websocket.close()
            return

        output_video_path = video_files[0]
        await websocket.send_json({"status": "Video generated, converting to H.264"})

        converted_video_path = f"/dev/shm/{temp_uuid}_h264.mp4"
        converted_path = await convert_to_h264(output_video_path, converted_video_path)

        if not converted_path:
            await websocket.send_json({"error": "Failed to convert video to H.264"})
            await websocket.close()
            return

        await websocket.send_json({
            "status": "Completed",
            "video_url": f"https://p8qwqlzty95gao-7888.proxy.runpod.net/videos/{temp_uuid}_h264.mp4"
        })

        async def cleanup():
            for file_path in [audio_path, output_video_path, converted_path]:
                if os.path.exists(file_path):
                    os.remove(file_path)

        asyncio.create_task(cleanup())

        await websocket.close()

    except Exception as e:
        logging.error(f"Error: {e}")
        await websocket.send_json({"error": str(e)})
        await websocket.close()
