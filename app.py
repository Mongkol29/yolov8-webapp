from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import os
import cv2
import numpy as np

app = FastAPI()

# Templates and Static
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/results", StaticFiles(directory="results"), name="results")

# Load YOLO model
model = YOLO("yolov8n.pt")

# Ensure result folders exist
os.makedirs("results/images", exist_ok=True)
os.makedirs("results/videos", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_image", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    results = model(img)
    annotated = results[0].plot()
    output_path = "results/images/result.jpg"
    cv2.imwrite(output_path, annotated)
    return templates.TemplateResponse("result.html", {
        "request": request,
        "image_path": "/results/images/result.jpg",
        "download_url": "/download/images/result.jpg"
    })

@app.post("/upload_video", response_class=HTMLResponse)
async def upload_video(request: Request, file: UploadFile = File(...)):
    video_path = "results/videos/uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return HTMLResponse("❌ ไม่สามารถเปิดไฟล์วิดีโอ", status_code=400)

    output_path = "results/videos/result_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0,
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated = results[0].plot()
        out.write(annotated)

    cap.release()
    out.release()

    return templates.TemplateResponse("result_video.html", {
        "request": request,
        "video_path": "/results/videos/result_video.mp4",
        "download_url": "/download/videos/result_video.mp4"
    })

@app.get("/download/{subfolder}/{filename}", response_class=FileResponse)
async def download_file(subfolder: str, filename: str):
    path = os.path.join("results", subfolder, filename)
    if os.path.exists(path):
        return FileResponse(path=path, filename=filename)
    return HTMLResponse(f"ไม่พบไฟล์ {filename}", status_code=404)