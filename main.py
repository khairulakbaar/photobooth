from fastapi import FastAPI, UploadFile, File, Form, HTTPException 
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional
from datetime import datetime
from insightface.app import FaceAnalysis
import os
import cv2
import insightface

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)
UPLOAD_DIR = "static/uploads"
TEMPLATE_DIR = "static/templates"
os.makedirs(UPLOAD_DIR, exist_ok=True)
def swap_faces(src_path: str, dst_path: str, output_path: str):
    img1 = cv2.imread(src_path)
    img2 = cv2.imread(dst_path)
    
    faces1 = face_app.get(img1)
    faces2 = face_app.get(img2)

    if len(faces1) == 0 or len(faces2) == 0:
        raise Exception("Wajah tidak terdeteksi di salah satu gambar.")

    face1 = faces1[0]
    face2 = faces2[0]

    result = swapper.get(img2.copy(), face2, face1, paste_back=True)
    cv2.imwrite(output_path, result)
    return output_path

def overlay_frame(base_image_path: str, frame_path: str, output_path: str):
    base_img = cv2.imread(base_image_path, cv2.IMREAD_UNCHANGED)
    frame_img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)

    # Resize frame ke ukuran base image
    frame_img = cv2.resize(frame_img, (base_img.shape[1], base_img.shape[0]))

    if frame_img.shape[2] == 4:  # PNG dengan alpha
        alpha_mask = frame_img[:, :, 3] / 255.0
        for c in range(3):
            base_img[:, :, c] = (1 - alpha_mask) * base_img[:, :, c] + alpha_mask * frame_img[:, :, c]
    else:
        base_img = cv2.addWeighted(base_img, 1, frame_img, 1, 0)

    cv2.imwrite(output_path, base_img)
    return output_path

@app.post("/api/swap")
async def swap_faces_api(
    template_name: str = Form(...),
    webcam: UploadFile = File(...),
    source: Optional[UploadFile] = File(None)
):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    src_filename = f"webcam_{timestamp}_{webcam.filename}"
    src_path = os.path.join(UPLOAD_DIR, src_filename)

    # Simpan file dari webcam (binary)
    with open(src_path, "wb") as f:
        f.write(await webcam.read())

    # Jika ada file source tambahan (opsional), timpa src_path
    if source:
        src_path = os.path.join(UPLOAD_DIR, source.filename)
        with open(src_path, "wb") as f:
            f.write(await source.read())

    # Cek apakah template_name valid
    template_path = os.path.join(TEMPLATE_DIR, template_name)
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Template tidak ditemukan.")

    result_filename = f"result_{timestamp}.png"
    result_path = os.path.join("static", result_filename)

    try:
        swap_faces(src_path, template_path, result_path)

        frame_path = os.path.join("static", "images", "frame1.png")
        overlay_frame(result_path, frame_path, result_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({
    "success": True,
    "message": "Face swap berhasil",
    "result_url": f"/static/{result_filename}"
})