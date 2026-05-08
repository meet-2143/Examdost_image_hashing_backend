from fastapi import FastAPI, UploadFile, File
from PIL import Image
import imagehash
import cv2
import numpy as np
import clip
import torch
import io

app = FastAPI()

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def preprocess_image(image_bytes: bytes) -> Image.Image:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    inverted = cv2.bitwise_not(binary)
    coords = cv2.findNonZero(inverted)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        padding = 15
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2*padding)
        h = min(img.shape[0] - y, h + 2*padding)
        cropped = binary[y:y+h, x:x+w]
    else:
        cropped = binary
    resized = cv2.resize(cropped, (256, 256))
    return Image.fromarray(resized)


@app.post("/phash")
async def get_phash(file: UploadFile = File(...)):
    image_bytes = await file.read()
    preprocessed = preprocess_image(image_bytes)
    ph = imagehash.phash(preprocessed, hash_size=16)
    return {
        "phash_hex": str(ph),
        "phash_binary": bin(int(str(ph), 16))[2:].zfill(256)
    }


@app.post("/clip")
async def get_clip_embedding(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    embedding_list = embedding.cpu().numpy().flatten().tolist()
    return {
        "embedding": embedding_list,
        "dims": len(embedding_list)
    }


@app.post("/match")
async def full_match(file: UploadFile = File(...)):
    image_bytes = await file.read()
    preprocessed = preprocess_image(image_bytes)
    ph = imagehash.phash(preprocessed, hash_size=16)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    embedding_list = embedding.cpu().numpy().flatten().tolist()
    return {
        "phash_hex": str(ph),
        "phash_binary": bin(int(str(ph), 16))[2:].zfill(256),
        "clip_embedding": embedding_list
    }