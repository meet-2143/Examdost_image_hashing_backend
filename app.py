from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import Response
from PIL import Image, ImageOps
import imagehash
import clip
import torch
import io
import cairosvg

app = FastAPI()

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def preprocess_image(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = ImageOps.exif_transpose(img)
    img = img.resize((256, 256), Image.LANCZOS)
    return img


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


@app.post("/render")
async def render_svg(request: Request):
    svg_bytes = await request.body()

    if not svg_bytes:
        raise HTTPException(status_code=400, detail="Empty request body — send raw SVG string.")

    svg_string = svg_bytes.decode("utf-8").strip()

    if not svg_string.startswith("<svg"):
        raise HTTPException(status_code=422, detail="Body must be a raw SVG string starting with <svg.")

    try:
        png_bytes = cairosvg.svg2png(bytestring=svg_string.encode(), output_width=800)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SVG render failed: {str(e)}")

    return Response(content=png_bytes, media_type="image/png")