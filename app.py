from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import Response
from PIL import Image, ImageOps
import imagehash
import clip
import torch
import io

app = FastAPI()

device = "cpu"
_clip_model = None
_clip_preprocess = None

def _get_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
    return _clip_model, _clip_preprocess
import pypandoc
import subprocess
import tempfile
import os
import shutil

import requests

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import io, json, re
from copy import deepcopy


from telethon import TelegramClient
api_id = '33019221'
api_hash = "41a12c735f8e8a3f61ae8ab402e16cd7"
client = TelegramClient("session", api_id, api_hash)

@app.on_event("startup")
async def startup():
    try:
        await client.start()
    except EOFError:
        print("Warning: Telegram client has no valid session and no TTY to authenticate "
              "interactively — /members will not work until session.session is set up on this host.")


_LATEX_TEMPLATE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{mathpazo}
\usepackage{amsmath,amssymb}
\usepackage[hidelinks]{hyperref}
\usepackage{graphicx}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{titlesec}
\titleformat{\section}{\large\bfseries}{\thesection}{1em}{}
\setlength{\parskip}{0.6em}
\begin{document}
%s
\end{document}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Fixed brand asset (Examdost logo) — baked into the service, not per-request
# ─────────────────────────────────────────────────────────────────────────────
LOGO_SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "examdost_logo.png")


def _ensure_logo_present(tmpdir: str):
    """Copy the fixed Examdost logo into the compile directory, if it exists on disk."""
    if os.path.exists(LOGO_SOURCE_PATH):
        try:
            shutil.copy(LOGO_SOURCE_PATH, os.path.join(tmpdir, "examdost_logo.png"))
        except Exception as e:
            # Never let a logo-copy failure block the whole compile
            print(f"Warning: failed to copy logo asset: {e}")


def _read_log_tail(tmpdir: str, lines: int = 40) -> str:
    """Read the last N lines of document.log for debugging failed/timed-out compiles."""
    log_path = os.path.join(tmpdir, "document.log")
    if not os.path.exists(log_path):
        return "(no .log file produced)"
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.readlines()
        return "".join(content[-lines:])
    except Exception as e:
        return f"(failed to read log: {e})"


def _sanitize_latex(latex_source: str) -> str:
    """Fix known pandoc-template quirks that break pdflatex."""
    # Collapse \documentclass[ <blank/whitespace> ]{...} onto one line
    latex_source = re.sub(
        r'\\documentclass\[\s*\n\s*\]',
        r'\\documentclass[]',
        latex_source
    )
    return latex_source


def _ensure_full_document(latex_source: str) -> str:
    """Wrap bare LaTeX fragments in a styled document shell if needed."""
    if "\\documentclass" in latex_source:
        return latex_source
    return _LATEX_TEMPLATE % latex_source



@app.post("/html-to-latex")
async def html_to_latex(request: Request):
    body = await request.body()
    html_string = body.decode("utf-8").strip()

    if not html_string:
        raise HTTPException(status_code=400,
                            detail="Empty request body — send raw HTML string.")

    try:
        latex_output = pypandoc.convert_text(
            html_string,
            to="latex",
            format="html",
            extra_args=["--standalone", "--wrap=preserve"]
        )
        latex_output = _sanitize_latex(latex_output)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"HTML→LaTeX conversion failed: {str(e)}")

    return Response(content=latex_output, media_type="text/x-tex")


import base64
LATEX_COMPILE_API_URL = "https://latex.ytotech.com/builds/sync"

LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "examdost_logo.png")

def compile_latex_to_pdf(latex_source: str) -> bytes:
    full_source = _ensure_full_document(latex_source)
    full_source = _sanitize_latex(full_source)

    resources = [{"main": True, "content": full_source}]

    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode("ascii")
        resources.append({
            "path": "examdost_logo.png",
            "file": logo_b64,
        })

    payload = {
        "compiler": "pdflatex",
        "resources": resources,
    }

    try:
        response = requests.post(LATEX_COMPILE_API_URL, json=payload, timeout=60)
    except requests.exceptions.Timeout:
        raise RuntimeError("Remote LaTeX compile timed out after 60s.")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Could not reach LaTeX compile API: {e}")

    if response.status_code != 201:
        raise RuntimeError(f"Remote compile failed (status {response.status_code}):\n{response.text[:2000]}")

    content_type = response.headers.get("content-type", "")
    if "pdf" not in content_type.lower():
        raise RuntimeError(f"Expected PDF, got '{content_type}'. Body: {response.text[:500]}")

    return response.content



@app.post("/latex-to-pdf")
async def latex_to_pdf(request: Request):
    body = await request.body()
    latex_string = body.decode("utf-8").strip()

    if not latex_string:
        raise HTTPException(status_code=400,
                            detail="Empty request body — send raw LaTeX string.")

    try:
        pdf_bytes = compile_latex_to_pdf(latex_string)
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"LaTeX→PDF compilation failed: {str(e)}")

    return Response(content=pdf_bytes, media_type="application/pdf")



@app.get("/members")
async def members():
    group = await client.get_entity(-1003990692345)
    participants = await client.get_participants(group)

    return [
        {
            "id": p.id,
            "name": p.first_name,
            "username": p.username,
        }
        for p in participants
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Telegram photo merge — fetch two photos by Bot API file_id, merge horizontally
# ─────────────────────────────────────────────────────────────────────────────

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_API_BASE = "https://api.telegram.org"


def _download_telegram_photo(file_id: str) -> bytes:
    """Resolve a Bot API file_id to its CDN path, then download the raw bytes."""
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable is not set.")

    get_file_url = f"{TELEGRAM_API_BASE}/bot{TELEGRAM_BOT_TOKEN}/getFile"
    resp = requests.get(get_file_url, params={"file_id": file_id}, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"getFile failed for {file_id} (status {resp.status_code}): {resp.text[:500]}")

    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"getFile returned not-ok for {file_id}: {data}")

    file_path = data["result"]["file_path"]
    download_url = f"{TELEGRAM_API_BASE}/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
    file_resp = requests.get(download_url, timeout=30)
    if file_resp.status_code != 200:
        raise RuntimeError(f"File download failed for {file_id} (status {file_resp.status_code})")

    return file_resp.content


def _merge_images_horizontally(image_bytes_1: bytes, image_bytes_2: bytes) -> bytes:
    """Scale both images to a common height, then paste them side by side."""
    img1 = Image.open(io.BytesIO(image_bytes_1)).convert("RGB")
    img2 = Image.open(io.BytesIO(image_bytes_2)).convert("RGB")

    target_height = min(img1.height, img2.height)

    def _resize_to_height(img: Image.Image, height: int) -> Image.Image:
        if img.height == height:
            return img
        width = round(img.width * (height / img.height))
        return img.resize((width, height), Image.LANCZOS)

    img1 = _resize_to_height(img1, target_height)
    img2 = _resize_to_height(img2, target_height)

    merged = Image.new("RGB", (img1.width + img2.width, target_height), "white")
    merged.paste(img1, (0, 0))
    merged.paste(img2, (img1.width, 0))

    buf = io.BytesIO()
    merged.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


@app.post("/merge-telegram-photos")
async def merge_telegram_photos(request: Request):
    body = await request.body()
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Body must be valid JSON.")

    file_id_1 = payload.get("file_id_1")
    file_id_2 = payload.get("file_id_2")
    if not file_id_1 or not file_id_2:
        raise HTTPException(status_code=400,
                            detail="Both 'file_id_1' and 'file_id_2' are required.")

    try:
        image_bytes_1 = _download_telegram_photo(file_id_1)
        image_bytes_2 = _download_telegram_photo(file_id_2)
        merged_bytes = _merge_images_horizontally(image_bytes_1, image_bytes_2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Photo merge failed: {str(e)}")

    return Response(content=merged_bytes, media_type="image/png")


# ─────────────────────────────────────────────────────────────────────────────
# Spec normalizer — repairs LLM-generated JSON before rendering
# ─────────────────────────────────────────────────────────────────────────────

def _repair_json_string(raw: str) -> str:
    """
    Best-effort repair of malformed JSON strings from LLMs.
    Handles:
      - Trailing commas before } or ]
      - Missing closing brackets/braces
      - Markdown code fences around JSON
    """
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw.strip(), flags=re.MULTILINE)
    raw = raw.strip()
    raw = re.sub(r",\s*([}\]])", r"\1", raw)

    try:
        json.loads(raw)
        return raw
    except json.JSONDecodeError:
        pass

    depth_brace = 0
    depth_bracket = 0
    in_string = False
    escape_next = False
    for ch in raw:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace -= 1
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            depth_bracket -= 1

    raw += "]" * max(0, depth_bracket) + "}" * max(0, depth_brace)
    return raw


def safe_parse(raw: str) -> dict:
    """Parse JSON with repair fallback. Raises ValueError if unrecoverable."""
    repaired = _repair_json_string(raw)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        raise ValueError(f"Cannot parse JSON even after repair: {e}")


_GRAPH_ARRAY_FIELDS = ["curves", "points", "regions", "lines", "annotations", "shapes"]

_FIELD_ALIASES = {
    "graph_type":  "type",
    "chart_type":  "type",
    "plot_type":   "type",
    "x_label":     "xlabel",
    "y_label":     "ylabel",
    "x_axis":      "xlabel",
    "y_axis":      "ylabel",
    "x_range":     "xrange",
    "y_range":     "yrange",
    "xlim":        "xrange",
    "ylim":        "yrange",
    "fig_width":   "figwidth",
    "fig_height":  "figheight",
    "width":       "figwidth",
    "height":      "figheight",
}

_ANNOTATION_FIELDS = {"text", "x", "y", "tx", "ty"}
_SHAPE_TYPES = {"circle", "polygon", "arrow", "line_segment", "point"}


def _looks_like_annotation(obj: object) -> bool:
    if not isinstance(obj, dict):
        return False
    return bool(_ANNOTATION_FIELDS & set(obj.keys()))


def _looks_like_shape(obj: object) -> bool:
    if not isinstance(obj, dict):
        return False
    return "type" in obj and obj.get("type") in _SHAPE_TYPES


def _flatten_nested_lists(items: list) -> list:
    """Flatten one level of accidental list-of-lists."""
    result = []
    for item in items:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result


def normalize_graph_spec(spec: dict) -> dict:
    """
    Normalize an LLM-generated graph spec dict.
    Fixes:
    - Field name aliases  (x_range → xrange, fig_width → figwidth, etc.)
    - annotations array accidentally containing shapes
    - shapes accidentally nested inside annotations
    - list-of-lists for any array field
    - Missing array fields (default to [])
    - xrange/yrange as dicts {min,max} or strings "[-10, 10]"
    - type casing  ("Geometric" → "geometric")
    - Arrow coordinate aliases  (start/end, from/to, origin/tip)
    - polygon "vertices" → "points"
    """
    spec = deepcopy(spec)

    # 1. Resolve top-level field aliases
    for alias, canonical in _FIELD_ALIASES.items():
        if alias in spec and canonical not in spec:
            spec[canonical] = spec.pop(alias)
        elif alias in spec:
            del spec[alias]

    # 2. Normalise type to lowercase
    if "type" in spec:
        spec["type"] = str(spec["type"]).lower()

    # 3. Ensure all array fields exist
    for field in _GRAPH_ARRAY_FIELDS:
        if field not in spec:
            spec[field] = []
        elif not isinstance(spec[field], list):
            spec[field] = [spec[field]] if spec[field] else []

    # 4. Flatten list-of-lists
    for field in _GRAPH_ARRAY_FIELDS:
        spec[field] = _flatten_nested_lists(spec[field])

    # 5. Rescue shapes that ended up inside annotations
    rescued_shapes = []
    clean_annotations = []
    for item in spec["annotations"]:
        if _looks_like_shape(item):
            rescued_shapes.append(item)
        elif _looks_like_annotation(item):
            clean_annotations.append(item)
    spec["annotations"] = clean_annotations

    existing_shape_ids = {id(s) for s in spec["shapes"]}
    for shape in rescued_shapes:
        if id(shape) not in existing_shape_ids:
            spec["shapes"].append(shape)

    # 6. Normalise xrange / yrange
    for range_key in ("xrange", "yrange"):
        val = spec.get(range_key)
        if val is None:
            continue
        if isinstance(val, str):
            nums = re.findall(r"-?\d+(?:\.\d+)?", val)
            if len(nums) >= 2:
                spec[range_key] = [float(nums[0]), float(nums[1])]
            else:
                del spec[range_key]
        elif isinstance(val, dict):
            lo = val.get("min", val.get("from", val.get("start")))
            hi = val.get("max", val.get("to", val.get("end")))
            if lo is not None and hi is not None:
                spec[range_key] = [float(lo), float(hi)]
            else:
                del spec[range_key]
        elif isinstance(val, (list, tuple)) and len(val) == 2:
            spec[range_key] = [float(val[0]), float(val[1])]
        else:
            del spec[range_key]

    # 7. Normalise shape arrow coordinate aliases
    normalized_shapes = []
    for s in spec.get("shapes", []):
        if not isinstance(s, dict):
            continue
        s = dict(s)
        if s.get("type") == "arrow":
            if "points" in s and isinstance(s["points"], list) and len(s["points"]) == 2:
                start_pt, end_pt = s["points"]
                if isinstance(start_pt, (list, tuple)) and isinstance(end_pt, (list, tuple)):
                    s.setdefault("x1", float(start_pt[0]))
                    s.setdefault("y1", float(start_pt[1]))
                    s.setdefault("x2", float(end_pt[0]))
                    s.setdefault("y2", float(end_pt[1]))
            for src_key, xk, yk in [
                ("start",  "x1", "y1"),
                ("end",    "x2", "y2"),
                ("from",   "x1", "y1"),
                ("to",     "x2", "y2"),
                ("origin", "x1", "y1"),
                ("tip",    "x2", "y2"),
            ]:
                if src_key in s:
                    pt = s[src_key]
                    if isinstance(pt, (list, tuple)):
                        s.setdefault(xk, float(pt[0]))
                        s.setdefault(yk, float(pt[1]))
                    elif isinstance(pt, dict):
                        s.setdefault(xk, float(pt.get("x", 0)))
                        s.setdefault(yk, float(pt.get("y", 0)))
        elif s.get("type") == "polygon":
            if "vertices" in s and "points" not in s:
                s["points"] = s["vertices"]
        normalized_shapes.append(s)
    spec["shapes"] = normalized_shapes

    return spec


_DIAGRAM_TYPE_ALIASES = {
    "graph":     "graph",
    "plot":      "graph",
    "curve":     "graph",
    "waveform":  "waveform",
    "wave":      "waveform",
    "signal":    "waveform",
    "geometric": "geometric",
    "geometry":  "geometric",
    "shapes":    "geometric",
    "phasor":    "phasor",
    "phasors":   "phasor",
    "circuit":   "circuit",
    "schematic": "circuit",
}


def normalize_diagram_payload(payload: dict) -> dict:
    """
    Normalize the top-level /render-diagram payload.

    Handles:
    - diagram_type aliases
    - graph_spec / circuit_spec embedded under unexpected keys
    - spec fields inlined directly into payload (no nested key)
    - diagram_needed missing (defaults to True)
    """
    payload = deepcopy(payload)

    payload.setdefault("diagram_needed", True)

    raw_type = str(payload.get("diagram_type", "")).lower().strip()
    payload["diagram_type"] = _DIAGRAM_TYPE_ALIASES.get(raw_type, raw_type)

    dtype = payload["diagram_type"]

    if dtype == "circuit":
        if "circuit_spec" not in payload:
            for key in ("spec", "circuit", "schematic_spec", "diagram_spec"):
                if key in payload and isinstance(payload[key], dict):
                    payload["circuit_spec"] = payload.pop(key)
                    break
    else:
        if "graph_spec" not in payload:
            for key in ("spec", "graph", "plot_spec", "diagram_spec", "chart_spec"):
                if key in payload and isinstance(payload[key], dict):
                    payload["graph_spec"] = payload.pop(key)
                    break

        # Fallback: spec fields inlined directly into the payload
        if "graph_spec" not in payload:
            inline_keys = {"type", "shapes", "curves", "points", "lines",
                           "regions", "annotations", "xrange", "yrange"}
            if inline_keys & set(payload.keys()):
                spec_keys = inline_keys | {"title", "xlabel", "ylabel",
                                           "figwidth", "figheight",
                                           "waveform", "frequency", "amplitude"}
                graph_spec = {k: payload.pop(k) for k in list(payload.keys())
                              if k in spec_keys}
                payload["graph_spec"] = graph_spec

        if "graph_spec" in payload and isinstance(payload["graph_spec"], dict):
            gs = payload["graph_spec"]
            if dtype in ("waveform", "geometric", "phasor"):
                gs.setdefault("type", dtype)
            else:
                gs.setdefault("type", "graph")
            payload["graph_spec"] = normalize_graph_spec(gs)

    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Colour helper
# ─────────────────────────────────────────────────────────────────────────────

def parse_color(color):
    """Convert CSS rgba()/rgb() strings to matplotlib (r,g,b,a) tuples."""
    if not isinstance(color, str):
        return color
    m = re.match(r'rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+))?\s*\)', color.strip())
    if m:
        r, g, b = int(m.group(1))/255, int(m.group(2))/255, int(m.group(3))/255
        a = float(m.group(4)) if m.group(4) is not None else 1.0
        return (r, g, b, a)
    return color


# ─────────────────────────────────────────────────────────────────────────────
# Image pre-processing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = ImageOps.exif_transpose(img)
    img = img.resize((256, 256), Image.LANCZOS)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints — hashing / embeddings
# ─────────────────────────────────────────────────────────────────────────────

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
    model, preprocess = _get_clip()
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
    model, preprocess = _get_clip()
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


# ─────────────────────────────────────────────────────────────────────────────
# Graph / waveform / geometric / phasor renderer
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_graph_spec(spec: dict) -> dict:
    """Absorb LLM field-name variations so renderers see canonical keys."""
    spec = dict(spec)
    normalized = []
    for s in spec.get("shapes", []):
        s = dict(s)
        if s.get("type") == "arrow":
            for src_key, xk, yk in [("start", "x1", "y1"), ("end",  "x2", "y2"),
                                     ("from",  "x1", "y1"), ("to",   "x2", "y2"),
                                     ("origin","x1", "y1"), ("tip",  "x2", "y2")]:
                if src_key in s:
                    pt = s[src_key]
                    if isinstance(pt, (list, tuple)):
                        s.setdefault(xk, float(pt[0]))
                        s.setdefault(yk, float(pt[1]))
                    elif isinstance(pt, dict):
                        s.setdefault(xk, float(pt.get("x", 0)))
                        s.setdefault(yk, float(pt.get("y", 0)))
        elif s.get("type") == "polygon":
            if "vertices" in s and "points" not in s:
                s["points"] = s["vertices"]
        normalized.append(s)
    spec["shapes"] = normalized
    return spec


def draw_graph(spec: dict) -> bytes:
    spec = _normalize_graph_spec(spec)
    graph_type = spec.get("type", "graph")

    # ── Auto-size figure for geometric diagrams to avoid squished aspect ──────
    if graph_type == "geometric" and "figwidth" not in spec and "figheight" not in spec:
        xr = spec.get("xrange", [-10, 10])
        yr = spec.get("yrange", None)
        x_span = xr[1] - xr[0]
        y_span = (yr[1] - yr[0]) if yr else x_span
        # Scale so the longer axis = 7 inches, shorter proportional, capped 3–10
        scale = 7.0 / max(x_span, y_span)
        figw = max(3.0, min(10.0, x_span * scale))
        figh = max(3.0, min(10.0, y_span * scale))
    else:
        default_w = 10 if graph_type == "waveform" else 7
        figw = spec.get("figwidth", default_w)
        figh = spec.get("figheight", 5 if graph_type == "waveform" else 4.5)

    fig, ax = plt.subplots(figsize=(figw, figh), dpi=120)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    title       = spec.get("title", "")
    xlabel      = spec.get("xlabel", "")
    ylabel      = spec.get("ylabel", "")
    spec_has_xrange = "xrange" in spec
    xrange      = spec.get("xrange", [-10, 10])
    yrange      = spec.get("yrange", None)
    curves      = spec.get("curves", [])
    points      = spec.get("points", [])
    regions     = spec.get("regions", [])
    lines       = spec.get("lines", [])
    annotations = spec.get("annotations", [])

    x = np.linspace(xrange[0], xrange[1], 1000)

    # ── Curves ───────────────────────────────────────────────────────────────
    for curve in curves:
        expr  = curve.get("expr", "")
        label = curve.get("label", "")
        color = curve.get("color", "blue")
        style = curve.get("style", "-")
        try:
            y = eval(expr, {"x": x, "np": np, "sin": np.sin,
                            "cos": np.cos, "tan": np.tan,
                            "exp": np.exp, "log": np.log,
                            "sqrt": np.sqrt, "pi": np.pi,
                            "abs": np.abs, "e": np.e})
            ax.plot(x, y, linestyle=style, color=color,
                    label=label if label else None, linewidth=2)
        except Exception:
            continue

    # ── Geometric shapes ─────────────────────────────────────────────────────
    if graph_type == "geometric":
        shapes = spec.get("shapes", [])
        for shape in shapes:
            stype = shape.get("type", "")
            color = shape.get("color", "black")
            lw    = shape.get("linewidth", 2)

            if stype == "circle":
                circle = plt.Circle(
                    (shape["cx"], shape["cy"]),
                    shape["r"],
                    fill=False, color=color, linewidth=lw
                )
                ax.add_patch(circle)

            elif stype == "polygon":
                pts = shape["points"]
                polygon = mpatches.Polygon(
                    pts, closed=True,
                    fill=False, color=color, linewidth=lw
                )
                ax.add_patch(polygon)

            elif stype == "arrow":
                ax.annotate("",
                    xy=(shape["x2"], shape["y2"]),
                    xytext=(shape["x1"], shape["y1"]),
                    arrowprops=dict(arrowstyle="->",
                                    color=color, lw=lw))

            elif stype == "line_segment":
                ax.plot(
                    [shape["x1"], shape["x2"]],
                    [shape["y1"], shape["y2"]],
                    color=color, linewidth=lw
                )

            elif stype == "point":
                ax.plot(shape["x"], shape["y"],
                        "o", color=color, markersize=8)

            if shape.get("label"):
                lx = shape.get("lx", shape.get("cx", shape.get("x2", shape.get("x1", 0))))
                ly = shape.get("ly", shape.get("cy", shape.get("y2", shape.get("y1", 0))))
                ax.text(lx, ly, shape["label"], fontsize=10,
                        ha="left", va="bottom")

        # Respect explicit ranges; only use equal aspect when no ranges given
        if "xrange" in spec:
            ax.set_xlim(xrange[0], xrange[1])
        if yrange:
            ax.set_ylim(yrange[0], yrange[1])
        if "xrange" not in spec and not yrange:
            ax.set_aspect("equal")
            ax.autoscale()

    # ── Phasor diagram ────────────────────────────────────────────────────────
    if graph_type == "phasor":
        shapes = spec.get("shapes", [])

        resolved = []
        for s in shapes:
            s = dict(s)
            if s.get("type") == "arrow":
                x1 = s.get("x1", 0)
                y1 = s.get("y1", 0)
                if "magnitude" in s and "angle_deg" in s:
                    mag = s["magnitude"]
                    ang = np.radians(s["angle_deg"])
                    s["_x2"] = x1 + mag * np.cos(ang)
                    s["_y2"] = y1 + mag * np.sin(ang)
                else:
                    s["_x2"] = s.get("x2", 1)
                    s["_y2"] = s.get("y2", 0)
                s["_mag"] = np.hypot(s["_x2"] - x1, s["_y2"] - y1)
            resolved.append(s)

        max_mag = max((s["_mag"] for s in resolved if "_mag" in s), default=1)

        if spec.get("reference_circle", True) and max_mag > 0:
            ref_circ = plt.Circle((0, 0), max_mag, fill=False,
                                  color="lightgray", linewidth=1, linestyle="--")
            ax.add_patch(ref_circ)

        for s in resolved:
            stype = s.get("type", "arrow")
            color = parse_color(s.get("color", "steelblue"))
            lw    = s.get("linewidth", 2)

            if stype == "arrow":
                x1 = s.get("x1", 0); y1 = s.get("y1", 0)
                x2 = s["_x2"];       y2 = s["_y2"]
                ax.annotate("",
                    xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=lw, mutation_scale=14))
                if s.get("label"):
                    off = s.get("label_offset", 0.08)
                    dx = x2 - x1; dy = y2 - y1
                    lx = x2 + dx * off if dx else x2
                    ly = y2 + dy * off if dy else y2 + off * max_mag
                    ax.text(lx, ly, s["label"], fontsize=9, color=color,
                            ha="left", va="center")

            elif stype == "circle":
                circ = plt.Circle(
                    (s.get("cx", 0), s.get("cy", 0)),
                    s.get("r", 1), fill=False, color=color, linewidth=lw)
                ax.add_patch(circ)

        ax.set_aspect("equal")

        if spec_has_xrange:
            ax.set_xlim(xrange[0], xrange[1])
        else:
            xs = ([s.get("x1", 0) for s in resolved if s.get("type") == "arrow"] +
                  [s.get("_x2", 0) for s in resolved if s.get("type") == "arrow"] + [0])
            pad_x = max(abs(min(xs)), abs(max(xs)), 1) * 1.3
            ax.set_xlim(-pad_x, pad_x)

        if not yrange:
            ys = ([s.get("y1", 0) for s in resolved if s.get("type") == "arrow"] +
                  [s.get("_y2", 0) for s in resolved if s.get("type") == "arrow"] + [0])
            pad_y = max(abs(min(ys)), abs(max(ys)), 1) * 1.3
            ax.set_ylim(-pad_y, pad_y)

    # ── Waveform presets ──────────────────────────────────────────────────────
    if graph_type == "waveform" and not curves:
        waveform = spec.get("waveform", "sine")
        freq     = spec.get("frequency", 1)
        amp      = spec.get("amplitude", 1)
        t = np.linspace(xrange[0], xrange[1], 2000)
        if waveform == "sine":
            y = amp * np.sin(2 * np.pi * freq * t)
        elif waveform == "square":
            y = amp * np.sign(np.sin(2 * np.pi * freq * t))
        elif waveform == "triangle":
            y = amp * (2/np.pi) * np.arcsin(np.sin(2 * np.pi * freq * t))
        elif waveform == "sawtooth":
            y = amp * (2 * (t * freq - np.floor(0.5 + t * freq)))
        else:
            y = amp * np.sin(2 * np.pi * freq * t)
        ax.plot(t, y, color="blue", linewidth=2,
                label=spec.get("label", waveform))

    # ── Reference lines ───────────────────────────────────────────────────────
    for line in lines:
        lcolor = line.get("color", "gray")
        lstyle = line.get("style", "--")
        llabel = line.get("label", "")
        if line.get("axis") == "x":
            xv = line.get("value", 0)
            ax.axvline(x=xv, color=lcolor, linestyle=lstyle, linewidth=1)
            if llabel:
                import matplotlib.transforms as mtransforms
                trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
                ax.text(xv, 0.97, llabel, transform=trans,
                        rotation=90, ha="right", va="top",
                        fontsize=8, color=lcolor)
        elif line.get("axis") == "y":
            yv = line.get("value", 0)
            ax.axhline(y=yv, color=lcolor, linestyle=lstyle, linewidth=1)
            if llabel:
                import matplotlib.transforms as mtransforms
                trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
                ax.text(0.99, yv, llabel, transform=trans,
                        ha="right", va="bottom",
                        fontsize=8, color=lcolor)

    # ── Shaded regions ────────────────────────────────────────────────────────
    _eval_ns = {"np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan,
                "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
                "pi": np.pi, "abs": np.abs, "e": np.e}
    for region in regions:
        raw_color = parse_color(region.get("color", "yellow"))
        if isinstance(raw_color, tuple) and len(raw_color) == 4:
            fill_color = raw_color[:3]
            fill_alpha = raw_color[3]
        else:
            fill_color = raw_color
            fill_alpha = region.get("alpha", 0.2)
        rlabel = region.get("label", "")

        if "x1_expr" in region or "x2_expr" in region:
            yr = yrange if yrange else [xrange[0], xrange[1]]
            y1_val = region.get("y1", yr[0])
            y2_val = region.get("y2", yr[1])
            y_vals = np.linspace(y1_val, y2_val, 500)
            ns = {**_eval_ns, "y": y_vals}
            try:
                x1_vals = eval(region.get("x1_expr", str(xrange[0])), ns)
                x2_vals = eval(region.get("x2_expr", str(xrange[1])), ns)
                ax.fill_betweenx(y_vals, x1_vals, x2_vals,
                                 color=fill_color, alpha=fill_alpha,
                                 label=rlabel if rlabel else None)
            except Exception:
                pass
        else:
            x1 = region.get("x1", xrange[0])
            x2 = region.get("x2", xrange[1])
            ax.axvspan(x1, x2, alpha=fill_alpha, color=fill_color,
                       label=rlabel if rlabel else None)

    # ── Points ────────────────────────────────────────────────────────────────
    for pt in points:
        ax.plot(pt["x"], pt["y"],
                marker=pt.get("marker", "o"),
                color=pt.get("color", "red"),
                markersize=8)
        if pt.get("label"):
            ax.annotate(pt["label"],
                        xy=(pt["x"], pt["y"]),
                        xytext=(pt["x"] + 0.3, pt["y"] + 0.3),
                        fontsize=10)

    # ── Text annotations ──────────────────────────────────────────────────────
    for ann in annotations:
        if "x" not in ann or "y" not in ann:
            continue
        ann_x, ann_y = ann["x"], ann["y"]
        ann_tx = ann.get("tx", ann_x)
        ann_ty = ann.get("ty", ann_y)
        kw = {"fontsize": ann.get("fontsize", 10)}
        if ann_tx != ann_x or ann_ty != ann_y:
            kw["xytext"] = (ann_tx, ann_ty)
            kw["arrowprops"] = dict(arrowstyle="->", color="black")
        ax.annotate(ann.get("text", ""), xy=(ann_x, ann_y), **kw)


    # ── Axes and grid ─────────────────────────────────────────────────────────
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    if yrange:
        ax.set_ylim(yrange[0], yrange[1])
    if curves or (graph_type == "waveform") or regions:
        ax.legend(fontsize=9)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints — graph rendering
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/render-graph")
async def render_graph(request: Request):
    body = await request.body()
    try:
        spec = json.loads(body.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Body must be valid JSON.")

    spec = normalize_graph_spec(spec)  # ← repair LLM inconsistencies

    try:
        png_bytes = draw_graph(spec)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph render failed: {str(e)}")

    return Response(content=png_bytes, media_type="image/png")


# ─────────────────────────────────────────────────────────────────────────────
# SVG → PNG helper
# ─────────────────────────────────────────────────────────────────────────────

def _svg_to_png(svg_string: str, width: int = 800) -> bytes:
    """Convert SVG string to PNG bytes via cairosvg or svglib fallback."""
    try:
        import cairosvg
        return cairosvg.svg2png(bytestring=svg_string.encode(), output_width=width)
    except (ImportError, OSError):
        pass

    try:
        import tempfile, os
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPDF
        import fitz

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False,
                                         mode="w", encoding="utf-8") as f:
            f.write(svg_string)
            tmp_path = f.name
        try:
            drawing = svg2rlg(tmp_path)
            if drawing is None:
                raise ValueError("svglib could not parse the SVG document.")
            pdf_buf = io.BytesIO()
            renderPDF.drawToFile(drawing, pdf_buf)
            pdf_buf.seek(0)
            pdf_bytes = pdf_buf.read()
            doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
            page = doc[0]
            zoom = (width / page.rect.width) if page.rect.width else 1.0
            pix  = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            return pix.tobytes("png")
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        raise RuntimeError(f"SVG render failed: {e}") from e


@app.post("/render")
async def render_svg(request: Request):
    svg_bytes = await request.body()

    if not svg_bytes:
        raise HTTPException(status_code=400,
                            detail="Empty request body — send raw SVG string.")

    svg_string = svg_bytes.decode("utf-8").strip()

    if not svg_string.startswith("<svg"):
        raise HTTPException(status_code=422,
                            detail="Body must be a raw SVG string starting with <svg.")

    try:
        png_bytes = _svg_to_png(svg_string, width=800)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return Response(content=png_bytes, media_type="image/png")


# ─────────────────────────────────────────────────────────────────────────────
# Circuit renderer
# ─────────────────────────────────────────────────────────────────────────────

import schemdraw
import schemdraw.elements as elm

ELEMENT_MAP = {
    # ── Passives ──────────────────────────────────────────────────────────────
    "resistor":              elm.Resistor,
    "resistor_iec":          elm.ResistorIEC,
    "resistor_ieee":         elm.ResistorIEEE,
    "resistor_var":          elm.ResistorVar,
    "resistor_var_iec":      elm.ResistorVarIEC,
    "rbox":                  elm.RBox,
    "rbox_var":              elm.RBoxVar,
    "potentiometer":         elm.Potentiometer,
    "potentiometer_iec":     elm.PotentiometerIEC,
    "potentiometer_ieee":    elm.PotentiometerIEEE,
    "capacitor":             elm.Capacitor,
    "capacitor2":            elm.Capacitor2,
    "capacitor_var":         elm.CapacitorVar,
    "capacitor_trim":        elm.CapacitorTrim,
    "inductor":              elm.Inductor,
    "inductor2":             elm.Inductor2,
    "crystal":               elm.Crystal,
    "memristor":             elm.Memristor,
    "memristor2":            elm.Memristor2,
    "cpe":                   elm.CPE,
    "fuse":                  elm.Fuse,
    "fuse_ieee":             elm.FuseIEEE,
    "fuse_iec":              elm.FuseIEC,
    "fuse_us":               elm.FuseUS,
    "thermistor":            elm.Thermistor,
    "photoresistor":         elm.Photoresistor,
    "photoresistor_iec":     elm.PhotoresistorIEC,
    "photoresistor_ieee":    elm.PhotoresistorIEEE,
    "photoresistor_box":     elm.PhotoresistorBox,
    "varactor":              elm.Varactor,
    # ── Sources ───────────────────────────────────────────────────────────────
    "battery":               elm.Battery,
    "battery_cell":          elm.BatteryCell,
    "battery_double":        elm.BatteryDouble,
    "source_v":              elm.SourceV,
    "source_i":              elm.SourceI,
    "source_sin":            elm.SourceSin,
    "source_square":         elm.SourceSquare,
    "source_triangle":       elm.SourceTriangle,
    "source_pulse":          elm.SourcePulse,
    "source_ramp":           elm.SourceRamp,
    "source_controlled_v":   elm.SourceControlledV,
    "source_controlled_i":   elm.SourceControlledI,
    "solar":                 elm.Solar,
    # ── Diodes ────────────────────────────────────────────────────────────────
    "diode":                 elm.Diode,
    "zener":                 elm.Zener,
    "led":                   elm.LED,
    "led2":                  elm.LED2,
    "schottky":              elm.Schottky,
    "tunnel_diode":          elm.DiodeTunnel,
    "diode_tvs":             elm.DiodeTVS,
    "diac":                  elm.Diac,
    "photodiode":            elm.Photodiode,
    "diode_shockley":        elm.DiodeShockley,
    # ── Switches ──────────────────────────────────────────────────────────────
    "switch":                elm.Switch,
    "switch_spdt":           elm.SwitchSpdt,
    "switch_spdt2":          elm.SwitchSpdt2,
    "switch_dpst":           elm.SwitchDpst,
    "switch_dpdt":           elm.SwitchDpdt,
    "switch_reed":           elm.SwitchReed,
    "switch_rotary":         elm.SwitchRotary,
    "button":                elm.Button,
    "relay":                 elm.Relay,
    "breaker":               elm.Breaker,
    # ── Grounds / Supply ──────────────────────────────────────────────────────
    "ground":                elm.Ground,
    "ground_chassis":        elm.GroundChassis,
    "ground_signal":         elm.GroundSignal,
    "vdd":                   elm.Vdd,
    "vss":                   elm.Vss,
    # ── BJT Transistors ───────────────────────────────────────────────────────
    "bjt_npn":               elm.BjtNpn,
    "bjt_pnp":               elm.BjtPnp,
    "bjt_npn2":              elm.BjtNpn2,
    "bjt_pnp2":              elm.BjtPnp2,
    "bjt_pnp2c":             elm.BjtPnp2c,
    "npn_photo":             elm.NpnPhoto,
    "pnp_photo":             elm.PnpPhoto,
    "npn_schottky":          elm.NpnSchottky,
    "pnp_schottky":          elm.PnpSchottky,
    # ── FET Transistors ───────────────────────────────────────────────────────
    "mosfet_n":              elm.NFet,
    "mosfet_p":              elm.PFet,
    "nfet":                  elm.NFet,
    "pfet":                  elm.PFet,
    "nfet2":                 elm.NFet2,
    "pfet2":                 elm.PFet2,
    "nmos":                  elm.NMos,
    "pmos":                  elm.PMos,
    "nmos2":                 elm.NMos2,
    "pmos2":                 elm.PMos2,
    "jfet_n":                elm.JFetN,
    "jfet_p":                elm.JFetP,
    "jfet_n2":               elm.JFetN2,
    "jfet_p2":               elm.JFetP2,
    "igbt_n":                elm.IgbtN,
    "igbt_p":                elm.IgbtP,
    "hemt":                  elm.Hemt,
    "analog_nfet":           elm.AnalogNFet,
    "analog_pfet":           elm.AnalogPFet,
    "analog_biased_fet":     elm.AnalogBiasedFet,
    # ── Op-Amps / ICs ─────────────────────────────────────────────────────────
    "opamp":                 elm.Opamp,
    "ic":                    elm.Ic,
    "ic555":                 elm.Ic555,
    "ic_dip":                elm.IcDIP,
    "multiplexer":           elm.Multiplexer,
    "dflipflop":             elm.DFlipFlop,
    "jkflipflop":            elm.JKFlipFlop,
    "rectifier":             elm.Rectifier,
    "wheatstone":            elm.Wheatstone,
    "optocoupler":           elm.Optocoupler,
    "scr":                   elm.SCR,
    "triac":                 elm.Triac,
    "josephson":             elm.Josephson,
    # ── Transformers ──────────────────────────────────────────────────────────
    "transformer":           elm.Transformer,
    # ── Meters & Annotation ───────────────────────────────────────────────────
    "meter":                 elm.MeterBox,
    "meter_box":             elm.MeterBox,
    "meter_v":               elm.MeterV,
    "meter_a":               elm.MeterA,
    "meter_i":               elm.MeterI,
    "meter_ohm":             elm.MeterOhm,
    "meter_analog":          elm.MeterAnalog,
    "meter_digital":         elm.MeterDigital,
    "meter_arrow":           elm.MeterArrow,
    "current_label":         elm.CurrentLabel,
    "current_label_inline":  elm.CurrentLabelInline,
    "voltage_label":         elm.VoltageLabelArc,
    "loop_current":          elm.LoopCurrent,
    "loop_arrow":            elm.LoopArrow,
    # ── Misc Passive/Active ───────────────────────────────────────────────────
    "lamp":                  elm.Lamp,
    "lamp2":                 elm.Lamp2,
    "speaker":               elm.Speaker,
    "mic":                   elm.Mic,
    "motor":                 elm.Motor,
    "antenna":               elm.Antenna,
    "antenna_loop":          elm.AntennaLoop,
    "antenna_loop2":         elm.AntennaLoop2,
    "neon":                  elm.Neon,
    "spark_gap":             elm.SparkGap,
    "gap":                   elm.Gap,
    "oscilloscope":          elm.Oscilloscope,
    # ── Connectors ────────────────────────────────────────────────────────────
    "terminal":              elm.Terminal,
    "jack":                  elm.Jack,
    "plug":                  elm.Plug,
    "header":                elm.Header,
    # ── Lines / Wire / Shapes ─────────────────────────────────────────────────
    "line":                  elm.Line,
    "wire":                  elm.Line,
    "arc2":                  elm.Arc2,
    "arc3":                  elm.Arc3,
    "arc_loop":              elm.ArcLoop,
    "dot":                   elm.Dot,
    "dot_dot_dot":           elm.DotDotDot,
    "no_connect":            elm.NoConnect,
    "arrow":                 elm.Arrow,
    "arrowhead":             elm.Arrowhead,
    "label":                 elm.Label,
    "annotate":              elm.Annotate,
    "tag":                   elm.Tag,
}

DIRECTION_MAP = {
    "right": "right", "r": "right",
    "left":  "left",  "l": "left",
    "up":    "up",    "u": "up",
    "down":  "down",  "d": "down",
}

_ELEMENT_PINS = [
    "start", "end", "center",
    "base", "collector", "emitter",
    "gate", "drain", "source",
    "in1", "in2", "out", "vs", "vd",
    "p1", "p2", "s1", "s2",
    "tap",
    "absw",
]

# Compound suffixes the LLM commonly references that we resolve manually
_ANCHOR_SUFFIXES = ["start", "end", "center", "gate", "base",
                    "collector", "emitter", "drain", "source",
                    "in1", "in2", "out"]



def _resolve_at(kwargs: dict, at, anchors: dict):
    if at is None:
        return
    if isinstance(at, (list, tuple)):
        kwargs["at"] = at
        return
    if not isinstance(at, str):
        return

    # Direct match
    if at in anchors:
        kwargs["at"] = anchors[at]
        return

    # Compound key fallback: "src.start" → try "src" if "src.start" missing
    if "." in at:
        base, suffix = at.rsplit(".", 1)
        if at not in anchors and base in anchors:
            base_pt = anchors[base]
            # For .start: the bare name saves .end, so we can't derive
            # .start from it — but we may have saved it as _cursor_before
            # on the element's name. Try the synthesised key first.
            synth_key = f"_before_{base}"
            if suffix == "start" and synth_key in anchors:
                kwargs["at"] = anchors[synth_key]
            elif suffix == "end":
                # bare name IS the end
                kwargs["at"] = base_pt
            elif suffix == "center" and f"{base}.center" not in anchors:
                kwargs["at"] = base_pt
            else:
                # Best available fallback: use the bare anchor
                kwargs["at"] = base_pt
            return

    # Unknown anchor — log and skip rather than crashing
    import warnings
    warnings.warn(f"Anchor '{at}' not found in anchors dict. Skipping 'at'.")


def _save_anchors(anchors: dict, name: str, el):
    # Always save bare name → end (most useful cursor position)
    if hasattr(el, 'end'):
        anchors[name] = el.end
    elif hasattr(el, 'center'):
        anchors[name] = el.center
    elif hasattr(el, 'start'):
        anchors[name] = el.start

    # Save every real schemdraw pin as name.pin
    for pin in _ANCHOR_SUFFIXES:
        val = getattr(el, pin, None)
        if val is not None:
            anchors[f"{name}.{pin}"] = val

    # CRITICAL: always save .start and .end explicitly even if
    # schemdraw doesn't expose them as named pins — derive from
    # BoundingBox or the drawing position stored on the element.
    if hasattr(el, 'start') and f"{name}.start" not in anchors:
        anchors[f"{name}.start"] = el.start
    if hasattr(el, 'end') and f"{name}.end" not in anchors:
        anchors[f"{name}.end"] = el.end

    # Fallback: if element has no .start but has .end, synthesise
    # .start as the cursor position BEFORE the element was added.
    # We store _prev_end before each element for exactly this purpose.
    if f"{name}.start" not in anchors and "_cursor_before" in anchors:
        anchors[f"{name}.start"] = anchors["_cursor_before"]



def draw_circuit(spec: dict) -> bytes:
    title    = spec.get("title", "")
    elements = spec.get("elements", [])
    figw     = spec.get("figwidth", 14)
    figh     = spec.get("figheight", 6)
    fontsize = spec.get("fontsize", 11)
    unit     = spec.get("unit", 3.0)

    fig, ax = plt.subplots(figsize=(figw, figh))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_axis_off()

    with schemdraw.Drawing(canvas=ax) as d:
        d.config(fontsize=fontsize, unit=unit)
        anchors: dict = {}

        for el_spec in elements:
            el_type   = el_spec.get("type", "line").lower()
            direction = DIRECTION_MAP.get(el_spec.get("direction", "right"), "right")
            name      = el_spec.get("name", None)

            # Snapshot cursor before this element so we can synthesise
            # .start anchors for elements that don't expose one directly.
            if hasattr(d, 'here'):
                anchors["_cursor_before"] = d.here

            # ── Label(s) config ─────────────────────────────────────────────
            label     = el_spec.get("label", "")
            label_loc = el_spec.get("label_loc", None)
            labels    = el_spec.get("labels", [])

            length    = el_spec.get("length", None)
            at        = el_spec.get("at", None)
            tox       = el_spec.get("tox", None)
            toy       = el_spec.get("toy", None)

            flip      = el_spec.get("flip", False)
            reverse   = el_spec.get("reverse", False)
            color     = el_spec.get("color", None)
            linewidth = el_spec.get("linewidth", None)
            fill      = el_spec.get("fill", None)
            dot       = el_spec.get("dot", False)
            idot      = el_spec.get("idot", False)
            zorder    = el_spec.get("zorder", None)

            extra     = el_spec.get("kwargs", {})

            if el_type in ("move", "jump"):
                kw = {"d": direction}
                if length:
                    kw["l"] = length
                _resolve_at(kw, at, anchors)
                el = d.add(elm.Line(**kw).color("white").linewidth(0))
                if name:
                    _save_anchors(anchors, name, el)
                if hasattr(el, 'end'):
                    anchors["_prev_end"] = el.end
                if hasattr(el, 'start'):
                    anchors["_prev_start"] = el.start
                continue

            ElClass = ELEMENT_MAP.get(el_type)
            if ElClass is None:
                continue

            # Save a named before-snapshot so compound ".start" refs work
            if name and hasattr(d, 'here'):
                anchors[f"_before_{name}"] = d.here

            # ── Build constructor kwargs ────────────────────────────────────
            kw = {"d": direction, **extra}
            if length:
                kw["l"] = length
            if flip:
                kw["flip"] = True
            if reverse:
                kw["reverse"] = True
            _resolve_at(kw, at, anchors)

            element = ElClass(**kw)

            if tox is not None:
                if isinstance(tox, str):
                    if tox in anchors:
                        tx = anchors[tox]
                    elif "." in tox:
                        # Compound fallback: "src.start" → try "_before_src", then "src"
                        base, suffix = tox.rsplit(".", 1)
                        synth = f"_before_{base}"
                        if suffix == "start" and synth in anchors:
                            tx = anchors[synth]
                        elif base in anchors:
                            tx = anchors[base]
                        else:
                            import warnings
                            warnings.warn(f"tox anchor '{tox}' not resolved; skipping.")
                            tx = None
                    else:
                        import warnings
                        warnings.warn(f"tox anchor '{tox}' not found; skipping.")
                        tx = None
                else:
                    tx = tox
                if tx is not None:
                    # Always pass a numeric x-coordinate to schemdraw, not a Point/string
                    if hasattr(tx, 'x'):
                        tx = float(tx.x)
                    elif isinstance(tx, (list, tuple)) and not isinstance(tx, str):
                        tx = float(tx[0])
                    element = element.tox(tx)

            if toy is not None:
                if isinstance(toy, str):
                    if toy in anchors:
                        ty = anchors[toy]
                    elif "." in toy:
                        base, suffix = toy.rsplit(".", 1)
                        synth = f"_before_{base}"
                        if suffix == "start" and synth in anchors:
                            ty = anchors[synth]
                        elif base in anchors:
                            ty = anchors[base]
                        else:
                            import warnings
                            warnings.warn(f"toy anchor '{toy}' not resolved; skipping.")
                            ty = None
                    else:
                        import warnings
                        warnings.warn(f"toy anchor '{toy}' not found; skipping.")
                        ty = None
                else:
                    ty = toy
                if ty is not None:
                    # Always pass a numeric y-coordinate to schemdraw, not a Point/string
                    if hasattr(ty, 'y'):
                        ty = float(ty.y)
                    elif isinstance(ty, (list, tuple)) and not isinstance(ty, str):
                        ty = float(ty[1])
                    element = element.toy(ty)
            if color:
                element = element.color(color)
            if linewidth is not None:
                element = element.linewidth(linewidth)
            if fill is not None:
                element = element.fill(fill)
            if dot:
                element = element.dot()
            if idot:
                element = element.idot()
            if zorder is not None:
                element = element.zorder(zorder)

            if label:
                if label_loc:
                    loc = label_loc
                elif direction in ("right", "left"):
                    loc = "top"
                elif direction == "up":
                    loc = "right"
                else:
                    loc = "right"
                element = element.label(label, loc=loc)
            for lbl in labels:
                ltext = lbl.get("text", "")
                if not ltext:
                    continue
                lkw = {"loc": lbl.get("loc", "top")}
                if lbl.get("color"):
                    lkw["color"] = lbl["color"]
                if lbl.get("fontsize"):
                    lkw["fontsize"] = lbl["fontsize"]
                if lbl.get("rotate"):
                    lkw["rotate"] = lbl["rotate"]
                element = element.label(ltext, **lkw)

            el = d.add(element)

            if name:
                _save_anchors(anchors, name, el)
            if hasattr(el, 'end'):
                anchors["_prev_end"] = el.end
            if hasattr(el, 'start'):
                anchors["_prev_start"] = el.start
            if hasattr(el, "center"):
                anchors["_prev_center"] = el.center

    for ann in spec.get("annotations", []):
        note    = ann.get("note", "")
        at_node = ann.get("at_node", None)
        offset  = ann.get("offset", [0, 0])
        if not note:
            continue
        if at_node and at_node in anchors:
            pt = anchors[at_node]
            x, y = float(pt[0]) + offset[0], float(pt[1]) + offset[1]
        else:
            x, y = offset[0], offset[1]
        ax.text(x, y, note,
                fontsize=ann.get("fontsize", 8),
                color=ann.get("color", "#333333"),
                ha=ann.get("ha", "left"),
                va=ann.get("va", "center"),
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=ann.get("bg", "#fffbe6"),
                          edgecolor=ann.get("border", "#cccccc"),
                          alpha=0.9),
                zorder=5)

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


@app.post("/render-circuit")
async def render_circuit(request: Request):
    body = await request.body()
    try:
        spec = json.loads(body.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Body must be valid JSON.")
    try:
        png_bytes = draw_circuit(spec)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Circuit render failed: {str(e)}")
    return Response(content=png_bytes, media_type="image/png")


# ─────────────────────────────────────────────────────────────────────────────
# Unified /render-diagram endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/render-diagram")
async def render_diagram(request: Request):
    """Accepts the full LLM JSON response and routes to the correct renderer."""
    body = await request.body()
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Body must be valid JSON.")

    payload = normalize_diagram_payload(payload)  # ← repair LLM inconsistencies

    if not payload.get("diagram_needed", True):
        raise HTTPException(status_code=400,
                            detail="diagram_needed is false — no diagram to render.")

    diagram_type = payload.get("diagram_type")
    if not diagram_type:
        raise HTTPException(status_code=400,
                            detail="diagram_type is null or missing.")

    try:
        if diagram_type == "circuit":
            spec = payload.get("circuit_spec")
            if not spec:
                raise HTTPException(status_code=400,
                                    detail="circuit_spec missing for diagram_type 'circuit'.")
            png_bytes = draw_circuit(spec)
        elif diagram_type in ("graph", "waveform", "geometric", "phasor"):
            spec = payload.get("graph_spec")
            if not spec:
                raise HTTPException(status_code=400,
                                    detail=f"graph_spec missing for diagram_type '{diagram_type}'.")
            png_bytes = draw_graph(spec)
        else:
            raise HTTPException(status_code=400,
                                detail=f"Unknown diagram_type: {diagram_type!r}.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Render failed: {str(e)}")

    return Response(content=png_bytes, media_type="image/png")