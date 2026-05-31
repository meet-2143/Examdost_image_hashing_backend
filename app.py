from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import Response
from PIL import Image, ImageOps
import imagehash
import clip
import torch
import io


app = FastAPI()

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import io, json, re

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
@app.post("/render-graph")
async def render_graph(request: Request):
    body = await request.body()
    try:
        spec = json.loads(body.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Body must be valid JSON.")
    
    try:
        png_bytes = draw_graph(spec)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph render failed: {str(e)}")
    
    return Response(content=png_bytes, media_type="image/png")


def draw_graph(spec: dict) -> bytes:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    graph_type = spec.get("type", "graph")
    title      = spec.get("title", "")
    xlabel     = spec.get("xlabel", "")
    ylabel     = spec.get("ylabel", "")
    xrange     = spec.get("xrange", [-10, 10])
    yrange     = spec.get("yrange", None)
    curves     = spec.get("curves", [])
    points     = spec.get("points", [])
    regions    = spec.get("regions", [])
    lines      = spec.get("lines", [])
    annotations = spec.get("annotations", [])

    x = np.linspace(xrange[0], xrange[1], 1000)

    # ── Draw each curve ──────────────────────────────────────────────────────
    for curve in curves:
        expr    = curve.get("expr", "")
        label   = curve.get("label", "")
        color   = curve.get("color", "blue")
        style   = curve.get("style", "-")
        try:
            # Safe eval with numpy context
            y = eval(expr, {"x": x, "np": np, "sin": np.sin,
                            "cos": np.cos, "tan": np.tan,
                            "exp": np.exp, "log": np.log,
                            "sqrt": np.sqrt, "pi": np.pi,
                            "abs": np.abs, "e": np.e})
            ax.plot(x, y, linestyle=style, color=color,
                    label=label if label else None, linewidth=2)
        except Exception:
            continue
    # ── Draw geometric shapes ─────────────────────────────────────────────────
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
                lx = shape.get("lx", shape.get("cx", shape.get("x1", 0)))
                ly = shape.get("ly", shape.get("cy", shape.get("y1", 0)))
                ax.text(lx, ly, shape["label"], fontsize=10,
                        ha="center", va="bottom")

        ax.set_aspect("equal")
        ax.autoscale()
        
    # ── Draw waveform presets ─────────────────────────────────────────────────
    if graph_type == "waveform":
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

    # ── Draw vertical/horizontal reference lines ──────────────────────────────
    for line in lines:
        if line.get("axis") == "x":
            ax.axvline(x=line.get("value", 0),
                       color=line.get("color", "gray"),
                       linestyle=line.get("style", "--"),
                       linewidth=1)
        elif line.get("axis") == "y":
            ax.axhline(y=line.get("value", 0),
                       color=line.get("color", "gray"),
                       linestyle=line.get("style", "--"),
                       linewidth=1)

    # ── Draw shaded regions ───────────────────────────────────────────────────
    _eval_ns = {"np": np, "sin": np.sin, "cos": np.cos, "tan": np.tan,
                "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
                "pi": np.pi, "abs": np.abs, "e": np.e}
    for region in regions:
        raw_color = parse_color(region.get("color", "yellow"))
        # If color already encodes alpha (tuple with 4 elements), don't double-apply alpha
        if isinstance(raw_color, tuple) and len(raw_color) == 4:
            fill_color = raw_color[:3]
            fill_alpha = raw_color[3]
        else:
            fill_color = raw_color
            fill_alpha = region.get("alpha", 0.2)
        rlabel = region.get("label", "")

        if "x1_expr" in region or "x2_expr" in region:
            # Shade between two x=f(y) curves over a y range
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

    # ── Draw points ───────────────────────────────────────────────────────────
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

    # ── Annotations (arrows with text) ────────────────────────────────────────
    for ann in annotations:
        ax.annotate(ann.get("text", ""),
                    xy=(ann["x"], ann["y"]),
                    xytext=(ann.get("tx", ann["x"] + 1),
                            ann.get("ty", ann["y"] + 1)),
                    arrowprops=dict(arrowstyle="->", color="black"),
                    fontsize=10)

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
    import cairosvg
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

import schemdraw
import schemdraw.elements as elm

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


# Supported element types
ELEMENT_MAP = {
    "resistor":       elm.Resistor,
    "capacitor":      elm.Capacitor,
    "inductor":       elm.Inductor,
    "diode":          elm.Diode,
    "zener":          elm.Zener,
    "led":            elm.LED,
    "battery":        elm.Battery,
    "source_v":       elm.SourceV,
    "source_i":       elm.SourceI,
    "source_sin":     elm.SourceSin,
    "source_controlled_v": elm.SourceControlledV,
    "source_controlled_i": elm.SourceControlledI,
    "ground":         elm.Ground,
    "dot":            elm.Dot,
    "switch":         elm.Switch,
    "transformer":    elm.Transformer,   # mutual coupling
    "opamp":          elm.Opamp,
    "bjt_npn":        elm.BjtNpn,
    "bjt_pnp":        elm.BjtPnp,
    "mosfet_n":       elm.NFet,
    "mosfet_p":       elm.PFet,
    "line":           elm.Line,
    "wire":           elm.Line,          # alias
}

DIRECTION_MAP = {
    "right": "right",
    "left":  "left",
    "up":    "up",
    "down":  "down",
}

def draw_circuit(spec: dict) -> bytes:
    title    = spec.get("title", "")
    elements = spec.get("elements", [])

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor('white')

    with schemdraw.Drawing(canvas=ax) as d:
        d.config(fontsize=11)
        anchors = {}

        for el_spec in elements:
            el_type   = el_spec.get("type", "line").lower()
            direction = DIRECTION_MAP.get(el_spec.get("direction", "right"), "right")
            label     = el_spec.get("label", "")
            length    = el_spec.get("length", None)
            name      = el_spec.get("name", None)
            at        = el_spec.get("at", None)
            label_loc = el_spec.get("label_loc", None)

            # Handle invisible move/reposition
            if el_type in ("move", "jump"):
                kwargs = {"d": direction}
                if length:
                    kwargs["l"] = length
                el = d.add(elm.Line(**kwargs).color("white").linewidth(0))
                if name:
                    anchors[name] = el.end
                anchors["_prev_end"] = el.end
                continue

            ElClass = ELEMENT_MAP.get(el_type)
            if ElClass is None:
                continue

            kwargs = {"d": direction}
            if length:
                kwargs["l"] = length
            if at and at in anchors:
                kwargs["at"] = anchors[at]

            element = ElClass(**kwargs)
            if label:
                if label_loc:
                    element = element.label(label, loc=label_loc)
                elif direction in ("right", "left"):
                    element = element.label(label, loc="top")
                else:
                    element = element.label(label, loc="left")

            el = d.add(element)

            if name:
                anchors[name] = el.end
            anchors["_prev_start"] = el.start
            anchors["_prev_end"]   = el.end

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.read()