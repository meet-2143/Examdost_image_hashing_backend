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


def _normalize_graph_spec(spec: dict) -> dict:
    """Absorb LLM field-name variations so renderers see canonical keys."""
    spec = dict(spec)
    normalized = []
    for s in spec.get("shapes", []):
        s = dict(s)
        if s.get("type") == "arrow":
            # start/end arrays  (most common LLM variant)
            # from/to arrays    (alternate LLM variant)
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
    # Normalise element type strings for circuit passthrough safety
    return spec


def draw_graph(spec: dict) -> bytes:
    spec = _normalize_graph_spec(spec)
    graph_type = spec.get("type", "graph")
    # Waveform / multi-curve plots need more horizontal room
    default_w = 10 if graph_type == "waveform" else 7
    figw = spec.get("figwidth", default_w)
    figh = spec.get("figheight", 5 if graph_type == "waveform" else 4.5)
    fig, ax = plt.subplots(figsize=(figw, figh), dpi=120)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # graph_type already read above
    title      = spec.get("title", "")
    xlabel     = spec.get("xlabel", "")
    ylabel     = spec.get("ylabel", "")
    spec_has_xrange = "xrange" in spec
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

    # ── Draw phasor diagram ───────────────────────────────────────────────────
    if graph_type == "phasor":
        shapes = spec.get("shapes", [])  # already normalized by _normalize_graph_spec

        # Resolve arrow tip coordinates once (supports polar + cartesian forms)
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

        # x range: use spec value if provided, else derive from arrow data
        if spec_has_xrange:
            ax.set_xlim(xrange[0], xrange[1])
        else:
            xs = ([s.get("x1", 0) for s in resolved if s.get("type") == "arrow"] +
                  [s.get("_x2", 0) for s in resolved if s.get("type") == "arrow"] + [0])
            pad_x = max(abs(min(xs)), abs(max(xs)), 1) * 1.3
            ax.set_xlim(-pad_x, pad_x)

        # y range: let the general yrange block below handle it;
        # only auto-compute when the spec omits yrange entirely
        if not yrange:
            ys = ([s.get("y1", 0) for s in resolved if s.get("type") == "arrow"] +
                  [s.get("_y2", 0) for s in resolved if s.get("type") == "arrow"] + [0])
            pad_y = max(abs(min(ys)), abs(max(ys)), 1) * 1.3
            ax.set_ylim(-pad_y, pad_y)

    # ── Draw waveform presets (only when no custom curves supplied) ───────────
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

    # ── Draw vertical/horizontal reference lines ──────────────────────────────
    for line in lines:
        lcolor = line.get("color", "gray")
        lstyle = line.get("style", "--")
        llabel = line.get("label", "")
        if line.get("axis") == "x":
            xv = line.get("value", 0)
            ax.axvline(x=xv, color=lcolor, linestyle=lstyle, linewidth=1)
            if llabel:
                # Use blended transform: x in data coords, y in axes fraction
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


def _svg_to_png(svg_string: str, width: int = 800) -> bytes:
    """Convert SVG string to PNG bytes.

    Tries cairosvg first (best quality, needs native Cairo library).
    Falls back to svglib → renderPDF → PyMuPDF which is fully pure-Python
    and works on Windows without any system library installation.
    """
    # ── Attempt 1: cairosvg (Linux/Mac/Windows-with-GTK) ──────────────────
    try:
        import cairosvg
        return cairosvg.svg2png(bytestring=svg_string.encode(), output_width=width)
    except (ImportError, OSError):
        pass  # Cairo C library not present on this host

    # ── Attempt 2: svglib → renderPDF → PyMuPDF (no Cairo needed) ─────────
    try:
        import tempfile, os
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPDF
        import fitz  # PyMuPDF

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False,
                                         mode="w", encoding="utf-8") as f:
            f.write(svg_string)
            tmp_path = f.name
        try:
            drawing = svg2rlg(tmp_path)
            if drawing is None:
                raise ValueError("svglib could not parse the SVG document.")

            # Render drawing to PDF bytes (pure Python, no Cairo)
            pdf_buf = io.BytesIO()
            renderPDF.drawToFile(drawing, pdf_buf)
            pdf_buf.seek(0)
            pdf_bytes = pdf_buf.read()

            # Convert first PDF page → PNG at requested width via PyMuPDF
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

# Element pins to auto-expose as named anchors (name.pin)
_ELEMENT_PINS = [
    "start", "end", "center",
    "base", "collector", "emitter",          # BJT
    "gate", "drain", "source",               # FET
    "in1", "in2", "out", "vs", "vd",         # Opamp
    "p1", "p2", "s1", "s2",                  # Transformer
    "tap",                                   # Potentiometer
    "absw",                                  # SwitchSpdt
]


def _resolve_at(kwargs: dict, at, anchors: dict):
    """Resolve `at` string/list into an anchor coordinate and add to kwargs."""
    if at is None:
        return
    if isinstance(at, (list, tuple)):
        kwargs["at"] = at
    elif isinstance(at, str) and at in anchors:
        kwargs["at"] = anchors[at]


def _save_anchors(anchors: dict, name: str, el):
    """Store element end and all named pins under `name` and `name.pin`."""
    anchors[name] = el.end
    for pin in _ELEMENT_PINS:
        val = getattr(el, pin, None)
        if val is not None:
            anchors[f"{name}.{pin}"] = val


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

            # ── Label(s) config ─────────────────────────────────────────────
            label     = el_spec.get("label", "")
            label_loc = el_spec.get("label_loc", None)
            labels    = el_spec.get("labels", [])  # [{text, loc, color, fontsize}]

            # ── Geometry / placement ────────────────────────────────────────
            length    = el_spec.get("length", None)
            at        = el_spec.get("at", None)      # anchor name, "name.pin", or [x,y]
            tox       = el_spec.get("tox", None)     # extend line to x coord / anchor
            toy       = el_spec.get("toy", None)     # extend line to y coord / anchor

            # ── Style ───────────────────────────────────────────────────────
            flip      = el_spec.get("flip", False)
            reverse   = el_spec.get("reverse", False)
            color     = el_spec.get("color", None)
            linewidth = el_spec.get("linewidth", None)
            fill      = el_spec.get("fill", None)
            dot       = el_spec.get("dot", False)    # junction dot at end
            idot      = el_spec.get("idot", False)   # junction dot at start
            zorder    = el_spec.get("zorder", None)

            # ── Pass-through kwargs for element-specific options ─────────────
            extra     = el_spec.get("kwargs", {})

            # ── Invisible reposition (move/jump) ────────────────────────────
            if el_type in ("move", "jump"):
                kw = {"d": direction}
                if length:
                    kw["l"] = length
                _resolve_at(kw, at, anchors)
                el = d.add(elm.Line(**kw).color("white").linewidth(0))
                if name:
                    _save_anchors(anchors, name, el)
                anchors["_prev_end"]    = el.end
                anchors["_prev_start"]  = el.start
                continue

            ElClass = ELEMENT_MAP.get(el_type)
            if ElClass is None:
                continue

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

            # ── Method-chain extensions ─────────────────────────────────────
            if tox is not None:
                tx = anchors[tox] if isinstance(tox, str) and tox in anchors else tox
                element = element.tox(tx)
            if toy is not None:
                ty = anchors[toy] if isinstance(toy, str) and toy in anchors else toy
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

            # ── Apply labels ────────────────────────────────────────────────
            if label:
                if label_loc:
                    loc = label_loc
                elif direction in ("right", "left"):
                    loc = "top"
                elif direction == "up":
                    loc = "right"   # keep label inside circuit, not pushed outside/below
                else:               # down
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
            anchors["_prev_start"] = el.start
            anchors["_prev_end"]   = el.end
            if hasattr(el, "center"):
                anchors["_prev_center"] = el.center

    # ── Circuit annotations (text notes pinned to named anchors) ─────────────
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


@app.post("/render-diagram")
async def render_diagram(request: Request):
    """Accepts the full LLM JSON response and routes to the correct renderer."""
    body = await request.body()
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Body must be valid JSON.")

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