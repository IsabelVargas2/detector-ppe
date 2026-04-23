import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.set_page_config(
    page_title="PPE Vision",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0f1923;
    color: #e2f4f4;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #0f1923 0%, #0a1f1f 100%);
}
[data-testid="stSidebar"] { background: #0a1515 !important; }

/* Header */
.app-header {
    text-align: center;
    padding: 2.5rem 0 2rem;
}
.app-badge {
    display: inline-block;
    background: rgba(32,201,191,0.12);
    border: 1px solid rgba(32,201,191,0.3);
    color: #20c9bf;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding: 0.35rem 1.1rem;
    border-radius: 99px;
    margin-bottom: 1.2rem;
}
.app-title {
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    font-weight: 800;
    color: #e2f4f4;
    line-height: 1.1;
    margin-bottom: 0.8rem;
    letter-spacing: -0.02em;
}
.app-title span { color: #20c9bf; }
.app-desc {
    color: #5a8a8a;
    font-size: 1rem;
    font-weight: 400;
    max-width: 440px;
    margin: 0 auto;
    line-height: 1.7;
}

/* Section label */
.sec-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #20c9bf;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(32,201,191,0.15);
}

/* Cards */
.info-card {
    background: rgba(32,201,191,0.05);
    border: 1px solid rgba(32,201,191,0.12);
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1rem;
}

/* Stats row */
.stats-row {
    display: flex;
    gap: 0.8rem;
    margin: 1rem 0;
}
.stat-box {
    flex: 1;
    background: rgba(32,201,191,0.07);
    border: 1px solid rgba(32,201,191,0.15);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.stat-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #20c9bf;
    line-height: 1;
}
.stat-lbl {
    font-size: 0.7rem;
    color: #5a8a8a;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.25rem;
}

/* Detection cards */
.det-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.7rem;
    margin-top: 0.5rem;
}
.det-card {
    background: rgba(32,201,191,0.06);
    border: 1px solid rgba(32,201,191,0.12);
    border-radius: 10px;
    padding: 0.9rem 1rem;
}
.det-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.det-name {
    font-weight: 600;
    font-size: 0.88rem;
    color: #e2f4f4;
    text-transform: capitalize;
}
.det-count {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #20c9bf;
    background: rgba(32,201,191,0.1);
    padding: 0.1rem 0.5rem;
    border-radius: 99px;
}
.det-bar-bg {
    height: 3px;
    background: rgba(32,201,191,0.1);
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 0.3rem;
}
.det-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #20c9bf, #5efff7);
    border-radius: 2px;
}
.det-conf {
    font-size: 0.72rem;
    color: #5a8a8a;
}

/* Model metrics */
.model-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid rgba(32,201,191,0.07);
}
.model-row:last-child { border-bottom: none; }
.model-emoji { font-size: 1.1rem; width: 1.8rem; }
.model-name {
    font-size: 0.82rem;
    font-weight: 500;
    color: #b0d0d0;
    width: 5rem;
    text-transform: capitalize;
}
.model-bar-wrap { flex: 1; }
.model-bar-bg {
    height: 4px;
    background: rgba(32,201,191,0.1);
    border-radius: 2px;
    overflow: hidden;
}
.model-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #20c9bf, #5efff7);
    border-radius: 2px;
}
.model-pct {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    color: #20c9bf;
    width: 2.5rem;
    text-align: right;
}

/* Empty state */
.empty-box {
    background: rgba(32,201,191,0.03);
    border: 1px dashed rgba(32,201,191,0.15);
    border-radius: 14px;
    padding: 3.5rem 2rem;
    text-align: center;
}
.empty-icon { font-size: 2.5rem; margin-bottom: 0.8rem; }
.empty-txt { color: #3a6060; font-size: 0.9rem; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #20c9bf, #17a89f) !important;
    color: #0f1923 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 1.5rem !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(32,201,191,0.06) !important;
    border: 1px solid rgba(32,201,191,0.12) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 3px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #5a8a8a !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    border-radius: 8px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(32,201,191,0.15) !important;
    color: #20c9bf !important;
    font-weight: 700 !important;
}

/* Slider */
[data-testid="stSlider"] label { color: #5a8a8a !important; font-size: 0.85rem !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(32,201,191,0.04) !important;
    border: 1px dashed rgba(32,201,191,0.2) !important;
    border-radius: 12px !important;
}

/* Divider */
.divider { border: none; border-top: 1px solid rgba(32,201,191,0.1); margin: 1.5rem 0; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Modelo ────────────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    return YOLO("best.pt")

model = cargar_modelo()
clases = model.names

metricas_modelo = {
    "helmet":   {"map50": 0.900, "emoji": "⛑️"},
    "vest":     {"map50": 0.909, "emoji": "🦺"},
    "boots":    {"map50": 0.859, "emoji": "👟"},
    "person":   {"map50": 0.853, "emoji": "🧍"},
    "glasses":  {"map50": 0.728, "emoji": "🥽"},
    "earmuffs": {"map50": 0.612, "emoji": "🎧"},
    "gloves":   {"map50": 0.384, "emoji": "🧤"},
}

emoji_map = {k: v["emoji"] for k, v in metricas_modelo.items()}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-badge">🦺 IA de Seguridad Industrial</div>
    <div class="app-title">PPE <span>Vision</span></div>
    <div class="app-desc">Detecta equipos de protección personal en imágenes usando YOLOv8 entrenado con más de 9.000 imágenes</div>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
col_izq, col_der = st.columns([1, 1.1], gap="large")

with col_izq:
    st.markdown('<div class="sec-label">⚙️ Configuración</div>', unsafe_allow_html=True)
    confianza = st.slider("Umbral de confianza", 0.10, 0.95, 0.25, 0.05)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">📥 Fuente de imagen</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📁  Subir imagen", "📷  Cámara"])
    imagen_input = None

    with tab1:
        archivo = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if archivo:
            imagen_input = Image.open(archivo).convert("RGB")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶ Activar"):
                st.session_state["cam_on"] = True
        with c2:
            if st.button("■ Apagar"):
                st.session_state["cam_on"] = False

        if st.session_state.get("cam_on", False):
            foto = st.camera_input("", label_visibility="collapsed")
            if foto:
                imagen_input = Image.open(foto).convert("RGB")
        else:
            st.markdown("""
            <div class="empty-box" style="padding:1.5rem;">
                <div class="empty-icon">📷</div>
                <div class="empty-txt">Cámara apagada</div>
            </div>
            """, unsafe_allow_html=True)

    if imagen_input:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">🖼️ Vista previa</div>', unsafe_allow_html=True)
        st.image(imagen_input, use_container_width=True)
        st.markdown('<br>', unsafe_allow_html=True)
        analizar = st.button("🔍 Analizar imagen")
    else:
        analizar = False

with col_der:
    st.markdown('<div class="sec-label">📊 Resultado</div>', unsafe_allow_html=True)

    if imagen_input is None:
        st.markdown("""
        <div class="empty-box">
            <div class="empty-icon">🔍</div>
            <div class="empty-txt">Sube una imagen o activa la cámara para comenzar</div>
        </div>
        """, unsafe_allow_html=True)
    elif analizar:
        with st.spinner("Analizando..."):
            img_array = np.array(imagen_input)
            resultados = model.predict(source=img_array, conf=confianza, verbose=False)
            img_res = resultados[0].plot()
            img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
            st.session_state["img_rgb"] = img_rgb
            boxes = resultados[0].boxes
            conteo = {}
            for box in boxes:
                cid = int(box.cls[0])
                nombre = clases[cid]
                conf_val = float(box.conf[0])
                if nombre not in conteo:
                    conteo[nombre] = {"count": 0, "confs": []}
                conteo[nombre]["count"] += 1
                conteo[nombre]["confs"].append(conf_val)
            st.session_state["conteo"] = conteo
            st.session_state["n_boxes"] = len(boxes)
            st.session_state["conf_avg"] = sum([b.conf[0].item() for b in boxes]) / len(boxes) if boxes else 0

        st.image(st.session_state["img_rgb"], use_container_width=True)
    elif "img_rgb" in st.session_state:
        st.image(st.session_state["img_rgb"], use_container_width=True)
    else:
        st.markdown("""
        <div class="empty-box">
            <div class="empty-icon">👈</div>
            <div class="empty-txt">Presiona "Analizar imagen" para ver el resultado</div>
        </div>
        """, unsafe_allow_html=True)

# ── Sección de resultados abajo ───────────────────────────────────────────────
if "conteo" in st.session_state and st.session_state["conteo"] is not None:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    conteo = st.session_state["conteo"]
    n = st.session_state["n_boxes"]
    conf_avg = st.session_state["conf_avg"]

    col_a, col_b = st.columns([1, 1.5], gap="large")

    with col_a:
        st.markdown('<div class="sec-label">🎯 Detecciones encontradas</div>', unsafe_allow_html=True)

        if n == 0:
            st.warning("⚠️ No se detectó ningún equipo de protección.")
        else:
            st.markdown(f"""
            <div class="stats-row">
                <div class="stat-box">
                    <div class="stat-val">{n}</div>
                    <div class="stat-lbl">Objetos</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val">{len(conteo)}</div>
                    <div class="stat-lbl">Clases</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val">{int(conf_avg*100)}%</div>
                    <div class="stat-lbl">Confianza</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            det_html = '<div class="det-grid">'
            for clase, info in conteo.items():
                promedio = sum(info["confs"]) / len(info["confs"])
                emoji = emoji_map.get(clase, "📦")
                det_html += f"""
                <div class="det-card">
                    <div class="det-top">
                        <div class="det-name">{emoji} {clase}</div>
                        <div class="det-count">×{info['count']}</div>
                    </div>
                    <div class="det-bar-bg">
                        <div class="det-bar-fill" style="width:{int(promedio*100)}%"></div>
                    </div>
                    <div class="det-conf">Confianza: {int(promedio*100)}%</div>
                </div>"""
            det_html += "</div>"
            st.markdown(det_html, unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="sec-label">📈 Rendimiento del modelo por clase</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-card">', unsafe_allow_html=True)

        rows = ""
        for clase, data in metricas_modelo.items():
            bar_w = int(data["map50"] * 100)
            rows += f"""
            <div class="model-row">
                <div class="model-emoji">{data['emoji']}</div>
                <div class="model-name">{clase}</div>
                <div class="model-bar-wrap">
                    <div class="model-bar-bg">
                        <div class="model-bar-fill" style="width:{bar_w}%"></div>
                    </div>
                </div>
                <div class="model-pct">{int(data['map50']*100)}%</div>
            </div>"""

        st.markdown(f"""
        {rows}
        <div style="margin-top:1rem;padding-top:0.8rem;border-top:1px solid rgba(32,201,191,0.1);
        display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:0.75rem;color:#5a8a8a;">mAP50 Global · 50 épocas · YOLOv8n · GPU T4</span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:1rem;
            font-weight:600;color:#20c9bf;">74.9%</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#2a5555;font-size:0.75rem;padding:2rem 0 1rem;
border-top:1px solid rgba(32,201,191,0.08);margin-top:2rem;">
    PPE Vision · YOLOv8 · mAP50 74.9% · UNAB Digital · IA Avanzada
</div>
""", unsafe_allow_html=True)