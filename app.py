import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.set_page_config(
    page_title="SafeEye — Detector PPE",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

/* Reset & Base */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Space Grotesk', sans-serif;
    background: #f5f0e8;
    color: #1a1a1a;
}

[data-testid="stAppViewContainer"] {
    background: #f5f0e8;
}

[data-testid="stSidebar"] {
    background: #1a1a1a !important;
    border-right: none !important;
}

[data-testid="stSidebar"] * {
    color: #f5f0e8 !important;
}

[data-testid="stSidebar"] .stSlider label {
    color: #aaa !important;
}

/* Header */
.header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 2rem 0 1.5rem;
    border-bottom: 2px solid #1a1a1a;
    margin-bottom: 2rem;
}
.header-left { display: flex; align-items: baseline; gap: 1rem; }
.logo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #1a1a1a;
    letter-spacing: -0.03em;
}
.logo span { color: #e85d04; }
.tagline {
    font-size: 0.8rem;
    color: #888;
    font-weight: 400;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.status-pill {
    background: #1a1a1a;
    color: #4ade80;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    padding: 0.4rem 1rem;
    border-radius: 2rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.dot { width: 6px; height: 6px; background: #4ade80; border-radius: 50%; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* Metric cards */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.metric-card {
    background: white;
    border: 1.5px solid #e0d9ce;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, #e85d04);
}
.metric-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #1a1a1a;
    line-height: 1;
}
.metric-sub {
    font-size: 0.72rem;
    color: #aaa;
    margin-top: 0.3rem;
}

/* Section title */
.section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #e0d9ce;
}

/* Detection result cards */
.detection-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.8rem;
    margin-top: 1rem;
}
.det-card {
    background: white;
    border: 1.5px solid #e0d9ce;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.det-emoji { font-size: 1.8rem; }
.det-info { flex: 1; }
.det-name {
    font-weight: 600;
    font-size: 0.9rem;
    color: #1a1a1a;
    text-transform: capitalize;
}
.det-count {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #888;
}
.det-bar-bg {
    height: 4px;
    background: #f0ebe3;
    border-radius: 2px;
    margin-top: 0.5rem;
    overflow: hidden;
}
.det-bar {
    height: 100%;
    border-radius: 2px;
    background: #e85d04;
    transition: width 0.5s ease;
}

/* Model metrics table */
.metrics-table {
    background: white;
    border: 1.5px solid #e0d9ce;
    border-radius: 12px;
    overflow: hidden;
    margin-top: 1rem;
}
.metrics-table-row {
    display: flex;
    align-items: center;
    padding: 0.8rem 1.2rem;
    border-bottom: 1px solid #f0ebe3;
    gap: 1rem;
}
.metrics-table-row:last-child { border-bottom: none; }
.mt-emoji { font-size: 1.2rem; width: 2rem; }
.mt-class { font-weight: 500; font-size: 0.88rem; flex: 1; text-transform: capitalize; }
.mt-bar-wrap { flex: 2; }
.mt-bar-bg { height: 6px; background: #f0ebe3; border-radius: 3px; overflow: hidden; }
.mt-bar { height: 100%; border-radius: 3px; }
.mt-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    color: #1a1a1a;
    width: 3rem;
    text-align: right;
}

/* Upload area */
.upload-area {
    background: white;
    border: 2px dashed #c0b8ae;
    border-radius: 16px;
    padding: 3rem 2rem;
    text-align: center;
    transition: border-color 0.2s;
}

/* Empty state */
.empty-state {
    background: white;
    border: 1.5px solid #e0d9ce;
    border-radius: 16px;
    padding: 4rem 2rem;
    text-align: center;
    color: #bbb;
}
.empty-icon { font-size: 3rem; margin-bottom: 1rem; }
.empty-text { font-size: 1rem; color: #ccc; }

/* Sidebar */
.sidebar-section {
    margin-bottom: 2rem;
}
.sidebar-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #666 !important;
    margin-bottom: 0.8rem;
}

/* Buttons */
.stButton > button {
    background: #1a1a1a !important;
    color: #f5f0e8 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 1.5rem !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: #333 !important; }

/* Camera toggle button */
.cam-btn > button {
    background: #e85d04 !important;
    color: white !important;
}
.cam-btn-off > button {
    background: #555 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #ede8e0 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: none !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #888 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    font-size: 0.88rem !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #1a1a1a !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] div { background: #e85d04 !important; }

/* Hide Streamlit default */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Cargar modelo ─────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    return YOLO("best.pt")

model = cargar_modelo()
clases = model.names

# Métricas del modelo entrenado (según resultados del entrenamiento)
metricas_modelo = {
    "helmet":   {"map50": 0.900, "color": "#e85d04"},
    "vest":     {"map50": 0.909, "color": "#16a34a"},
    "boots":    {"map50": 0.859, "color": "#2563eb"},
    "person":   {"map50": 0.853, "color": "#7c3aed"},
    "glasses":  {"map50": 0.728, "color": "#db2777"},
    "earmuffs": {"map50": 0.612, "color": "#d97706"},
    "gloves":   {"map50": 0.384, "color": "#64748b"},
}

emoji_map = {
    "helmet": "⛑️", "vest": "🦺", "boots": "👟",
    "gloves": "🧤", "glasses": "🥽", "earmuffs": "🎧", "person": "🧍"
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1.5rem 0 1rem;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:700;color:#f5f0e8;letter-spacing:-0.02em;">
            Safe<span style="color:#e85d04">Eye</span>
        </div>
        <div style="font-size:0.7rem;color:#555;text-transform:uppercase;letter-spacing:0.1em;margin-top:0.3rem;">
            Panel de control
        </div>
    </div>
    <hr style="border:none;border-top:1px solid #333;margin-bottom:1.5rem;">
    """, unsafe_allow_html=True)

    st.markdown('<p class="sidebar-label">🎯 Sensibilidad de detección</p>', unsafe_allow_html=True)
    confianza = st.slider("", min_value=0.10, max_value=0.95, value=0.25, step=0.05, label_visibility="collapsed")

    st.markdown('<hr style="border:none;border-top:1px solid #333;margin:1.5rem 0;">', unsafe_allow_html=True)

    st.markdown('<p class="sidebar-label">📊 Rendimiento del modelo</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#111;border-radius:10px;padding:1rem;margin-top:0.5rem;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:1.6rem;font-weight:700;color:#4ade80;">74.9%</div>
        <div style="font-size:0.7rem;color:#555;text-transform:uppercase;letter-spacing:0.1em;margin-top:0.2rem;">mAP50 global</div>
        <div style="margin-top:1rem;font-size:0.7rem;color:#555;">50 épocas · YOLOv8n · GPU T4</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border:none;border-top:1px solid #333;margin:1.5rem 0;">', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label">ℹ️ Acerca de</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem;color:#555;line-height:1.7;">
        Modelo entrenado con dataset PPE Factory de Roboflow.<br>
        Detecta 7 clases de equipos de protección personal.
    </div>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <div class="header-left">
        <div class="logo">Safe<span>Eye</span></div>
        <div class="tagline">Detector de Equipos de Protección Personal</div>
    </div>
    <div class="status-pill">
        <div class="dot"></div>
        MODELO ACTIVO
    </div>
</div>
""", unsafe_allow_html=True)

# ── Layout principal ──────────────────────────────────────────────────────────
col_izq, col_der = st.columns([1.1, 1], gap="large")

with col_izq:
    # Tabs entrada
    tab1, tab2 = st.tabs(["📁  Subir imagen", "📷  Cámara"])

    imagen_input = None
    camara_activa = st.session_state.get("camara_activa", False)

    with tab1:
        archivo = st.file_uploader(
            "Sube una imagen JPG, JPEG o PNG",
            type=["jpg", "jpeg", "png"],
            label_visibility="visible"
        )
        if archivo:
            imagen_input = Image.open(archivo).convert("RGB")

    with tab2:
        # Toggle cámara
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("📷 Activar cámara"):
                st.session_state["camara_activa"] = True
        with col_btn2:
            if st.button("⏹ Desactivar cámara"):
                st.session_state["camara_activa"] = False

        if st.session_state.get("camara_activa", False):
            foto = st.camera_input("", label_visibility="collapsed")
            if foto:
                imagen_input = Image.open(foto).convert("RGB")
        else:
            st.markdown("""
            <div style="background:#f0ebe3;border-radius:12px;padding:2rem;text-align:center;color:#aaa;margin-top:1rem;">
                <div style="font-size:2rem">📷</div>
                <div style="margin-top:0.5rem;font-size:0.9rem;">Cámara desactivada</div>
                <div style="font-size:0.75rem;margin-top:0.3rem;">Presiona "Activar cámara" para comenzar</div>
            </div>
            """, unsafe_allow_html=True)

    if imagen_input:
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Vista previa</div>', unsafe_allow_html=True)
        st.image(imagen_input, use_container_width=True)
        st.markdown('<br>', unsafe_allow_html=True)
        analizar = st.button("🔍  Analizar imagen")
    else:
        analizar = False

    # ── Métricas del modelo ──────────────────────────────────────────────
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Métricas por clase (mAP50)</div>', unsafe_allow_html=True)

    rows_html = ""
    for clase, data in metricas_modelo.items():
        pct = data["map50"]
        color = data["color"]
        emoji = emoji_map.get(clase, "📦")
        bar_w = int(pct * 100)
        rows_html += f"""
        <div class="metrics-table-row">
            <div class="mt-emoji">{emoji}</div>
            <div class="mt-class">{clase}</div>
            <div class="mt-bar-wrap">
                <div class="mt-bar-bg">
                    <div class="mt-bar" style="width:{bar_w}%;background:{color};"></div>
                </div>
            </div>
            <div class="mt-pct">{int(pct*100)}%</div>
        </div>
        """
    st.markdown(f'<div class="metrics-table">{rows_html}</div>', unsafe_allow_html=True)

with col_der:
    if imagen_input is None:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">👁</div>
            <div style="font-size:1rem;color:#bbb;font-weight:500;">Sin imagen activa</div>
            <div style="font-size:0.82rem;color:#ccc;margin-top:0.5rem;">
                Sube una imagen o activa la cámara para comenzar
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif analizar:
        with st.spinner("Procesando..."):
            img_array = np.array(imagen_input)
            resultados = model.predict(source=img_array, conf=confianza, verbose=False)
            img_resultado = resultados[0].plot()
            img_rgb = cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB)

        st.markdown('<div class="section-title">Resultado de detección</div>', unsafe_allow_html=True)
        st.image(img_rgb, use_container_width=True)

        boxes = resultados[0].boxes
        n = len(boxes)

        st.markdown('<br>', unsafe_allow_html=True)

        if n == 0:
            st.warning("⚠️ No se detectó ningún equipo de protección en la imagen.")
        else:
            # Conteo
            conteo = {}
            for box in boxes:
                clase_id = int(box.cls[0])
                nombre = clases[clase_id]
                conf_val = float(box.conf[0])
                if nombre not in conteo:
                    conteo[nombre] = {"count": 0, "confs": []}
                conteo[nombre]["count"] += 1
                conteo[nombre]["confs"].append(conf_val)

            conf_media = sum([b.conf[0].item() for b in boxes]) / n

            # Stats superiores
            st.markdown(f"""
            <div class="metrics-grid" style="grid-template-columns:repeat(3,1fr);">
                <div class="metric-card" style="--accent:#e85d04">
                    <div class="metric-label">Total objetos</div>
                    <div class="metric-value">{n}</div>
                    <div class="metric-sub">detectados</div>
                </div>
                <div class="metric-card" style="--accent:#16a34a">
                    <div class="metric-label">Clases únicas</div>
                    <div class="metric-value">{len(conteo)}</div>
                    <div class="metric-sub">de 7 posibles</div>
                </div>
                <div class="metric-card" style="--accent:#2563eb">
                    <div class="metric-label">Confianza media</div>
                    <div class="metric-value">{int(conf_media*100)}%</div>
                    <div class="metric-sub">promedio</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Detecciones por clase
            st.markdown('<div class="section-title">Objetos detectados</div>', unsafe_allow_html=True)

            det_html = '<div class="detection-grid">'
            for clase, info in conteo.items():
                promedio = sum(info["confs"]) / len(info["confs"])
                emoji = emoji_map.get(clase, "📦")
                bar_w = int(promedio * 100)
                color = metricas_modelo.get(clase, {}).get("color", "#e85d04")
                det_html += f"""
                <div class="det-card">
                    <div class="det-emoji">{emoji}</div>
                    <div class="det-info">
                        <div class="det-name">{clase}</div>
                        <div class="det-count">{info['count']} objeto(s) · {int(promedio*100)}% conf.</div>
                        <div class="det-bar-bg">
                            <div class="det-bar" style="width:{bar_w}%;background:{color};"></div>
                        </div>
                    </div>
                </div>
                """
            det_html += "</div>"
            st.markdown(det_html, unsafe_allow_html=True)

    elif imagen_input:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🔍</div>
            <div style="font-size:1rem;color:#bbb;font-weight:500;">Listo para analizar</div>
            <div style="font-size:0.82rem;color:#ccc;margin-top:0.5rem;">
                Presiona "Analizar imagen" para detectar
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#ccc;font-size:0.75rem;padding:2rem 0 1rem;
border-top:1px solid #e0d9ce;margin-top:3rem;font-family:'JetBrains Mono',monospace;">
    SafeEye · YOLOv8 · mAP50 74.9% · UNAB Digital · IA Avanzada
</div>
""", unsafe_allow_html=True)