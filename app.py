import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.set_page_config(
    page_title="SafeEye // PPE Scanner",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0d0015;
    color: #e0d0ff;
    font-family: 'Rajdhani', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0d0015 0%, #0a0020 50%, #0d0015 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #120025 0%, #0a0018 100%) !important;
    border-right: 1px solid #9d4edd44 !important;
}
[data-testid="stSidebar"] * { color: #e0d0ff !important; }

/* Scanline overlay effect */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(157, 78, 221, 0.015) 2px,
        rgba(157, 78, 221, 0.015) 4px
    );
    pointer-events: none;
    z-index: 0;
}

/* Header */
.cyber-header {
    padding: 1.5rem 0 1rem;
    border-bottom: 1px solid #9d4edd44;
    margin-bottom: 1.5rem;
    position: relative;
}
.cyber-header::after {
    content: '';
    position: absolute;
    bottom: -2px; left: 0;
    width: 200px; height: 2px;
    background: linear-gradient(90deg, #c77dff, transparent);
}
.cyber-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 900;
    color: #c77dff;
    letter-spacing: 0.08em;
    text-shadow: 0 0 20px #9d4edd88, 0 0 40px #9d4edd44;
    line-height: 1;
}
.cyber-title span { color: #e040fb; }
.cyber-sub {
    font-size: 0.72rem;
    color: #9d4edd;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}
.cyber-status {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(157,78,221,0.1);
    border: 1px solid #9d4edd55;
    color: #c77dff;
    font-family: 'Orbitron', monospace;
    font-size: 0.6rem;
    padding: 0.3rem 0.8rem;
    border-radius: 2px;
    letter-spacing: 0.15em;
    float: right;
    margin-top: 0.5rem;
}
.blink { animation: blink 1s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

/* Panel cards */
.cyber-card {
    background: rgba(157,78,221,0.06);
    border: 1px solid #9d4edd33;
    border-radius: 4px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.cyber-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #c77dff, #e040fb);
}
.cyber-card-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.6rem;
    color: #9d4edd;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}

/* Metric row */
.metric-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0;
    border-bottom: 1px solid #9d4edd22;
}
.metric-row:last-child { border-bottom: none; }
.metric-name {
    font-size: 0.85rem;
    font-weight: 500;
    color: #c77dff;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.metric-bar-wrap { flex: 1; margin: 0 1rem; }
.metric-bar-bg {
    height: 3px;
    background: rgba(157,78,221,0.15);
    border-radius: 2px;
    overflow: hidden;
}
.metric-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, #9d4edd, #e040fb);
    box-shadow: 0 0 8px #c77dff88;
}
.metric-pct {
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem;
    color: #e040fb;
    min-width: 2.5rem;
    text-align: right;
}

/* Detection items */
.det-item {
    background: rgba(157,78,221,0.08);
    border: 1px solid #9d4edd33;
    border-left: 3px solid #e040fb;
    border-radius: 2px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
}
.det-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
}
.det-name {
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem;
    color: #c77dff;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.det-count {
    background: rgba(224,64,251,0.15);
    border: 1px solid #e040fb44;
    color: #e040fb;
    font-family: 'Orbitron', monospace;
    font-size: 0.6rem;
    padding: 0.15rem 0.5rem;
    border-radius: 2px;
}
.det-conf-bar-bg {
    height: 2px;
    background: rgba(157,78,221,0.15);
    border-radius: 1px;
    overflow: hidden;
}
.det-conf-bar {
    height: 100%;
    background: linear-gradient(90deg, #9d4edd, #e040fb);
    box-shadow: 0 0 6px #c77dff;
}
.det-conf-text {
    font-size: 0.75rem;
    color: #9d4edd;
    margin-top: 0.3rem;
}

/* Stats */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.5rem;
    margin-bottom: 1rem;
}
.stat-item {
    background: rgba(157,78,221,0.08);
    border: 1px solid #9d4edd33;
    border-radius: 4px;
    padding: 0.8rem;
    text-align: center;
}
.stat-val {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem;
    font-weight: 900;
    color: #e040fb;
    text-shadow: 0 0 15px #e040fb88;
    line-height: 1;
}
.stat-lbl {
    font-size: 0.62rem;
    color: #9d4edd;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

/* Empty state */
.empty-cyber {
    border: 1px dashed #9d4edd44;
    border-radius: 4px;
    padding: 3rem 1rem;
    text-align: center;
    color: #9d4edd55;
}
.empty-icon {
    font-size: 2.5rem;
    margin-bottom: 0.8rem;
    filter: grayscale(0.5);
}
.empty-txt {
    font-family: 'Orbitron', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #9d4edd55;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7b2fbe, #9d4edd) !important;
    color: #fff !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.12em !important;
    border: 1px solid #c77dff44 !important;
    border-radius: 2px !important;
    padding: 0.7rem 1.5rem !important;
    width: 100% !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 15px #9d4edd44 !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #9d4edd, #c77dff) !important;
    box-shadow: 0 0 25px #9d4edd88 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(157,78,221,0.08) !important;
    border: 1px solid #9d4edd33 !important;
    border-radius: 2px !important;
    padding: 3px !important;
    gap: 3px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #9d4edd !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 0.1em !important;
    border-radius: 2px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(157,78,221,0.2) !important;
    color: #c77dff !important;
    box-shadow: 0 0 10px #9d4edd44 !important;
}

/* Slider */
[data-testid="stSlider"] [role="slider"] { background: #c77dff !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(157,78,221,0.05) !important;
    border: 1px dashed #9d4edd44 !important;
    border-radius: 4px !important;
}

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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem;">
        <div style="font-family:'Orbitron',monospace;font-size:1.1rem;font-weight:900;
        color:#c77dff;letter-spacing:0.1em;text-shadow:0 0 15px #9d4edd88;">
            SAFE<span style="color:#e040fb">EYE</span>
        </div>
        <div style="font-size:0.6rem;color:#9d4edd;letter-spacing:0.2em;
        text-transform:uppercase;margin-top:0.2rem;">
            PPE SCANNER v2.0
        </div>
    </div>
    <hr style="border:none;border-top:1px solid #9d4edd33;margin:1rem 0;">
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'Orbitron',monospace;font-size:0.58rem;color:#9d4edd;
    letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.5rem;">
        ⬡ Sensibilidad
    </div>
    """, unsafe_allow_html=True)
    confianza = st.slider("", 0.10, 0.95, 0.25, 0.05, label_visibility="collapsed")

    st.markdown('<hr style="border:none;border-top:1px solid #9d4edd33;margin:1rem 0;">', unsafe_allow_html=True)

    # Métricas del modelo
    st.markdown("""
    <div style="font-family:'Orbitron',monospace;font-size:0.58rem;color:#9d4edd;
    letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.8rem;">
        ⬡ Rendimiento del modelo
    </div>
    <div style="background:rgba(157,78,221,0.08);border:1px solid #9d4edd33;
    border-radius:4px;padding:1rem;margin-bottom:1rem;text-align:center;">
        <div style="font-family:'Orbitron',monospace;font-size:2rem;font-weight:900;
        color:#e040fb;text-shadow:0 0 20px #e040fb88;">74.9%</div>
        <div style="font-size:0.6rem;color:#9d4edd;letter-spacing:0.15em;
        text-transform:uppercase;margin-top:0.3rem;">mAP50 Global</div>
        <div style="font-size:0.65rem;color:#9d4edd66;margin-top:0.5rem;">
            50 épocas · YOLOv8n · GPU T4
        </div>
    </div>
    """, unsafe_allow_html=True)

    for clase, data in metricas_modelo.items():
        pct = data["map50"]
        bar_w = int(pct * 100)
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-name">{data['emoji']} {clase}</div>
            <div class="metric-bar-wrap">
                <div class="metric-bar-bg">
                    <div class="metric-bar-fill" style="width:{bar_w}%"></div>
                </div>
            </div>
            <div class="metric-pct">{int(pct*100)}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr style="border:none;border-top:1px solid #9d4edd33;margin:1rem 0;">', unsafe_allow_html=True)

    # Resultados de detección en sidebar
    st.markdown("""
    <div style="font-family:'Orbitron',monospace;font-size:0.58rem;color:#9d4edd;
    letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.8rem;">
        ⬡ Detecciones
    </div>
    """, unsafe_allow_html=True)

    if "resultado_deteccion" not in st.session_state:
        st.markdown("""
        <div class="empty-cyber" style="padding:1.5rem;">
            <div class="empty-txt">Sin datos aún</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        conteo = st.session_state["resultado_deteccion"]
        n_total = sum(v["count"] for v in conteo.values())
        n_clases = len(conteo)
        conf_avg = st.session_state.get("conf_avg", 0)

        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-item">
                <div class="stat-val">{n_total}</div>
                <div class="stat-lbl">Total</div>
            </div>
            <div class="stat-item">
                <div class="stat-val">{n_clases}</div>
                <div class="stat-lbl">Clases</div>
            </div>
            <div class="stat-item">
                <div class="stat-val">{int(conf_avg*100)}%</div>
                <div class="stat-lbl">Conf.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        for clase, info in conteo.items():
            promedio = sum(info["confs"]) / len(info["confs"])
            emoji = metricas_modelo.get(clase, {}).get("emoji", "📦")
            bar_w = int(promedio * 100)
            st.markdown(f"""
            <div class="det-item">
                <div class="det-header">
                    <div class="det-name">{emoji} {clase}</div>
                    <div class="det-count">×{info['count']}</div>
                </div>
                <div class="det-conf-bar-bg">
                    <div class="det-conf-bar" style="width:{bar_w}%"></div>
                </div>
                <div class="det-conf-text">Confianza: {int(promedio*100)}%</div>
            </div>
            """, unsafe_allow_html=True)

# ── Header principal ──────────────────────────────────────────────────────────
st.markdown("""
<div class="cyber-header">
    <span class="cyber-status"><span class="blink">▮</span> SISTEMA ACTIVO</span>
    <div class="cyber-title">SAFE<span>EYE</span></div>
    <div class="cyber-sub">// Detector de Equipos de Protección Personal · YOLOv8 Neural Network</div>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
col_izq, col_der = st.columns([1, 1.2], gap="large")

with col_izq:
    tab1, tab2 = st.tabs(["📁  ARCHIVO", "📷  CÁMARA"])

    imagen_input = None

    with tab1:
        archivo = st.file_uploader(
            "Selecciona imagen",
            type=["jpg", "jpeg", "png"],
            label_visibility="visible"
        )
        if archivo:
            imagen_input = Image.open(archivo).convert("RGB")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶ ACTIVAR"):
                st.session_state["cam_on"] = True
        with c2:
            if st.button("■ DETENER"):
                st.session_state["cam_on"] = False

        if st.session_state.get("cam_on", False):
            foto = st.camera_input("", label_visibility="collapsed")
            if foto:
                imagen_input = Image.open(foto).convert("RGB")
        else:
            st.markdown("""
            <div class="empty-cyber" style="margin-top:0.8rem;">
                <div class="empty-icon">📷</div>
                <div class="empty-txt">Cámara offline</div>
            </div>
            """, unsafe_allow_html=True)

    if imagen_input:
        st.markdown('<br>', unsafe_allow_html=True)
        st.image(imagen_input, use_container_width=True)
        st.markdown('<br>', unsafe_allow_html=True)
        if st.button("⬡ INICIAR ESCANEO"):
            with st.spinner("Procesando señal..."):
                img_array = np.array(imagen_input)
                resultados = model.predict(source=img_array, conf=confianza, verbose=False)
                img_resultado = resultados[0].plot()
                img_rgb = cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB)

                boxes = resultados[0].boxes
                conteo = {}
                for box in boxes:
                    clase_id = int(box.cls[0])
                    nombre = clases[clase_id]
                    conf_val = float(box.conf[0])
                    if nombre not in conteo:
                        conteo[nombre] = {"count": 0, "confs": []}
                    conteo[nombre]["count"] += 1
                    conteo[nombre]["confs"].append(conf_val)

                n = len(boxes)
                conf_avg = sum([b.conf[0].item() for b in boxes]) / n if n > 0 else 0

                st.session_state["img_resultado"] = img_rgb
                st.session_state["resultado_deteccion"] = conteo
                st.session_state["conf_avg"] = conf_avg
            st.rerun()

with col_der:
    st.markdown("""
    <div style="font-family:'Orbitron',monospace;font-size:0.58rem;color:#9d4edd;
    letter-spacing:0.2em;text-transform:uppercase;margin-bottom:1rem;
    border-bottom:1px solid #9d4edd33;padding-bottom:0.5rem;">
        ⬡ Output visual
    </div>
    """, unsafe_allow_html=True)

    if "img_resultado" in st.session_state:
        st.image(st.session_state["img_resultado"], use_container_width=True)
        if st.session_state.get("resultado_deteccion"):
            n = sum(v["count"] for v in st.session_state["resultado_deteccion"].values())
            if n == 0:
                st.warning("Sin detecciones en este frame.")
            else:
                st.success(f"✓ {n} objeto(s) detectado(s) — revisa el panel lateral")
    else:
        st.markdown("""
        <div class="empty-cyber" style="height:400px;display:flex;flex-direction:column;
        align-items:center;justify-content:center;">
            <div class="empty-icon">⬡</div>
            <div class="empty-txt">Esperando señal de entrada...</div>
            <div style="font-size:0.65rem;color:#9d4edd33;margin-top:0.5rem;
            font-family:'Orbitron',monospace;letter-spacing:0.1em;">
                CARGA UNA IMAGEN O ACTIVA LA CÁMARA
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#9d4edd33;font-size:0.65rem;padding:2rem 0 1rem;
border-top:1px solid #9d4edd22;margin-top:3rem;font-family:'Orbitron',monospace;
letter-spacing:0.15em;">
    SAFEEYE · YOLOV8 · MAP50 74.9% · UNAB DIGITAL · IA AVANZADA
</div>
""", unsafe_allow_html=True)