import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="PPE Vision",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS personalizado ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f;
    color: #f0f0f0;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 0%, #1a1a3e 0%, #0a0a0f 60%);
}

.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #ff6b35, #f7c948);
    color: #0a0a0f;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    padding: 0.3rem 1rem;
    border-radius: 2rem;
    margin-bottom: 1.2rem;
    text-transform: uppercase;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.5rem, 6vw, 4.5rem);
    font-weight: 800;
    line-height: 1.05;
    margin: 0 0 1rem;
    background: linear-gradient(135deg, #ffffff 0%, #a0a0c0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p {
    color: #888;
    font-size: 1.05rem;
    font-weight: 300;
    max-width: 480px;
    margin: 0 auto 2rem;
    line-height: 1.7;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #ff6b35;
    margin-bottom: 1rem;
}
.stats-row {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0;
}
.stat-box {
    flex: 1;
    background: rgba(255,107,53,0.08);
    border: 1px solid rgba(255,107,53,0.2);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.stat-number {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #ff6b35;
    line-height: 1;
}
.stat-label {
    font-size: 0.75rem;
    color: #666;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.tag {
    display: inline-block;
    background: rgba(247,201,72,0.12);
    border: 1px solid rgba(247,201,72,0.3);
    color: #f7c948;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 0.3rem 0.8rem;
    border-radius: 2rem;
    margin: 0.2rem;
}
.tag-high {
    background: rgba(52,211,153,0.12);
    border-color: rgba(52,211,153,0.3);
    color: #34d399;
}
.tag-low {
    background: rgba(251,113,133,0.12);
    border-color: rgba(251,113,133,0.3);
    color: #fb7185;
}
.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 2rem 0;
}
.stButton > button {
    background: linear-gradient(135deg, #ff6b35, #f7c948) !important;
    color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #888 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(255,107,53,0.2) !important;
    color: #ff6b35 !important;
}
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🦺 IA de Seguridad Industrial</div>
    <h1>PPE Vision</h1>
    <p>Detecta equipos de protección personal en tiempo real usando inteligencia artificial avanzada con YOLOv8</p>
</div>
""", unsafe_allow_html=True)

# ── Cargar modelo ─────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    return YOLO("best.pt")

with st.spinner("Iniciando motor de IA..."):
    model = cargar_modelo()

clases = model.names

# ── Layout ────────────────────────────────────────────────────────────────────
col_izq, col_der = st.columns([1, 1], gap="large")

with col_izq:
    st.markdown('<div class="card-title">⚙️ Configuración</div>', unsafe_allow_html=True)

    confianza = st.slider(
        "Umbral de confianza",
        min_value=0.1, max_value=1.0,
        value=0.25, step=0.05
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📥 Fuente de imagen</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📁 Subir imagen", "📷 Usar cámara"])

    imagen_input = None

    with tab1:
        archivo = st.file_uploader(
            "Arrastra o selecciona",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if archivo:
            imagen_input = Image.open(archivo).convert("RGB")

    with tab2:
        st.markdown("Toma una foto con tu cámara:")
        foto = st.camera_input("", label_visibility="collapsed")
        if foto:
            imagen_input = Image.open(foto).convert("RGB")

    if imagen_input:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🖼️ Imagen original</div>', unsafe_allow_html=True)
        st.image(imagen_input, use_container_width=True)
        detectar = st.button("🔍 Analizar con IA")
    else:
        detectar = False

with col_der:
    st.markdown('<div class="card-title">📊 Resultados del análisis</div>', unsafe_allow_html=True)

    if imagen_input is None:
        st.markdown("""
        <div style="height:400px;display:flex;flex-direction:column;align-items:center;
        justify-content:center;background:rgba(255,255,255,0.02);border:1px dashed
        rgba(255,255,255,0.08);border-radius:16px;color:#444;text-align:center;padding:2rem;">
            <div style="font-size:3rem;margin-bottom:1rem">🔍</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;color:#555">
                Sube una imagen o activa la cámara
            </div>
            <div style="font-size:0.85rem;margin-top:0.5rem;color:#333">para comenzar el análisis</div>
        </div>
        """, unsafe_allow_html=True)

    elif detectar:
        with st.spinner("Analizando con IA..."):
            img_array = np.array(imagen_input)
            resultados = model.predict(source=img_array, conf=confianza, verbose=False)
            img_resultado = resultados[0].plot()
            img_rgb = cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB)

        st.image(img_rgb, use_container_width=True)

        boxes = resultados[0].boxes
        n = len(boxes)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        if n == 0:
            st.warning("⚠️ No se detectó ningún equipo de protección personal")
        else:
            conteo = {}
            for box in boxes:
                clase_id = int(box.cls[0])
                nombre = clases[clase_id]
                conf_val = float(box.conf[0])
                if nombre not in conteo:
                    conteo[nombre] = []
                conteo[nombre].append(conf_val)

            conf_media = sum([b.conf[0].item() for b in boxes]) / n

            st.markdown(f"""
            <div class="stats-row">
                <div class="stat-box">
                    <div class="stat-number">{n}</div>
                    <div class="stat-label">Detecciones</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{len(conteo)}</div>
                    <div class="stat-label">Clases</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{round(conf_media*100)}%</div>
                    <div class="stat-label">Confianza</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="card-title">🏷️ Objetos detectados</div>', unsafe_allow_html=True)

            emoji_map = {
                "helmet": "⛑️", "vest": "🦺", "boots": "👟",
                "gloves": "🧤", "glasses": "🥽", "earmuffs": "🎧", "person": "🧍"
            }
            tags_html = ""
            for clase, confs in conteo.items():
                promedio = sum(confs) / len(confs)
                tag_class = "tag-high" if promedio > 0.6 else ("tag" if promedio > 0.4 else "tag-low")
                emoji = emoji_map.get(clase, "📦")
                tags_html += f'<span class="{tag_class}">{emoji} {clase} ×{len(confs)} — {promedio*100:.0f}%</span> '

            st.markdown(tags_html, unsafe_allow_html=True)

    elif imagen_input:
        st.markdown("""
        <div style="height:300px;display:flex;align-items:center;justify-content:center;
        background:rgba(255,107,53,0.04);border:1px dashed rgba(255,107,53,0.15);
        border-radius:16px;color:#555;text-align:center;">
            <div>
                <div style="font-size:2rem">👈</div>
                <div style="margin-top:0.5rem;font-family:'Syne',sans-serif;">
                    Presiona "Analizar con IA"
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#333;font-size:0.8rem;padding-bottom:2rem;">
    PPE Vision · YOLOv8 · Inteligencia Artificial Avanzada · UNAB Digital
</div>
""", unsafe_allow_html=True)