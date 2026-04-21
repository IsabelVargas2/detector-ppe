import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

# Configuración de la página
st.set_page_config(
    page_title="Detector de PPE",
    page_icon="🦺",
    layout="centered"
)

# Título
st.title("🦺 Detector de Equipos de Protección Personal")
st.markdown("Sube una imagen para detectar: **cascos, chalecos, botas, guantes, gafas y más**")
st.divider()

# Cargar el modelo
@st.cache_resource
def cargar_modelo():
    modelo = YOLO("best.pt")
    return modelo

with st.spinner("⏳ Cargando modelo..."):
    model = cargar_modelo()
st.success("✅ Modelo cargado correctamente")

st.divider()

# Configuración de confianza
confianza = st.slider(
    "🎯 Umbral de confianza",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Entre más alto, solo muestra detecciones más seguras"
)

st.divider()

# Subir imagen
imagen_subida = st.file_uploader(
    "📸 Sube una imagen",
    type=["jpg", "jpeg", "png"],
    help="Formatos aceptados: JPG, JPEG, PNG"
)

if imagen_subida is not None:
    # Mostrar imagen original
    imagen = Image.open(imagen_subida)
    st.subheader("🖼️ Imagen original")
    st.image(imagen, use_container_width=True)

    # Botón para detectar
    if st.button("🔍 Detectar PPE", type="primary", use_container_width=True):
        with st.spinner("⏳ Analizando imagen..."):

            # Convertir a array numpy
            img_array = np.array(imagen)

            # Hacer la predicción
            resultados = model.predict(
                source=img_array,
                conf=confianza,
                verbose=False
            )

            # Obtener imagen con detecciones
            img_resultado = resultados[0].plot()
            img_resultado_rgb = cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB)

        # Mostrar resultado
        st.subheader("✅ Resultado de la detección")
        st.image(img_resultado_rgb, use_container_width=True)

        # Mostrar estadísticas
        boxes = resultados[0].boxes
        n_detecciones = len(boxes)

        st.divider()
        st.subheader("📊 Estadísticas")

        if n_detecciones == 0:
            st.warning("⚠️ No se detectó ningún equipo de protección")
        else:
            st.success(f"✅ Se encontraron **{n_detecciones}** objeto(s)")

            # Tabla de detecciones
            nombres_clases = model.names
            st.markdown("**Detecciones encontradas:**")

            conteo = {}
            for box in boxes:
                clase_id = int(box.cls[0])
                nombre_clase = nombres_clases[clase_id]
                confianza_det = float(box.conf[0])
                if nombre_clase not in conteo:
                    conteo[nombre_clase] = []
                conteo[nombre_clase].append(round(confianza_det * 100, 1))

            for clase, confianzas in conteo.items():
                promedio = sum(confianzas) / len(confianzas)
                st.write(f"- **{clase}**: {len(confianzas)} detección(es) — confianza promedio: {promedio:.1f}%")

else:
    st.info("👆 Sube una imagen para comenzar")

# Footer
st.divider()
st.markdown(
    "<div style='text-align:center; color:gray;'>Detector de PPE con YOLOv8 — UNAB Digital</div>",
    unsafe_allow_html=True
)