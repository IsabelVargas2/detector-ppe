# 🦺 Detector de PPE con YOLOv8 y Streamlit

Proyecto desarrollado para el curso de **Inteligencia Artificial Avanzada — UNAB Digital**

Aplicación web que detecta equipos de protección personal (PPE) en imágenes usando YOLOv8 y Streamlit.

---

## 📦 Contenido del repositorio

| Archivo | Descripción |
|--------|-------------|
| `app.py` | Aplicación web con Streamlit |
| `best.pt` | Modelo YOLOv8 entrenado |
| `PPE_Detector_Final.ipynb` | Notebook de entrenamiento en Google Colab |

---

## 🎯 Clases detectadas

- 👷 **helmet** — Casco de seguridad
- 🦺 **vest** — Chaleco reflectivo
- 👟 **boots** — Botas de seguridad
- 🧤 **gloves** — Guantes
- 🥽 **glasses** — Gafas de protección
- 🎧 **earmuffs** — Protectores auditivos
- 🧍 **person** — Persona

---

## 🚀 Cómo ejecutar la app

### 1. Clonar el repositorio
```bash
git clone https://github.com/TU_USUARIO/detector-ppe.git
cd detector-ppe
```

### 2. Instalar dependencias
```bash
pip install streamlit ultralytics opencv-python pillow
```

### 3. Ejecutar la app
```bash
streamlit run app.py
```

---

## 🏋️ Entrenamiento del modelo

El modelo fue entrenado en **Google Colab con GPU Tesla T4** usando:
- Dataset: PPE Factory (Roboflow)
- Modelo base: YOLOv8n (Transfer Learning)
- Épocas: 50
- mAP50: 0.749

---

## 📊 Resultados

| Clase | mAP50 |
|-------|-------|
| helmet | 0.900 |
| vest | 0.909 |
| boots | 0.859 |
| person | 0.853 |
| glasses | 0.728 |
| gloves | 0.384 |
| earmuffs | 0.612 |

---

## 🛠️ Tecnologías usadas

- [YOLOv8 - Ultralytics](https://docs.ultralytics.com)
- [Streamlit](https://streamlit.io)
- [Google Colab](https://colab.research.google.com)
- [Roboflow](https://roboflow.com)
