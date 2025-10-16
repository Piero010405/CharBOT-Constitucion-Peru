import streamlit as st
import pymongo
import google.generativeai as genai
import os

# =======================
# CONFIGURACIÓN
# =======================

GOOGLE_API_KEY = st.secrets["app"]["GOOGLE_API_KEY"]
MONGODB_URI = st.secrets["app"]["MONGODB_URI"]

if not GOOGLE_API_KEY or not MONGODB_URI:
    st.error("❌ Faltan las variables de entorno GOOGLE_API_KEY o MONGODB_URI")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Conexión a MongoDB Atlas
client = pymongo.MongoClient(MONGODB_URI)
db = client["pdf_embeddings_db"]
collection = db["pdf_vectors"]

# =======================
# FUNCIONES
# =======================

def crear_embedding(texto):
    """Genera embedding de la pregunta"""
    model = "text-embedding-004"
    resp = genai.embed_content(model=model, content=texto)
    return resp["embedding"]

def buscar_similares(embedding, k=5):
    """
    Busca los documentos más similares en MongoDB Atlas.
    Requiere que el índice vectorial haya sido creado desde Atlas UI.
    """
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": 100,
                "limit": k
            }
        },
        {
            "$project": {
                "_id": 0,
                "texto": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    return list(collection.aggregate(pipeline))

def generar_respuesta(pregunta, contextos):
    """Usa Gemini para responder con contexto"""
    modelo = genai.GenerativeModel("gemini-flash-latest")
    contexto = "\n\n".join([c["texto"] for c in contextos])
    prompt = f"""
Eres un asistente experto en derecho constitucional peruano. Usa el siguiente contexto (extraído de la Constitución Política del Perú) para responder la pregunta del usuario.

Contexto:
{contexto}

Pregunta: {pregunta}

Responde de forma clara, objetiva y en español, citando los artículos relevantes cuando corresponda.
"""
    respuesta = modelo.generate_content(prompt)
    return respuesta.text

# =======================
# INTERFAZ STREAMLIT
# =======================

st.set_page_config(
    page_title="Chat Constitución Política del Perú 🇵🇪",
    page_icon="📜",
    layout="centered"
)

# =======================
# ESTILOS CSS ADAPTATIVOS
# =======================
st.markdown("""
<style>
:root {
    --rojo-bandera: #b71c1c;
    --rojo-claro: #fce4e4;
    --rojo-oscuro: #7f0000;
    --gris-suave: #f5f5f5;
}

/* Adaptación automática según tema */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--background-color);
    color: var(--text-color);
    font-family: 'Merriweather', serif;
}

h1, h2, h3 {
    color: var(--rojo-bandera);
}

/* Cuadro informativo */
.info-box {
    background-color: rgba(183, 28, 28, 0.1);
    border: 1px solid var(--rojo-bandera);
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 25px;
}

/* Chat contenedor */
.chat-container {
    border-radius: 12px;
    padding: 15px;
    background-color: rgba(255,255,255,0.05);
    box-shadow: 0 0 8px rgba(0,0,0,0.15);
}

/* Mensajes */
.user-box, .bot-box {
    padding: 10px 15px;
    border-radius: 8px;
    margin-bottom: 10px;
    line-height: 1.6;
}

.user-box {
    background-color: rgba(100, 100, 100, 0.1);
    border-left: 4px solid #616161;
}

.bot-box {
    background-color: rgba(183, 28, 28, 0.1);
    border-left: 4px solid var(--rojo-bandera);
}

/* Texto dentro del chat */
.user-box b, .bot-box b {
    color: var(--text-color);
}

/* Tema oscuro ajustes */
[data-theme="dark"] .info-box {
    background-color: rgba(183, 28, 28, 0.15);
}
[data-theme="dark"] .user-box {
    background-color: rgba(255,255,255,0.05);
}
[data-theme="dark"] .bot-box {
    background-color: rgba(183, 28, 28, 0.2);
}
[data-theme="dark"] h1, [data-theme="dark"] h2 {
    color: #ff6b6b;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =======================
# CABECERA
# =======================
st.title("📜 Asistente Constitucional del Perú")
st.caption("Basado en la **Constitución Política del Perú (1993)**, con embeddings en MongoDB y respuestas generadas por **Gemini** 🇵🇪")

# Cuadro informativo
st.markdown("""
<div class='info-box'>
<b>📖 Fuente:</b> Constitución Política del Perú (1993, con reformas vigentes).  
<b>Propósito:</b> Este asistente responde preguntas legales y ciudadanas basándose en el texto oficial del documento.  
<b>Advertencia:</b> No constituye asesoría jurídica profesional.
</div>
""", unsafe_allow_html=True)

# =======================
# CHAT
# =======================
if "historial" not in st.session_state:
    st.session_state.historial = []

pregunta = st.chat_input("Escribe tu pregunta sobre la Constitución del Perú...")

if pregunta:
    with st.spinner("Buscando respuesta en la Constitución..."):
        emb = crear_embedding(pregunta)
        similares = buscar_similares(emb, k=5)

        if not similares:
            respuesta = "No encontré información relevante en la Constitución."
        else:
            respuesta = generar_respuesta(pregunta, similares)

        st.session_state.historial.append({"rol": "usuario", "texto": pregunta})
        st.session_state.historial.append({"rol": "bot", "texto": respuesta})

# Mostrar historial con estilo
for msg in st.session_state.historial:
    if msg["rol"] == "usuario":
        st.markdown(f"<div class='user-box'><b>👤 Tú:</b><br>{msg['texto']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-box'><b>🤖 Asistente Constitucional:</b><br>{msg['texto']}</div>", unsafe_allow_html=True)
