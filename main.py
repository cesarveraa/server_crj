from dotenv import load_dotenv
import os
from typing import List
import openai
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.config import AdditionalConfig, Timeout
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import atexit

# --- Cargar variables de entorno ---
load_dotenv()

OPENAI_API_KEY         = os.getenv("OPENAI_API_KEY")
WEAVIATE_URL_DORIA     = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY_DORIA = os.getenv("WEAVIATE_API_KEY")

if not all([OPENAI_API_KEY, WEAVIATE_URL_DORIA, WEAVIATE_API_KEY_DORIA]):
    raise RuntimeError("Faltan variables de entorno necesarias.")

openai.api_key = OPENAI_API_KEY

# --- Conexión a Weaviate Cloud (sin deprecated) ---
client_doria = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL_DORIA,
    auth_credentials=AuthApiKey(WEAVIATE_API_KEY_DORIA),
    headers={"X-OpenAI-Api-Key": OPENAI_API_KEY},
    additional_config=AdditionalConfig(timeout=Timeout(init=5, query=60, insert=120))
)

atexit.register(client_doria.close)

# --- Obtener colección ---
collection_doria = client_doria.collections.get(name="RAGChunk")

# --- FastAPI app ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Funciones Lógicas ---
def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    resp = openai.embeddings.create(input=[text], model=model)
    return resp.data[0].embedding

def search_chunks(query: str, collection, top_k: int = 6) -> List[str]:
    vector = get_embedding(query)
    response = collection.query.near_vector(near_vector=vector, limit=top_k)
    return [obj.properties["text"] for obj in response.objects]

def construir_prompt_doria(query: str, contexto: List[str]) -> str:
    ctx = "\n\n".join(contexto)
    return f"""
Eres Samuel Doria Medina, empresario y político boliviano. Vas a responder la siguiente pregunta como si fueras él, manteniendo su estilo:

- Tono sobrio, directo, emocionalmente patriótico.
- Uso de frases como: "hay que tomar decisiones", "no podemos dejar de luchar", "100 días, carajo".
- Contrasta modelos: "estatismo o emprendimiento", "el Estado obeso y el sector privado anémico".
- Usa cifras, ejemplos económicos y personales.
- Finaliza con fuerza emocional: "¡Viva Bolivia!".

Contexto relevante:
{ctx}

Pregunta:
{query}

Respuesta como Samuel Doria Medina:"""

def responder_como_doria(query: str) -> str:
    contexto = search_chunks(query, collection_doria)
    prompt = construir_prompt_doria(query, contexto)
    respuesta = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )
    return respuesta.choices[0].message.content.strip()

# --- Endpoint ---
@app.post("/api/responder")
async def endpoint_doria(payload: dict):
    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Debes enviar 'query' en el body")
    return {"response": responder_como_doria(query)}
