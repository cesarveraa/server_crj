from dotenv import load_dotenv
import os
from typing import List

import openai
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.config import AdditionalConfig, Timeout
from fastapi import FastAPI, HTTPException

# 1) Carga tus vars de .env
load_dotenv()

# 2) Lee credenciales
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
WEAVIATE_URL     = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

if not all([OPENAI_API_KEY, WEAVIATE_URL, WEAVIATE_API_KEY]):
    raise RuntimeError(
        "Faltan variables de entorno: OPENAI_API_KEY, WEAVIATE_URL o WEAVIATE_API_KEY"
    )

# 3) Inicializa OpenAI (v<1.x)
openai.api_key = OPENAI_API_KEY

# 4) Conecta a Weaviate Cloud con timeouts extendidos 
#    (init: 5s, query: 60s, insert: 120s) :contentReference[oaicite:0]{index=0}
client = weaviate.connect_to_weaviate_cloud(
    cluster_url       = WEAVIATE_URL,
    auth_credentials  = AuthApiKey(api_key=WEAVIATE_API_KEY),
    headers           = {"X-OpenAI-Api-Key": OPENAI_API_KEY},
    additional_config = AdditionalConfig(
        timeout=Timeout(init=5, query=60, insert=120)
    )
)

# 5) Obtén tu colección
collection = client.collections.get("RAGChunk")

app = FastAPI()


def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    resp = openai.embeddings.create(input=[text], model=model)
    return resp.data[0].embedding


def search_similar_chunks(query: str, top_k: int = 5) -> List[str]:
    query_vector = get_embedding(query)
    response = collection.query.near_vector(
        near_vector=query_vector,
        limit=top_k
    )
    return [obj.properties["text"] for obj in response.objects]


def construir_prompt_maestro(query: str, contexto: List[str]) -> str:
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

Respuesta como Samuel Doria Medina:
"""


def responder_como_samuel(query: str) -> str:
    contexto = search_similar_chunks(query, top_k=6)
    prompt   = construir_prompt_maestro(query, contexto)
    respuesta = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )
    return respuesta.choices[0].message.content.strip()


@app.post("/api/responder")
async def responder_endpoint(payload: dict):
    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Debes enviar 'query' en el body")
    return {"response": responder_como_samuel(query)}
