from dotenv import load_dotenv
import os

from fastapi import FastAPI, HTTPException
import openai
import weaviate
from weaviate.auth import AuthApiKey

# 1) Carga las vars de .env
load_dotenv()

# 2) Obtén credenciales
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
WEAVIATE_URL     = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

if not all([OPENAI_API_KEY, WEAVIATE_URL, WEAVIATE_API_KEY]):
    raise RuntimeError(
        "Faltan variables de entorno: OPENAI_API_KEY, WEAVIATE_URL o WEAVIATE_API_KEY"
    )

openai.api_key = OPENAI_API_KEY

# 3) Conéctate a tu instancia Weaviate Cloud con el helper v4
client = weaviate.connect_to_weaviate_cloud(
    cluster_url    = WEAVIATE_URL,
    auth_credentials = AuthApiKey(api_key=WEAVIATE_API_KEY),
    headers        = {"X-OpenAI-Api-Key": OPENAI_API_KEY},
)  # :contentReference[oaicite:0]{index=0}

app = FastAPI()

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    resp = openai.Embedding.create(input=[text], model=model)
    return resp["data"][0]["embedding"]

def search_similar_chunks(query: str, top_k: int = 5) -> list[str]:
    vector = get_embedding(query)
    result = (
        client.query
              .get("RAGChunk", ["text"])
              .with_near_vector({"vector": vector})
              .with_limit(top_k)
              .do()
    )
    return [obj["text"] for obj in result["data"]["Get"]["RAGChunk"]]

def construir_prompt_maestro(query: str, contexto: list[str]) -> str:
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
    chat     = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500,
    )
    return chat.choices[0].message.content.strip()

@app.post("/api/responder")
async def responder_endpoint(payload: dict):
    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Debes enviar 'query' en el body")
    return {"response": responder_como_samuel(query)}
