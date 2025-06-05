from dotenv import load_dotenv
import os
from typing import List
import openai
import weaviate
import requests
from weaviate.auth import AuthApiKey
from weaviate.config import AdditionalConfig, Timeout
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# --- ENV & VARIABLES ---
load_dotenv()

OPENAI_API_KEY         = os.getenv("OPENAI_API_KEY")
WEAVIATE_URL_DORIA     = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY_DORIA = os.getenv("WEAVIATE_API_KEY")

WEAVIATE_URL_DUNN     = os.getenv("WEAVIATE_URL_DUNN")
WEAVIATE_API_KEY_DUNN = os.getenv("WEAVIATE_API_DUNN")

WEAVIATE_URL_CAPITAN     = os.getenv("WEAVIATE_URL_CAPITAN")
WEAVIATE_API_KEY_CAPITAN = os.getenv("WEAVIATE_API_KEY_CAPITAN")

SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
SAMBANOVA_URL     = "https://api.sambanova.ai/v1/chat/completions"
SAMBANOVA_MODEL   = "Meta-Llama-3.1-8B-Instruct"

if not all([OPENAI_API_KEY, WEAVIATE_URL_DORIA, WEAVIATE_API_KEY_DORIA, WEAVIATE_URL_CAPITAN, WEAVIATE_API_KEY_CAPITAN, SAMBANOVA_API_KEY]):
    raise RuntimeError("Faltan variables de entorno.")

openai.api_key = OPENAI_API_KEY

# --- CONEXIONES ---
client_doria = weaviate.connect_to_weaviate_cloud(
    cluster_url       = WEAVIATE_URL_DORIA,
    auth_credentials  = AuthApiKey(api_key=WEAVIATE_API_KEY_DORIA),
    headers           = {"X-OpenAI-Api-Key": OPENAI_API_KEY},
    additional_config = AdditionalConfig(timeout=Timeout(init=5, query=60, insert=120))
)
collection_doria = client_doria.collections.get("RAGChunk")

client_dunn = weaviate.connect_to_weaviate_cloud(
    cluster_url       = WEAVIATE_URL_DUNN,
    auth_credentials  = AuthApiKey(api_key=WEAVIATE_API_KEY_DUNN),
    headers           = {"X-OpenAI-Api-Key": OPENAI_API_KEY},
    additional_config = AdditionalConfig(timeout=Timeout(init=5, query=60, insert=120))
)
collection_dunn = client_doria.collections.get("RAGChunk")

client_capitan = weaviate.connect_to_weaviate_cloud(
    cluster_url       = WEAVIATE_URL_CAPITAN,
    auth_credentials  = AuthApiKey(api_key=WEAVIATE_API_KEY_CAPITAN),
    headers           = {"X-OpenAI-Api-Key": OPENAI_API_KEY},
    additional_config = AdditionalConfig(timeout=Timeout(init=5, query=60, insert=120))
)
collection_capitan = client_capitan.collections.get("RAGChunk")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- FUNCIONES COMUNES ---
def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    resp = openai.embeddings.create(input=[text], model=model)
    return resp.data[0].embedding

def search_chunks(query: str, collection, top_k: int = 6) -> List[str]:
    vector = get_embedding(query)
    response = collection.query.near_vector(near_vector=vector, limit=top_k)
    return [obj.properties["text"] for obj in response.objects]

# --- DORIA ---
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

# --- DUNN ---
def construir_prompt_dunn(query: str, contexto: List[str]) -> str:
    ctx = "\n\n".join(contexto)
    return f"""
Responde como Jaime Dunn de Ávila, candidato presidencial boliviano (NGP) en 2025.

Estilo: tecnocrático, liberal clásico, frontal. Defiende un Estado mínimo, la creación de riqueza, y la lucha contra la corrupción. Usa frases como:
- "Bolivia no es pobre, ha sido empobrecida por un Estado que castiga al que trabaja."
- "La riqueza no se distribuye, se crea."
- "Estado mínimo pero fuerte."

Temas clave:
- Reducción del gasto público
- Liberalización de mercados (hidrocarburos, litio, agroindustria)
- Reforma judicial sin elección popular
- Autonomía regional y descentralización
- Inversión privada y política exterior pro-mercado

Contexto:
{ctx}

Pregunta:
{query}

Respuesta como Jaime Dunn:"""

def responder_como_dunn(query: str) -> str:
    contexto = search_chunks(query, collection_dunn)
    prompt = construir_prompt_dunn(query, contexto)
    respuesta = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )
    return respuesta.choices[0].message.content.strip()

# --- CAPITÁN LARA ---
def construir_prompt_capitan(query: str, contexto: List[str]) -> str:
    contexto_texto = "\n\n".join(contexto)
    return f"""
Eres el Capitán Edman Lara Montaño, exoficial de policía, abogado y líder político boliviano. Responde a la siguiente pregunta como tú mismo, en tu estilo frontal, patriótico y combativo. Usa frases como:

- "Esto es una guerra contra la corrupción", "¡Basta ya de impunidad!", "Soy tu Capitán", "Vamos a limpiar Bolivia".
- Denuncia a los corruptos, apunta al MAS, a los dinosaurios políticos y la vieja oposición.
- Defiende propuestas concretas, habla con pasión y termina con un grito: "¡Que viva Bolivia libre y sin corrupción!".

Contexto relevante:
{contexto_texto}

Pregunta:
{query}

Respuesta como Capitán Edman Lara:""".strip()

def llamar_sambanova(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {SAMBANOVA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": SAMBANOVA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 500
    }
    response = requests.post(SAMBANOVA_URL, json=payload, headers=headers)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Error en la API de SambaNova")
    return response.json()["choices"][0]["message"]["content"].strip()

def responder_como_capitan(query: str) -> str:
    contexto = search_chunks(query, collection_capitan)
    prompt = construir_prompt_capitan(query, contexto)
    return llamar_sambanova(prompt)

# --- ENDPOINTS ---
@app.post("/api/responder")
async def endpoint_doria(payload: dict):
    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Debes enviar 'query' en el body")
    return {"response": responder_como_doria(query)}

@app.post("/api/responder_dunn")
async def endpoint_dunn(payload: dict):
    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Debes enviar 'query' en el body")
    return {"response": responder_como_dunn(query)}

@app.post("/api/responder_capitan")
async def endpoint_capitan(payload: dict):
    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Debes enviar 'query' en el body")
    return {"response": responder_como_capitan(query)}
