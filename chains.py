"""
Cadenas (chains) para el agente de reflexión de LinkedIn.

Este módulo define los prompts y cadenas de LangChain utilizadas para generar
y evaluar publicaciones de LinkedIn mediante un proceso iterativo de reflexión.
"""
# 1. IMPORTACIONES
#------------------------------------------------------------------------------------------------
# Importamos ChatPromptTemplate para crear plantillas de prompts estructuradas
# y MessagesPlaceholder para insertar el historial de mensajes en el prompt
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Importamos el modelo de lenguaje de OpenAI para ejecutar las cadenas
from langchain_openai import ChatOpenAI

# Si queremos trabajar con modelos en local, importamos el modelo de Ollama
#from langchain_ollama import ChatOllama


# 2. PLANTILLAS DE PROMPT
#------------------------------------------------------------------------------------------------
# Plantilla de prompt para la fase de generación: el modelo crea o mejora publicaciones
generation_prompt = ChatPromptTemplate.from_messages(
    [
        # Mensaje de sistema: define al modelo como asistente que escribe publicaciones
        (
            "system",
            "Eres un asistente de influencer tecnológico de LinkedIn encargado de escribir excelentes publicaciones. "
            "Genera la mejor publicación posible según la petición del usuario. "
            "Si el usuario aporta crítica, responde con una versión revisada de tus intentos anteriores.",
        ),
        # Marcador para el historial de mensajes
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Plantilla de prompt para la fase de reflexión: el modelo actúa como evaluador
# y genera crítica y recomendaciones sobre la publicación del usuario
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        # Mensaje de sistema: define el rol y comportamiento del modelo
        (
            "system",
            "Eres un influencer viral de LinkedIn que evalúa publicaciones. Genera crítica y recomendaciones para la publicación del usuario. "
            "Siempre proporciona recomendaciones detalladas, incluyendo aspectos como longitud, viralidad, estilo, etc.",
        ),
        # Marcador que se reemplaza por el historial de mensajes de la conversación
        # (petición del usuario, borradores previos, críticas, etc.)
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# 3. CREACIÓN DE CADENAS
#------------------------------------------------------------------------------------------------
# Instancia del modelo de lenguaje (usa configuración por defecto, típicamente desde .env)
# Reemplazas el texto de ejemplo por la clave que acabas de copiar
# Prueba con este modelo en OpenRouter:
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=600,
    request_timeout=20,  # Falla rápido si no hay respuesta en 20s
    temperature=0.7
)

try:
    res = llm.invoke("Responde 'Conectado a Groq'")
    print("Éxito:", res.content)
except Exception as e:
    print("Detalle del fallo:", e)

# --- Configuración del modelo Ollama si trabajamos en local ---
# Descargar Ollama de https://ollama.com/download/
# Asegúrate de haber ejecutado antes en cmd: ollama pull gpt-oss:20b o cualquier otro modelo que tengas instalado
#llm = ChatOllama(model="gpt-oss:20b")

# Cadena de generación: combina el prompt de generación con el LLM mediante el operador |
# Flujo: generation_prompt formatea los mensajes → llm genera la respuesta
generate_chain = generation_prompt | llm

# Cadena de reflexión: combina el prompt de reflexión con el LLM
# Flujo: reflection_prompt formatea los mensajes → llm genera crítica y recomendaciones
reflect_chain = reflection_prompt | llm
