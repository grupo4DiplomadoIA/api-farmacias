from flask import Flask, request, jsonify
import json
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
import http.client
from bs4 import BeautifulSoup
import urllib.parse
import groq
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough

load_dotenv(override=True)

RADIO_TIERRA_KM = 6378.137
COLLECTION_NAME = 'vademecum'
BATCH_SIZE = 100
MAX_WORKERS = 4
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_PORT = 6333
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
qdrant_client = QdrantClient(host=QDRANT_URL, port=QDRANT_PORT, api_key=QDRANT_API_KEY)
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
groq_client = groq.Groq(api_key=GROQ_API_KEY)
LANGCHAIN_API_KEY = os.environ["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "proyecto_diplomado_testing"

# 0. clasificar intención usuario
def get_classify_user_intent_chain():
    #print(user_message)
    system_message = """Eres un asistente de clasificación de intenciones. Tu tarea es determinar si el mensaje del usuario está relacionado con:
    1. Buscar una farmacia
    2. Solicitar información sobre un medicamento.
    3. Otro tipo de consulta no relacionada con farmacias o medicamentos
    4. recomendacion de medicamento o recomendacion para aliviar dolores o enfermedades
    Responde únicamente con el número correspondiente: 1, 2, 3 o 4."""

    prompt = ChatPromptTemplate.from_messages([("system", system_message), ("human", "Clasifica la siguiente consulta: {query}")])
    model = ChatGroq(temperature=0.2, model_name="llama3-8b-8192", max_tokens=1)
    classify_user_intent_chain = prompt | model | StrOutputParser()
    #res = classify_user_intent_chain.invoke({"human_message": user_message})
    return classify_user_intent_chain

# 1. locales cercanos
@tool
def locales_cercanos_chain(query: str, lat: float, lng: float) -> dict:
    """Obtiene locales cercanos"""
    #locales_cercanos = api_buscar_locales_cercanos(lat, lng)
    #locales_cercanos_turno = api_buscar_locales_turnos(lat, lng)
    respuesta = {
    'Localización': [lat,lng],
    'Farmacias': ["ahumada","cruz verde"],
    'Turno': {"ahumada":"abierta","cruz verde":"cerrada"}
    }
    return respuesta

# 2. buscar_farmaco
@tool
def buscar_farmaco_chain(query: str, lat: float, lng: float) -> dict:
    """Busca información de un farmaco"""
    ####
    # DUMMY PROC
    ###

    final_response = {
        "gpt_response": "Si, buen medicamento",
        "qdrant_results": {"medicamento1":"info1", "medicamento2":"info2"},
        "productos": "fake_product_text"
    }
    return final_response

# 3. handle_other_query
@tool
def handle_other_query_chain(query: str, lat: float, lng: float) -> str:
    """Maneja información general"""
    system_message = """Eres un asistente especializado en información sobre medicamentos y farmacias. 
    Cuando recibas una consulta que no esté directamente relacionada con estos temas, debes:
    1. Reconocer amablemente la consulta del usuario.
    2. Explicar brevemente que tu especialidad es proporcionar información sobre medicamentos y farmacias.
    3. No ofrecer ayuda relacionada con otra area que no sea medicamentos o farmacias.
    Sé conciso y amigable en tu respuesta."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]

    response = groq_client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
        temperature=0.7,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

# 4. especialista
@tool
def especialista_chain(query: str, lat: float, lng: float) -> str:
    """Especialista"""
    return "Lo siento mucho que estés pasando por eso."

# def route(info):
#     if "1" in info["intent_classification"]:
#         return locales_cercanos_chain
#     elif "2" in info["intent_classification"]:
#         return buscar_farmaco_chain
#     elif "4" in info["intent_classification"]:
#         return especialista_chain
#     else:
#         return handle_other_query_chain

def route(info):
    clasificacion = info["intent_classification"]
    if "1" in clasificacion:
        return {"tipo": clasificacion, "data": locales_cercanos_chain.invoke(info)}
    elif "2" in clasificacion:
        return {"tipo": clasificacion, "data": buscar_farmaco_chain.invoke(info)}
    elif "4" in clasificacion:
        return {"tipo": clasificacion, "data": especialista_chain.invoke(info)}
    else:
        return {"tipo": clasificacion, "data": handle_other_query_chain.invoke(info)}

lat = -37.4
lng = -72.3

user_message = "Que locales tengo cerca?" # 1. locales
#user_message = "Dame información sobre el paracetamol" # 2. farmaco
#user_message = "Dime información del chocolate" # 3. other query
#user_message = "Me duele el estomago" # 4. especialista

print(user_message)

classify_user_intent_chain = get_classify_user_intent_chain()

full_chain = {"intent_classification": classify_user_intent_chain, "query": lambda x: x["query"], "lat": lambda x: x["lat"], "lng": lambda x: x["lng"]} | RunnableLambda(route)
response = full_chain.invoke({"query": user_message, "lat":lat,"lng":lng})

print(response)


# Según resultado inicial se routea a que cadena irá después, usando función route
# Son solo runnables

# clasificacion -> 1,2,3,4
#     1->locales_cercanos_chain
#     2->buscar_farmaco_chain
#     3->handle_other_query_chain
#     4->especialista_chain    

# Limitancia que encontré (hasta ahora no logro evitarla): todas los runnables deben tener mismo input (query, lat, lng)
# https://stackoverflow.com/questions/77201295/how-do-i-use-a-routerchain-with-multiple-chains-that-have-different-inputs