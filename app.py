from flask import Flask, request, jsonify
import math
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

from typing import Literal, Dict, Any, List
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langchain.agents import AgentOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import AIMessage
#from langchain.schema import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage

from output_structures import QdrantResult, Producto, BuscarFarmacoStructure, ToolsResponse

load_dotenv(override=True)
app = Flask(__name__)

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
os.environ["LANGCHAIN_PROJECT"] = "proyecto_diplomado"

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def buscar_productos(query):
    if not query:
        return {"error": "Se requiere un parámetro de búsqueda 'query'"}

    conn = http.client.HTTPSConnection("www.farmaciasahumada.cl")
    payload = ""
    headers = {
        'referer': "https://www.farmaciasahumada.cl/",
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    encoded_query = urllib.parse.quote(query)
    path = f"/on/demandware.store/Sites-ahumada-cl-Site/default/SearchServices-GetSuggestions?q={encoded_query}"
    
    try:
        conn.request("GET", path, payload, headers)
        res = conn.getresponse()
        data = res.read().decode("utf-8")

        soup = BeautifulSoup(data, 'html.parser')
        productos = []

        for item in soup.find_all('li', class_='col-12 item mb-3'):
            nombre = item.find('span', class_='name').text.strip() if item.find('span', class_='name') else "N/A"
            url = "https://www.farmaciasahumada.cl" + item.find('a', class_='link')['href'] if item.find('a', class_='link') else "N/A"
            imagen = item.find('img', class_='swatch-circle')['src'] if item.find('img', class_='swatch-circle') else "N/A"
            if imagen != "N/A" and not imagen.startswith('http'):
                imagen = "https://www.farmaciasahumada.cl" + imagen
            precio = item.find('span', class_='value').text.strip() if item.find('span', class_='value') else "N/A"

            productos.append({
                'nombre': nombre,
                'url': url,
                'imagen': imagen,
                'precio': precio
            })

        if not productos:
            print("No se encontraron productos. Contenido HTML:")
            print(data)

        return productos

    except Exception as e:
        return {"error": f"Error al procesar la respuesta del servicio externo: {str(e)}"}

with open('locales.json', encoding="utf8") as f:
    locales = json.load(f)

with open('turnos.json', encoding="utf8") as f:
    turnos = json.load(f)

def distancia(lat1, lng1, lat2, lng2):
    """
    Calcula la distancia entre dos puntos en la superficie de la Tierra.

    :param lat1: Latitud del punto 1
    :param lng1: Longitud del punto 1
    :param lat2: Latitud del punto 2
    :param lng2: Longitud del punto 2
    :return: Distancia entre los dos puntos en kilómetros
    """
    rad = lambda x: (x * math.pi) / 180
    d_lat = rad(lat2 - lat1)
    d_lng = rad(lng2 - lng1)
    a = math.sin(d_lat / 2) * math.sin(d_lat / 2) + math.cos(rad(lat1)) * math.cos(rad(lat2)) * math.sin(d_lng / 2) * math.sin(d_lng / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = RADIO_TIERRA_KM * c
    return distance

def api_buscar_locales_cercanos(lat, lng):
    """
    Busca los locales más cercanos a una ubicación dada.

    :param lat: Latitud de la ubicación
    :param lng: Longitud de la ubicación
    :return: Lista de locales más cercanos
    """
    distancias = []
    for local in locales:
        distancia_local = distancia(lat, lng, local['local_lat'], local['local_lng'])
        distancias.append((local, distancia_local))
    distancias.sort(key=lambda x: x[1])
    return [local[0] for local in distancias[:10]]

def api_buscar_locales_turnos(lat, lng):
    """
    Busca los locales más cercanos de turno a una ubicación dada.

    :param lat: Latitud de la ubicación
    :param lng: Longitud de la ubicación
    :return: Lista de locales más cercanos
    """
    distancias = []
    for local in turnos:
        distancia_local = distancia(lat, lng, local['local_lat'], local['local_lng'])
        distancias.append((local, distancia_local))
    distancias.sort(key=lambda x: x[1])
    return [local[0] for local in distancias[:1]]

def get_gpt4_response(query, qdrant_results):
    system_message = """Eres un experto en farmacología con amplio conocimiento sobre medicamentos. 
    Tu tarea es proporcionar información precisa y útil sobre los medicamentos basándote en la 
    información proporcionada y tu conocimiento general. Asegúrate de incluir detalles sobre 
    usos, dosis, efectos secundarios y precauciones cuando sea relevante de forma breve centrada en la informacion relevante. Si no tienes información 
    suficiente o segura sobre algo, indícalo claramente. Prioriza la seguridad del paciente en 
    tus respuestas. Nunca respondas recomendaciones de medicamentos para un dolor o enfermedad que indique el usuario"""

    context = "Información de la base de datos:\n"
    for result in qdrant_results:
        context += f"- {result['payload']['nombre']} ({result['payload']['farmaco']}):\n"
        for key, value in result['payload'].items():
            if key not in ['nombre', 'farmaco'] and value:
                context += f"  {key}: {value}\n"
        context += "\n"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Contexto: {context}\n\nPregunta del usuario: {query}"}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.2,
        max_tokens=2500
    )

    return response.choices[0].message.content.strip()

def get_groq_response(query, qdrant_results):
    system_message = """Eres un experto en farmacología con un conocimiento profundo sobre medicamentos. 
    Tu tarea es proporcionar información precisa y útil sobre los medicamentos, enfocándote en los siguientes aspectos clave:

    1. Usos principales del medicamento.
    2. Dosis recomendadas, incluyendo variaciones según edad o condición.
    3. Contraindicaciones y efectos secundarios importantes.
    4. Precauciones que deben tomarse antes de su uso.
    5. Formatos y dosificaciones disponibles.

    Si no tienes información suficiente o segura sobre algún aspecto, indícalo de manera breve y clara. Prioriza siempre la seguridad del paciente en tus respuestas. No hagas recomendaciones para tratar dolores o enfermedades específicas."""


    context = "Información relevante:\n"
    for result in qdrant_results[:2]:  # Limitamos a los 2 resultados más relevantes
        context += f"- {result['payload']['nombre']} ({result['payload']['farmaco']}): "
        context += f"{result['payload'].get('indicaciones', 'No hay información de indicaciones disponible.')}\n"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Contexto: {context}\n\nPregunta: {query}\nResponde de forma concisa en no más de 3 oraciones."}
    ]

    response = groq_client.chat.completions.create(
        messages=messages,
        model="llama-3.1-70b-versatile",
        temperature=0.2,
        max_tokens=1500,  # Reducimos el número máximo de tokens
        top_p=0.9,
        frequency_penalty=0.5
    )

    return response.choices[0].message.content.strip()

def get_groq_response2(query, qdrant_results):
    system_message = """Eres un experto en farmacología con amplio conocimiento sobre medicamentos. 
    Tu tarea es proporcionar información precisa y útil sobre los medicamentos basándote en la 
    información proporcionada y tu conocimiento general. Asegúrate de incluir detalles sobre 
    usos, dosis, efectos secundarios y precauciones cuando sea relevante de forma breve centrada en la informacion relevante. Si no tienes información 
    suficiente o segura sobre algo, indícalo claramente. Prioriza la seguridad del paciente en 
    tus respuestas. Nunca respondas recomendaciones de medicamentos para un dolor o enfermedad que indique el usuario"""

    context = "Información de la base de datos:\n"
    for result in qdrant_results:
        context += f"- {result['payload']['nombre']} ({result['payload']['farmaco']}):\n"
        for key, value in result['payload'].items():
            if key not in ['nombre', 'farmaco'] and value:
                context += f"  {key}: {value}\n"
        context += "\n"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Contexto: {context}\n\nPregunta del usuario: {query}"}
    ]

    response = groq_client.chat.completions.create(
        messages=messages,
        model="llama-3.1-70b-versatile",
        temperature=0.2,
        max_tokens=2500
    )
    return response.choices[0].message.content.strip()

# 2. buscar_farmaco
@tool
def buscar_farmaco(query: str) -> dict:
    """Busca información de un farmaco"""
    # TODO: Agregar busqueda mejorada con nuevo embedding
    
    if not query:
        return jsonify({"error": "Se requiere un parámetro de búsqueda 'query'"}), 400
    query_vector = get_embedding(query)
    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=5  
    )
    results = []
    resultsM = []
    for scored_point in search_result:
        result = {
            "id": scored_point.id,
            "score": scored_point.score,
            "payload": scored_point.payload
        }
        resultM = {
            "nombre": scored_point.payload.get('nombre', ''),
            "farmaco": scored_point.payload.get('farmaco', ''),
            "laboratorio": scored_point.payload.get('laboratorio', ''),
            "score": scored_point.score
        }
        results.append(result)
        resultsM.append(resultM)
    #gpt_response = get_gpt4_response(query, results)
    gpt_response = get_groq_response(query, results)
    productos = buscar_productos(resultsM[0].get("nombre", ""))
    final_response = {
        "gpt_response": gpt_response,
        "qdrant_results": resultsM,
        "productos": productos
    }
    print("FINAL_RESPONSE FARMACO:\n",final_response)
    #final_response = {"nombre_medicamento": "Paracetamol", "uso_medicamento": "es un medicamento para el dolor y la inflamación", "valor": "$5.990"}
    #print("final_response buscar_farmaco: ", final_response)
    print("----- END TOOL buscar_farmaco -----")
    return final_response

# 1. locales cercanos
@tool
def locales_cercanos(lat: float, lng: float) -> dict:
    """Obtiene locales cercanos"""
    locales_cercanos = api_buscar_locales_cercanos(lat, lng)
    locales_cercanos_turno = api_buscar_locales_turnos(lat, lng)
    respuesta = {
    'Farmacias': locales_cercanos,
    'Turno': locales_cercanos_turno
    }
    return respuesta

def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together."""
    return first_int * second_int

# No usado
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

# 4. especialista
@tool
def especialista(query: str) -> str:
    """Especialista. Para recomendacion de medicamento o recomendacion para aliviar dolores o enfermedades"""
    system_message = """Eres un asistente de clasificación de especializaciones medicas y debes indicar la especializacion medica adecuada para la consulta del usuario debes responder solo la especialidad medica"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"consulta: {query}"}
    ]
    response = groq_client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
        temperature=0.2,
        max_tokens=200
    )
    especialidad = response.choices[0].message.content.strip()
    resp =  "Lo siento mucho que estés pasando por eso. Mi especialidad es proporcionar información sobre medicamentos y farmacias, no puedo darte consejos sobre medicamentos o recomendaciones, te recomiendo que visites a un especialista en el area de " + especialidad + " que te podria ayudar en tu problema." 
    return resp # response.choices[0].message.content.strip()

# 3. handle_other_query
@tool
def handle_other_query(query: str, lat: float, lng: float) -> str:
    """Maneja consultas generales no relacionadas con farmacias o medicamentos"""
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

# Inherit 'messages' key from MessagesState, which is a list of chat messages
class AgentState(MessagesState):
    # Final structured response from the agent
    final_response: ToolsResponse

# Usado para forma de obtener mensajes revisando todos los pasos del agente
def extract_tool_responses(response):
    """
    Revisa los pasos realizados por el agente. Extrae las respuestas de las herramientas.
    En la llamada a herramientas, un paso es la llamada y el siguiente es la respuesta.
    """
    # TODO: Revisar, me lo generó Claude y no he confirmado si funciona en el 100% de los casos
    tool_responses = []
    messages = response["messages"]

    i = 0
    while i < len(messages):
        msg = messages[i]
        if isinstance(msg, AIMessage) and "tool_calls" in msg.additional_kwargs:
            tool_name = msg.additional_kwargs['tool_calls'][0]['function']['name']
            tool_args = msg.additional_kwargs['tool_calls'][0]['function']['arguments']
            
            # La respuesta de la herramienta debería estar en el siguiente mensaje
            if i + 1 < len(messages):
                tool_response = messages[i + 1].content
                tool_responses.append({"tool_name":tool_name, "tool_response":tool_response})
                i += 2  # Saltar a la siguiente llamada o respuesta
            else:
                i += 1  # Avanzar si no hay respuesta a esta llamada
        else:
            i += 1  # Avanzar si no es una llamada a herramienta

    return tool_responses

system_message = """Eres un asistente especializado en información farmacéutica. Tu tarea es responder consultas de usuarios de manera precisa y útil.

Categorías de consultas:
1. Búsqueda de farmacias cercanas
2. Información sobre medicamentos
3. Consultas generales no relacionadas con farmacias o medicamentos
4. Consultar especialista

Instrucciones:
1. Analiza cuidadosamente la consulta del usuario para identificar la categoría o categorías relevantes.
2. Tienes acceso a herramientas especializadas para cada categoría. Utiliza las que necesites. Puedes usar hasta 3 herramientas por consulta.
3. Considera la ubicación del usuario solo si es relevante para la consulta (ej. búsqueda de farmacias).
4. Procesa la información obtenida de las herramientas y formula una respuesta clara y concisa.
5. Para obtener información, siempre debes usar alguna herramienta, no debes usar tu memoria para responder, debes usar las herramientas.
6. Tu respuesta final no debe incluir directamente las respuestas de las herramientas, pues estas se agregarán automaticamente a tu respuesta. Solo debes sintetizar e introducir lo que se responderá.
7. Para consultas médicas complejas o recomendaciones de tratamiento, sugiere siempre consultar a un profesional de la salud.
8. Si la consulta no está relacionada con farmacias o medicamentos, responde amablemente explicando en que ámbitos puedes ayudarlo.
9. Si la consulta es sobre recomendaciones de medicamentos o dolores, debes sugerir que el usuario se dirija a un especialista. Usar la herramienta especialista es suficiente para esto.

Importante:
- No invoques herramientas que no sean necesarias para la consulta específica.
- Si el usuario te pregunta por más de un medicamento, puedes usar la herramienta de medicamentos varias veces.
- Mantén un tono profesional y empático en tus respuestas.
- Prioriza la precisión y la relevancia de la información proporcionada.
- No incluyas en tu respuesta el resultado de las herramientas, pues estas se agregarán automaticamente a tu respuesta.
- Tu respuesta debe ser máximo 1 o 2 oraciones cortas.

Recuerda: Tu objetivo es proporcionar información útil y confiable sobre farmacias y medicamentos, siempre dentro del ámbito de tu especialización.
Tu objetivo no es dar recomendaciones de medicamentos o dolores, si no solo proporcionar informacion relevante sobre farmacias y medicamentos

Ejemplos de respuesta:

Búsqueda de farmacias cercanas:
Usuario: "¿Hay alguna farmacia abierta cerca de mi ubicación?"
Herramienta: locales_cercanos
Respuesta: "Entiendo que necesitas encontrar una farmacia cercana. Buscaré información de farmacias cercanas para proporcionarte esa información."

Información sobre medicamentos:
Usuario: "¿Qué me puedes decir sobre el paracetamol?"
Herramienta: buscar_farmaco
Respuesta: "Claro, buscar información sobre el paracetamol."

Consulta general no relacionada:
Usuario: "¿Cuál es la capital de Francia?"
Herramienta: handle_other_query
Respuesta: "Entiendo tu curiosidad, pero mi especialidad es proporcionar información sobre farmacias y medicamentos. Para preguntas generales como esta, te sugiero consultar una fuente de información general o un motor de búsqueda."

Consulta sobre dolores o recomendación de medicamentos, derivar a especialista:
Usuario: "Me duele mucho la cabeza, ¿qué me recomiendas tomar?"
Respuesta: "Lamento que estés experimentando dolor de cabeza. Como asistente virtual, no puedo recomendar medicamentos. Buscaré información sobre especialistas en el area de medicina que te puedan ayudar."

Información de medicamentos y donde comprarlos
Usuario: "Dame información sobre el paracetamol y dime que local tengo cerca donde comprarlo"
Herramienta: buscar_farmaco y locales_cercanos
Respuesta: "Entiendo que necesites información sobre medicamentos y farmacias cercanas. Buscaré información relevante sobre medicamentos y proporcionaré información de farmacias cercanas."
"""

model = ChatOpenAI(temperature=0.2, model_name="gpt-4o")
tools = [locales_cercanos, buscar_farmaco, especialista, handle_other_query]
model_with_tools = model.bind_tools(tools)
model_with_structured_output = model.with_structured_output(ToolsResponse)

# Define the function that calls the model
def call_model(state: AgentState):
    response = model_with_tools.invoke(state['messages'])
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define the function that responds to the user
# Nodo que responde al usuario, su input será el historial de mensajes y su outputs tendrá la estructura de ToolsResponse
def respond(state: AgentState):
    # We call the model with structured output in order to return the same format to the user every time
    # Calcular el número de mensajes desde el último mensaje del usuario
    n = 0
    for message in reversed(state['messages']):
        if message.type == "human":
            break
        n += 1
    print("n: ", n)
    # -2 es el penultimo mensaje, que es el de la herramienta
    # En caso de tener más de un mensaje de herramienta, debería considerar todos los mensajes de herramienta. TBD
    messages_to_use = json.dumps({"tool":state['messages'][-2].name, "tool_response":state['messages'][-2].content})
    # response = model_with_structured_output.invoke([HumanMessage(content=state['messages'][-2].content)])
    response = model_with_structured_output.invoke([HumanMessage(content=messages_to_use)])
    # We return the final answer
    #response = state['messages']
    return {"final_response": response}

# Define the function that determines whether to continue or not
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we respond to the user
    if not last_message.tool_calls:
        return "respond"
    # Otherwise if there is, we continue
    else:
        return "continue"

# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between, and the tool node
workflow.add_node("agent", call_model)
workflow.add_node("respond", respond)
workflow.add_node("tools", ToolNode(tools))

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
# Continue para ir a herramientas, respond para responder al usuario
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "respond": "respond",
    },
)

workflow.add_edge("tools", "agent")
workflow.add_edge("respond", END)
runnable_graph = workflow.compile()

# Usado para forma de obtener mensajes revisando todos los pasos del agente
# prompt = ChatPromptTemplate.from_messages([("system", system_message)])
# model = ChatOpenAI(temperature=0.2, model_name="gpt-4o")
# tools = [locales_cercanos, buscar_farmaco, especialista, handle_other_query]

# runnable_graph = create_react_agent(
#     model, 
#     tools=tools, 
#     messages_modifier=system_message
#     #use_history=True,
#     #history_function=historial
# )


@app.route('/chat', methods=['POST'])
def chat():
    # Data desde el frontend
    data = request.json
    user_message = data.get('mensaje')
    lat = data.get('lat')
    lng = data.get('lng')
    if not user_message:
        return jsonify({"error": "Se requiere un mensaje del usuario"}), 400
    
    # Llamada a agente
    # ubicacion = f"\nLa ubicación actual es: lat = {lat} lng = {lng}"
    # print("user_message:",user_message + ubicacion)
    #response = runnable_graph.invoke({"messages": [("user", user_message + ubicacion)]})
    response = runnable_graph.invoke({"messages": [{"role": "user", "content": user_message}]})
    print(response)
    # print("--- LAST RESPONSE ---")
    # print(response["messages"][-1].content)
    # print("--- END LAST RESPONSE ---")
    # print("------------------------------------------------")
    # for i in response["messages"]:
    #     print("--- MESSAGE ---")
    #     print(i)
    # print("------------------------------------------------")
    
    #print("--- TOOL OUTPUTS ---")
    # extraer las respuestas de las herramientas
    # tool_outputs = extract_tool_responses(response)
    # print(tool_outputs)
    # api_response = {
    #     "respuesta": response["messages"][-1].content,
    #     "herramientas": tool_outputs
    # }
    print("--------------------")
    #to_return = jsonpickle.encode(response["final_response"])
    to_return = response["final_response"].dict()
    print(to_return)
    #to_return = json.dumps(response["final_response"].__dict__)
    return to_return # jsonify(api_response) # response_json #jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)