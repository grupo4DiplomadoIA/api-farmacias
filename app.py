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
import requests
import uuid
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
# from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from typing import Literal, Dict, Any, List, Annotated, Union
# from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain.agents import AgentOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import AIMessage
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
import redis
from redis_checkpointer import RedisSaver
import io
from PIL import Image
import torch
from transformers import ViTFeatureExtractor, ViTModel
load_dotenv(override=True)
app = Flask(__name__)

locales_cercanos_resultado = None
buscar_farmaco_resultado = None
buscar_medicos_resultado = None

RADIO_TIERRA_KM = 6378.137
COLLECTION_NAME = 'vademecum'
BATCH_SIZE = 100
MAX_WORKERS = 4
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
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
redis_client = redis.Redis(host='localhost', port=6379, db=0)

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

def get_ciudad(lat,lng):
    locales_cercanos = api_buscar_locales_cercanos(lat,lng)
    return locales_cercanos[0]["comuna_nombre"]

def api_buscar_locales_cercanos(lat, lng, k_cercanos=5):
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
    locales_cercanos = [local[0] for local in distancias[:k_cercanos]]
    return locales_cercanos

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

def get_info_needed(query, info_needed, results):
    system_message = """Eres un experto en farmacología con amplio conocimiento sobre medicamentos y búsqueda de información. 
    Tu tarea es extraer la información requerida por un usuario relacionada a un medicamento.
    Se te entregará el nombre del medicamento, la información requerida y una base de datos de donde buscar la información.
    Responde solo basandote en la información de la base de datos.

    Responde indicando la información requerida por el usuario y el nombre del medicamento relacionado.
    Si no tienes información suficiente o segura sobre algo, indícalo claramente.
    """

    messages = [
        ("system",system_message),
        ("human", f"Medicamento: {query}\n\nInformación requerida: {info_needed}\n\nBase de datos: {results}")
    ]

    model = ChatGroq(temperature=0.2, model_name="llama-3.1-70b-versatile")
    chain = model | StrOutputParser()
    return chain.invoke(messages)

def not_farmaco(query):
    system_message = """Eres un experto en farmacología con amplio conocimiento sobre medicamentos y búsqueda de información.
    Se te entregará el nombre de un medicamento, tu tarea es identificar si ese nombre es realmente un medicamento o no.
    Si no estás muy seguro de si corresponde a un medicamento, es probable que si lo sea.
    Pon mucha atención a si intentan engañarte con un medicamento falso.

    Responde 'S' si es un medicamento
    Responde 'N' si no un medicamento.
    
    No respondas nada más, solo el caracter
    """

    messages = [
        ("system",system_message),
        ("human", f"Medicamento: {query}")
    ]

    model = ChatOpenAI(temperature=0.2, model_name="gpt-4o-mini")
    chain = model | StrOutputParser()
    resp = chain.invoke(messages)
    es_medicamento = resp == 'S'
    return not es_medicamento

@tool
def buscar_farmaco(query: str, info_needed: str = None) -> Union[bool, str]:
    """Busca información de un farmaco.
    query: Nombre del fármaco a buscar
    info_needed: Indica información requerida del fármaco, en caso de especificarse. None en caso de requerir información general.
    
    Retorna True en caso de búsqueda exitosa de información general del medicamento. False en caso de error o medicamento no encontrado.
    Retorna str con información requerida en caso de especificarse en info_needed.
    """
    global buscar_farmaco_resultado
    if not query or not_farmaco(query):
        return jsonify({"error": "Se requiere un parámetro de búsqueda 'query'"}), 400
    query_vector = get_embedding(query)
    search_result = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=5,
        score_threshold = 0.81
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
            "score": scored_point.score,
            "indicaciones": scored_point.payload.get('indicaciones', '')
        }
        results.append(result)
        resultsM.append(resultM)

    # LLM para verificar si se encontró información relevante
    if info_needed:
        return get_info_needed(query, info_needed, results)

    productos = buscar_productos(resultsM[0].get("nombre", ""))
    buscar_farmaco_resultado = {
        "qdrant_results": results, # resultsM
        "productos": productos
    }
    return True

@tool
def locales_cercanos(lat: float, lng: float) -> bool:
    """Obtiene locales cercanos"""
    global locales_cercanos_resultado
    locales_cercanos = api_buscar_locales_cercanos(lat, lng)
    locales_cercanos_turno = api_buscar_locales_turnos(lat, lng)
    locales_cercanos_resultado = {
    'Farmacias': locales_cercanos,
    'Turno': locales_cercanos_turno
    }
    return True

@tool
def especialista(query: str) -> str:
    """Especialista. Para recomendacion de medicamento o recomendacion para aliviar dolores o enfermedades. Encuentra la especialidad medica adecuada para la consulta del usuario"""
    system_message = """Eres un asistente de clasificación de especializaciones medicas y debes indicar la especializacion médica adecuada para la consulta del usuario.
    Responde solo solo la especialidad medica"""

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
    return especialidad # resp

@tool
def buscar_medicos(especialidad: str, ciudad: str) -> bool:
    """Busca medicos según especialidad y ciudad"""
    global buscar_medicos_resultado
    ciudad_encoded = urllib.parse.quote(ciudad)
    url = f"https://www.doctoralia.cl/buscar?q={especialidad}&loc={ciudad_encoded}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Esto levantará una excepción para códigos de estado HTTP erróneos
        soup = BeautifulSoup(response.content, 'html.parser')
    except requests.RequestException as e:
        print(f"Error al hacer la solicitud: {e}")
        return []
    
    medicos = []
    for item in soup.find_all('li'): 
        if len(medicos) >= 3:
            break
        try:
            nombre = item.find('h3', class_='h4').text.strip() if item.find('h3', class_='h4') else "N/A"
            img_tag = item.find('img', itemprop="image")
            img_url = img_tag['src'] if img_tag and 'src' in img_tag.attrs else "N/A"
            especialidades = item.find('h4', class_='h5').text.strip() if item.find('h4', class_='h5') else "N/A"
            direccion = item.find('p', class_='m-0 d-flex align-items-center').text.strip() if item.find('p', class_='m-0 d-flex align-items-center') else "N/A"
            
            # Solo añadir si se encontró al menos un dato válido
            if nombre != "N/A" or especialidades != "N/A" or direccion != "N/A":
                medicos.append({
                    'nombre': nombre,
                    'imagen': img_url,
                    'especialidades': especialidades,
                    'direccion': direccion.replace('\n•\n\nMapa', '')
                })
        except AttributeError as e:
            # Este bloque capturará errores si algún elemento esperado no se encuentra
            print(f"Error al procesar un elemento: {e}")
            continue
    buscar_medicos_resultado = {
        "medicos": medicos,
        "ciudad": ciudad,
        "especialidad": especialidad
    }
    # print("MEDICOS:", medicos)
    # print("----- END TOOL buscar_medicos -----")
    return True

system_message_ia_farma = """
CORE FUNCTIONS:
- Solo se buscará información de productos farmaceuticos, tanto para búsqueda de farmacias como información médica.
- Queda extrictamente prohibida la recomendación médica de fármacos.
- La información de locales cercanos es exclusiva para farmacias. En caso de otro tipo de locales, responde 'No es posible realizar esa búsqueda'.
- Recuerda que hospitales, clínicas y consultorios NO venden medicamentos. Solo las farmacias venden medicamentos. En caso de consultas de este tipo, responde 'No es posible realizar esa búsqueda'.

Eres un asistente especializado en información farmacéutica. Sólo información de productos farmaceuticos.
Tu tarea es responder consultas de usuarios de manera precisa y útil, para ello tienes acceso a herramientas especializadas.
Los resultados de algunas herramientas se mostrarán posteriormente, no tendrás acceso a las respuestas de esas herramientas.

Retorna True en caso de búsqueda exitosa de información general del medicamento. False en caso de error o medicamento no encontrado.
Retorna str con información requerida en caso de especificarse en info_needed.

Categorías de consultas:
1. Búsqueda de farmacias cercanas: Usa la herramienta locales_cercanos. Retorna True si la busqueda es exitosa, False en caso contrario. Solo uso productos farmaceuticos.
2. Información sobre medicamentos: Usa la herramienta buscar_farmaco. Retorna True si la busqueda es exitosa y se mostrará posteriormente, retorna la información necesaria para preguntas específicas.
3. Consultar especialista: Usa la herramienta especialista, retorna el especialista con quien se debería derivar al usuario.
4. Buscar medicos: Usa la herramienta buscar_medicos, retorna una lista de medicos.

Instrucciones:
1. Analiza cuidadosamente la consulta del usuario para identificar la categoría o categorías relevantes.
2. Tienes acceso a herramientas especializadas para cada categoría. Utiliza las que necesites. Siempre indica al usuario que buscaste información si así lo solicitó.
3. Si una búsqueda con una herramienta no es exitosa (respuesta False), no intentes nuevamente, indicale al usuario que no encontraste resultados.
4. Considera la ubicación del usuario solo si es relevante para la consulta (ej. búsqueda de farmacias).
5. Procesa la información obtenida de las herramientas y formula una respuesta clara y concisa.
6. Para obtener información, siempre debes usar alguna herramienta, no debes usar tu memoria para responder, debes usar las herramientas.
7. Tu respuesta final consistirá en una oración para indicar lo que encontraste.
8. Para consultas médicas complejas o recomendaciones de tratamiento, sugiere siempre consultar a un profesional de la salud.
9. Si la consulta no está relacionada con farmacias o medicamentos, responde amablemente explicando que no puedes ayudar en esos temas.
10. Si la consulta es sobre recomendaciones de medicamentos o dolores, debes sugerir que el usuario se dirija a un especialista. Usar la herramienta especialista es suficiente para esto.
11. Si en tu consulta debes derivar a un especialista, busca además médicos en la localidad con la función buscar_medicos.

Importante:
- No invoques herramientas que no sean necesarias para la consulta específica.
- Aunque el usuario te pregunte por algo que ya haya preguntado previamente, vuelve a hacer la búsqueda usando las herramientas que sean necesarias.
- Si el usuario te pregunta por más de un medicamento, puedes usar la herramienta de medicamentos varias veces.
- Mantén un tono profesional y empático en tus respuestas, responde de forma agradable y amable.
- Prioriza la precisión y la relevancia de la información proporcionada.
- Responde siempre en español, a menos que el usuario te indique lo contrario.
- Nunca indiques de forma explícita la ubicación del usuario en tu respuesta final.

Recuerda: Tu objetivo es proporcionar información útil y confiable sobre farmacias y medicamentos, siempre dentro del ámbito de tu especialización.
Tu objetivo no es dar recomendaciones de medicamentos o dolores, si no solo proporcionar informacion relevante sobre farmacias y medicamentos.
Nunca des recomendaciones de medicamentos o dolores más allá de derivar al usuario a un especialista, disculpate por no poder ayudar con sugerencias de medicamentos o dolores.

Ejemplos de respuesta:

Búsqueda de farmacias cercanas:
Usuario: "¿Hay alguna farmacia abierta cerca de mi ubicación?"
Herramienta: locales_cercanos
Respuesta: "Entiendo que necesitas encontrar una farmacia cercana. Buscaré información de farmacias cercanas para proporcionarte esa información."

Información general sobre medicamentos:
Usuario: "¿Qué me puedes decir sobre el paracetamol?"
Herramienta: buscar_farmaco
Respuesta: "Claro, buscaré información sobre el paracetamol."

Información específica sobre medicamentos:
Usuario: "Dime contraindicaciones del paracetamol"
Herramienta: buscar_farmaco(query = 'paracetamol', info_needed = 'Contraindicaciones')
Respuesta: <Respuesta basada en respuesta de tool buscar_farmaco>

Consulta general no relacionada:
Usuario: "¿Cuál es la capital de Francia?"
Herramienta: Ninguna
Respuesta: "Entiendo tu curiosidad, pero mi especialidad es proporcionar información sobre farmacias y medicamentos. Para preguntas generales como esta, te sugiero consultar una fuente de información general o un motor de búsqueda."

Consulta general:
Usuario: "Hola, como estas?"
Herramienta: Ninguna
Respuesta: "Hola, gracias por preguntar. Estoy muy bien. ¿Cómo puedo ayudarte hoy?"

Consulta sobre dolores o recomendación de medicamentos, derivar a especialista y buscar medicos:
Usuario: "Me duele mucho la cabeza, ¿qué me recomiendas tomar?"
Herramienta: especialista y buscar_medicos
Respuesta: "Lamento que estés experimentando dolor de cabeza. Como asistente virtual, no puedo recomendar medicamentos. Te recomiendo consultar con un especialista del área de '<especialidad>'."
"""

# Información de medicamentos y donde comprarlos
# Usuario: "Dame información sobre el paracetamol y dime que local tengo cerca donde comprarlo"
# Herramienta: buscar_farmaco y locales_cercanos
# Respuesta: "Entiendo que necesites información sobre medicamentos y farmacias cercanas. Buscaré información relevante sobre medicamentos y proporcionaré información de farmacias cercanas."

# Inherit 'messages' key from MessagesState, which is a list of chat messages
class AgentState(MessagesState):
    messages: Annotated[list, add_messages]

def create_agent_ia_farma(model, tools = None, checkpointer=None):
    model_with_tools = model.bind_tools(tools)

    # Define the function that calls the model
    def call_model(state: AgentState):
        response = model_with_tools.invoke(state['messages'])
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

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

    # Define the graph
    workflow = StateGraph(AgentState)

    # Define the nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    # Set the entrypoint as `agent`, this means that this node is the first one called
    workflow.set_entry_point("agent")

    # We now add a conditional edge. Continue para ir a herramientas, respond para responder al usuario
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "respond": END,
        },
    )
    workflow.add_edge("tools", "agent")
    runnable_graph = workflow.compile(checkpointer=checkpointer)

    return runnable_graph

def generar_conversation_id():
    """Genera un ID único para la conversación."""
    return str(uuid.uuid4())

def guardar_contexto(conversation_id, role, content):
    """Guarda el mensaje en Redis bajo el ID de la conversación."""
    key = f"chat:{conversation_id}"
    mensaje = json.dumps({"role": role, "content": content})  # Almacenamos el rol y el contenido como JSON
    redis_client.rpush(key, mensaje)  # Agrega el mensaje al final de la lista

def obtener_contexto(conversation_id):
    """Recupera todo el historial de mensajes de la conversación."""
    key = f"chat:{conversation_id}"
    mensajes = redis_client.lrange(key, 0, -1)  # Obtiene todos los mensajes de la lista
    return [json.loads(mensaje.decode('utf-8')) for mensaje in mensajes]  # Decodifica cada mensaje JSON


@app.route('/chat', methods=['POST'])
def chat():
    global buscar_farmaco_resultado, locales_cercanos_resultado, buscar_medicos_resultado
    buscar_farmaco_resultado = None
    locales_cercanos_resultado = None
    buscar_medicos_resultado = None
    
    # Data desde el frontend
    data = request.json
    user_message = data.get('mensaje')
    conversation_id = data.get('conversation_id')
    lat = data.get('lat')
    lng = data.get('lng')
    model_name = data.get('model_name')
    experiment_name = data.get('experiment_name')

    if not user_message:
        return jsonify({"error": "Se requiere un mensaje del usuario"}), 400
    
    if not conversation_id or conversation_id.strip() == "":
        conversation_id = generar_conversation_id()
        ubicacion = f"\nUbicación actual: lat = {lat} lng = {lng}"
        ciudad = f"\nEstoy en la ciudad de: {get_ciudad(lat,lng)}"
        user_message = f"Consulta: {user_message}{ubicacion}{ciudad}"

   
    contexto_prev = obtener_contexto(conversation_id)
  
    if not model_name:
        model = ChatOpenAI(temperature=0.2, model_name="gpt-4o")
    elif model_name == "gpt-4o":
        model = ChatOpenAI(temperature=0.2, model_name="gpt-4o")
    elif model_name == "llama3-8b-8192":
        model = ChatGroq(temperature=0.2, model_name="llama3-8b-8192")
    elif model_name == "llama-3.1-70b-versatile":
        model = ChatGroq(temperature=0.2, model_name="llama-3.1-70b-versatile")
    elif model_name == "gpt-4o-mini":
        model = ChatOpenAI(temperature=0.2, model_name="gpt-4o-mini")
    # elif model_name == "claude-3-sonnet-20240229":
    #     model = ChatAnthropic(temperature=0.2, model_name="claude-3-sonnet-20240229")
    else:
        return jsonify({"error": "Modelo no soportado"}), 400
    
    if not experiment_name: experiment_name = model_name

    contexto_prev.append({"role": "user", "content": user_message})
    messages_input = [
        {"role": "system", "content": system_message_ia_farma}
    ] + contexto_prev  

    # Crear agente
    tools = [locales_cercanos, buscar_farmaco, especialista, buscar_medicos]
   
    
    #with RedisSaver.from_conn_info(host="localhost", port=6379, db=0) as checkpointer:
    #    config = {"configurable": {"thread_id": "00001"}}
    #    latest_checkpoint = checkpointer.get(config)
    #    if latest_checkpoint is not None:
    #        messages_input = messages_input[1:] # Si existe conversación previa, no incluir system_prompt

    #    print("\nmessages_input:", messages_input)
    ia_farma_agent = create_agent_ia_farma(model, tools=tools).with_config({"run_name": experiment_name})
    response = ia_farma_agent.invoke({"messages": messages_input})
    
    guardar_contexto(conversation_id, "user", user_message)
    guardar_contexto(conversation_id, "assistant", response["messages"][-1].content)

    to_return = {
                "conversation_id": conversation_id,
                "respuesta_agente": response["messages"][-1].content,
                "buscar_farmaco_resultado": buscar_farmaco_resultado,
                "locales_cercanos_resultado": locales_cercanos_resultado,
                "buscar_medicos_resultado": buscar_medicos_resultado}
    # print("\nto_return:\n", to_return)
    return jsonify(to_return)

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTModel.from_pretrained('google/vit-base-patch16-224')

def resize_image(image):
    # Implement your image resizing logic here
    return image.resize((224, 224))  # Example: resize to 224x224

def generate_image_embedding(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy().flatten()

@app.route("/search_by_image", methods=['POST'])
def search_by_image():
    global buscar_farmaco_resultado, locales_cercanos_resultado, buscar_medicos_resultado
    buscar_farmaco_resultado = None
    locales_cercanos_resultado = None
    buscar_medicos_resultado = None

    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Get additional parameters
        limit = request.form.get('limit', 1, type=int)
        lat = request.form.get('lat')
        lng = request.form.get('lng')
        model_name = request.form.get('model_name', 'gpt-4o')
        experiment_name = request.form.get('experiment_name', model_name)

        # Process the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = resize_image(image)
        image_embedding = generate_image_embedding(image)

        # Search in Qdrant
        search_result = qdrant_client.search(
            collection_name="imagenes_productos",
            query_vector=image_embedding.tolist(),
            limit=limit
        )

        if not search_result:
            return jsonify({"error": "No se encontraron resultados para la imagen"}), 404

        # Use the first result as the user_message
        top_result = search_result[0]
        user_message = f"farmaco:{top_result.payload.get('nombre')}"

        # Generate conversation_id
        conversation_id = generar_conversation_id()
        ubicacion = f"\nUbicación actual: lat = {lat} lng = {lng}"
        ciudad = f"\nEstoy en la ciudad de: {get_ciudad(lat,lng)}"
        user_message = f"Consulta: {user_message}{ubicacion}{ciudad}"

        # Set up the model
        if model_name == "gpt-4o":
            model = ChatOpenAI(temperature=0.2, model_name="gpt-4o")
        elif model_name == "llama3-8b-8192":
            model = ChatGroq(temperature=0.2, model_name="llama3-8b-8192")
        elif model_name == "llama-3.1-70b-versatile":
            model = ChatGroq(temperature=0.2, model_name="llama-3.1-70b-versatile")
        elif model_name == "gpt-4o-mini":
            model = ChatOpenAI(temperature=0.2, model_name="gpt-4o-mini")
        else:
            return jsonify({"error": "Modelo no soportado"}), 400

        # Set up the context and messages
        contexto_prev = [{"role": "user", "content": user_message}]
        messages_input = [
            {"role": "system", "content": system_message_ia_farma}
        ] + contexto_prev

        # Create and invoke the agent
        tools = [locales_cercanos, buscar_farmaco, especialista, buscar_medicos]
        ia_farma_agent = create_agent_ia_farma(model, tools=tools).with_config({"run_name": experiment_name})
        response = ia_farma_agent.invoke({"messages": messages_input})

        # Save context
        guardar_contexto(conversation_id, "user", user_message)
        guardar_contexto(conversation_id, "assistant", response["messages"][-1].content)

        # Prepare the response
        image_results = [
            {
                "score": hit.score,
                "product": {
                    "nombre": hit.payload.get("nombre")
                }
            } for hit in search_result
        ]

        to_return = {
            "conversation_id": conversation_id,
            "respuesta_agente": response["messages"][-1].content,
            "buscar_farmaco_resultado": buscar_farmaco_resultado,
            "locales_cercanos_resultado": locales_cercanos_resultado,
            "buscar_medicos_resultado": buscar_medicos_resultado,
            "image_search_results": image_results
        }

        return jsonify(to_return)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)