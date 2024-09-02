from pydantic import BaseModel, Field # prueba con pydantic.v1 fue un poco más lenta
from typing_extensions import Literal, Dict, Any, List, Union, Optional # Cambié typing por typing_extensions, según lo que dice la documentación de Langchain

# Estructura BuscarFarmaco
class QdrantResult(BaseModel):
    """Structure of the response of the buscar_farmaco tool, part of qdrant_results"""
    nombre: str
    farmaco: str
    laboratorio: str
    score: float

class Producto(BaseModel):
    """Structure of the response of the buscar_farmaco tool, part of productos"""
    nombre: str
    url: str
    imagen: str
    precio: str

class BuscarFarmacoStructure(BaseModel):
    """Structure of the response of the buscar_farmaco tool"""
    gpt_response: str = Field(description="Result of the buscar_farmaco tool, part of gpt_response")
    qdrant_results: List[QdrantResult] = Field(description="Result of the buscar_farmaco tool, part of qdrant_results")
    productos: List[Producto] = Field(description="Result of the buscar_farmaco tool, part of productos")

# Estructura LocalesCercanos
# TODO: Probando con menos campos, para que sea más rápido. Ideal optimizar para que sea lo mínimo posible.
class LocalesCercanosFarmacia(BaseModel):
    """Structure of the response of the locales_cercanos tool, part of Farmacias"""
    fecha: str
    local_id: str
    # local_nombre: str
    # comuna_nombre: str
    # localidad_nombre: str
    # local_direccion: str
    # funcionamiento_hora_apertura: str
    # funcionamiento_hora_cierre: str
    # local_telefono: Optional[str]
    local_lat: float
    local_lng: float
    # funcionamiento_dia: str
    # fk_region: str
    # fk_comuna: str
    # fk_localidad: str

class LocalesCercanosTurno(BaseModel):
    """Structure of the response of the locales_cercanos tool, part of Turno"""
    fecha: str
    local_id: str
    fk_region: str
    fk_comuna: str
    fk_localidad: str
    local_nombre: str
    comuna_nombre: str
    localidad_nombre: str
    local_direccion: str
    funcionamiento_hora_apertura: str
    funcionamiento_hora_cierre: str
    local_telefono: Optional[str]
    local_lat: float
    local_lng: float
    funcionamiento_dia: str

class LocalesCercanosStructure(BaseModel):
    """Structure of the response of the locales_cercanos tool"""
    Farmacias: List[LocalesCercanosFarmacia] = Field(description="Result of the locales_cercanos tool, part of Farmacias")
    Turno: List[LocalesCercanosTurno] = Field(description="Result of the locales_cercanos tool, part of Turno")

class EspecialistaStructure(BaseModel):
    """Structure of the response of the especialista tool"""
    especialista: str = Field(description="Respuesta de la herramienta especialista")

# TODO: Add estructura de handle other query
class HandleOtherQueryStructure(BaseModel):
    """Structure of the response of the handle_other_query tool"""
    gpt_response: str = Field(description="Respuesta de la herramienta handle_other_query")

# Estructura Respuesta de herramientas
class ToolsResponseStructure(BaseModel):
    """Structure of the responses of the used tools"""
    tool: str = Field(description="Name of the tool used")
    # Choose depending on tool
    tool_response: Union[BuscarFarmacoStructure,LocalesCercanosStructure, EspecialistaStructure] = Field(description="Content of the used tool, depending on the tool")

class MessageAndToolsResponse(BaseModel):
    """Respond to the user with this"""
    assistant_message: str = Field(description="Introducción a la respuesta final. Solo agrega texto, no urls ni etiquetas de ningún tipo")
    tools_responses: Optional[List[ToolsResponseStructure]] = Field(description="Respuestas de herramientas usadas")
