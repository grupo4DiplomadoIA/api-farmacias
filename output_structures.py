class QdrantResult(BaseModel):
    nombre: str
    farmaco: str
    laboratorio: str
    score: float

class Producto(BaseModel):
    nombre: str
    url: str
    imagen: str
    precio: str

class BuscarFarmacoStructure(BaseModel):
    gpt_response: str = Field(description="Resultado de la herramienta buscar_farmaco, parte de gpt_response")
    qdrant_results: List[QdrantResult] = Field(description="Resultado de la herramienta buscar_farmaco, parte de qdrant_results")
    productos: List[Producto] = Field(description="Resultado de la herramienta buscar_productos, parte de productos")

# TODO: Definir estructura para respuesta de las demás herramientas
# class LocalCercanoStructure(BaseModel):
#     nombre: str
#     ...

class ToolsResponse(BaseModel):
    """Respond to the user with this"""
    tool: str = Field(description="Nombre de la herramienta que se usó")
    tool_response: List[BuscarFarmacoStructure] = Field(description="Contenido de herramienta usada para buscar_farmaco")