"""
Punto de entrada del agente de reflexión para LinkedIn.

Define un grafo de estado con LangGraph que alterna entre generar publicaciones
y reflexionar sobre ellas hasta alcanzar un resultado satisfactorio.
"""
# 1. IMPORTACIONES
#------------------------------------------------------------------------------------------------
# TypedDict define estructuras de datos tipadas; Annotated permite añadir metadatos
from typing import TypedDict, Annotated

# Carga variables de entorno desde .env (por ejemplo, OPENAI_API_KEY)
from dotenv import load_dotenv

# Ejecutamos la carga de variables de entorno al importar el módulo
load_dotenv()

# BaseMessage es la clase base para mensajes; HumanMessage representa mensajes del usuario
from langchain_core.messages import BaseMessage, HumanMessage

# END indica fin del grafo; StateGraph construye el grafo de flujo
from langgraph.graph import END, StateGraph

# Reductor que fusiona listas de mensajes (añade en lugar de reemplazar)
from langgraph.graph.message import add_messages

# Importamos las cadenas de generación y reflexión definidas en chains.py
from chains import generate_chain, reflect_chain

#2. ESQUEMA DEL ESTADO DEL GRAFO
#------------------------------------------------------------------------------------------------
# Esquema del estado del grafo: un diccionario con clave "messages"
# Annotated con add_messages hace que los mensajes se acumulen en lugar de sustituirse
class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 3. CREACIÓN DE LOS NODOS DEL GRAFO
#------------------------------------------------------------------------------------------------
# Identificadores de los nodos del grafo
GENERATE = "generate"  # Nodo que genera o mejora la publicación
REFLECT = "reflect"   # Nodo que evalúa y critica la publicación


# Nodo de generación: invoca la cadena de generación con el historial actual
# y devuelve el estado actualizado con la nueva publicación generada
def generation_node(state: MessageGraph):
    # Invocamos la cadena de generación (prompt + LLM) con el historial
    generated_response = generate_chain.invoke({"messages": state["messages"]})

    # Devolvemos el estado actualizado; add_messages fusionará la respuesta al historial
    return {"messages": [generated_response]}


# Nodo de reflexión: invoca la cadena de reflexión que critica la publicación
# y devuelve la crítica como mensaje humano para que el siguiente ciclo la use
def reflection_node(state: MessageGraph):
    # Invocamos la cadena de reflexión con todo el historial
    res = reflect_chain.invoke({"messages": state["messages"]})
    # Envolvemos la respuesta en HumanMessage para que el modelo la interprete como feedback de humano
    return {"messages": [HumanMessage(content=res.content)]}


# 4. CREACIÓN DEL GRAFO
#------------------------------------------------------------------------------------------------
# Creamos el grafo de estado con el esquema MessageGraph
builder = StateGraph(state_schema=MessageGraph)

# Añadimos los dos nodos al grafo especificando la función que se ejecutará en cada nodo
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)

# El grafo comienza en el nodo de generación
builder.set_entry_point(GENERATE)

# 5. CREACIÓN DE LAS ARISTAS DEL GRAFO
#------------------------------------------------------------------------------------------------
# Función de decisión: determina si continuar iterando o finalizar
# Limita las iteraciones para evitar bucles infinitos (ejemplo:máx. ~3 ciclos con 6 mensajes)
def should_continue(state: MessageGraph):
    # Si hay más de 6 mensajes, damos por finalizado el proceso
    if len(state["messages"]) > 2:
        return END
    # Si no, pasamos al nodo de reflexión para otra ronda de mejora
    return REFLECT


# Arista condicional desde GENERATE: según should_continue va a END o a REFLECT, path_map define las aristas a seguir para visualizar correctamente el diagrama mermail
builder.add_conditional_edges(GENERATE, should_continue, path_map={REFLECT: REFLECT, END: END})

# Arista fija: tras reflexionar, siempre volvemos a generar
builder.add_edge(REFLECT, GENERATE)

# 6. COMPILACIÓN DEL GRAFO
#------------------------------------------------------------------------------------------------
# Compilamos el grafo para poder ejecutarlo
graph = builder.compile()

# Mostramos la representación visual del grafo (Mermaid y ASCII)
print(graph.get_graph().draw_mermaid()) #se puede visualizar en https://mermaidviewer.com/editor
#graph.get_graph().print_ascii()

# 7. EJECUCIÓN DEL GRAFO
#------------------------------------------------------------------------------------------------   
# Bloque que se ejecuta al lanzar el script directamente (python main.py)
if __name__ == "__main__":
    print("Hola LangGraph")

    # Entrada de ejemplo: un mensaje humano pidiendo mejorar una publicación
    inputs = {
        "messages": [
            HumanMessage(
                content="""Mejorar esta publicación de LinkedIn:

¡La nueva funcionalidad Tool Calling de @LangChainAI es realmente una revolución!

Después de mucha espera, por fin está disponible y facilita enormemente la implementación de agentes en diferentes modelos gracias a la integración con function calling.

¿Ya la probaste? Cuéntame tu experiencia en los comentarios.
"""
            )
        ]
    }

    # Invocamos el grafo completo con la entrada
    response = graph.invoke(inputs)

    # Mostramos la respuesta final (estado con todos los mensajes generados)
    print(response)
