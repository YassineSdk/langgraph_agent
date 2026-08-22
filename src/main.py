from dotenv import find_dotenv, load_dotenv 

from langchain.chat_models import init_chat_model 
from langgraph.graph import END, START, StateGraph
from prompt_toolkit import prompt 
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
# custom import 
from state import IntentClassifier, State
from nodes import *
from utils import *


load_dotenv(find_dotenv(".env"))


llm = init_chat_model(
    "groq:openai/gpt-oss-120b")

TRACING=os.getenv("LANGSMITH_TRACING")
PROJECT=os.getenv("LANGSMITH_PROJECT")
API_KEY=os.getenv("LANGSMITH_API_KEY") 

def classify_intent(state: State):
    structured_llm = llm.with_structured_output(IntentClassifier,
                        method="json_schema",
                        strict=True
                        )

    result = structured_llm.invoke([
        {
            "role": "system",
            "content": read_prompts("classifier_prompt.yaml"),
        },
        {"role": "user", "content": state["messages"][-1].content},
    ])
    if result.message_intent is None:
        raise ValueError('the intent classifier did not classify the user prompt')

    return {"message_intent": result.message_intent}

def route_input(state: State):

    if state.get("filename"):
        return "RAG"
    return "classifier"

async def chat_node(state):
    return await prompt_llm_chat(llm, state)

async def rag_node(state):
    return await prompt_llm_rag(llm, state)

async def web_search_node(state):
    return await prompt_llm_web_search(llm, state)

graph_builder = StateGraph(State)
graph_builder.add_node("classifier", classify_intent)
graph_builder.add_node("chat_agent", chat_node)
graph_builder.add_node("RAG_agent", rag_node)
graph_builder.add_node("web_search_agent",web_search_node)


# conditional Edges 

graph_builder.add_conditional_edges(
    "classifier",
    lambda state: state["message_intent"],
    {
        "chat": "chat_agent",
        "knowledge": "RAG_agent",
        "websearch":"web_search_agent"
    },
)

graph_builder.add_conditional_edges(
    START,
    route_input,
    {
        "RAG": "RAG_agent",
        "classifier": "classifier"
    }
    
)


# Nodes

graph_builder.add_edge("chat_agent", END)
graph_builder.add_edge("RAG_agent", END)
graph_builder.add_edge("web_search_agent",END)

checkpoint = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpoint)



