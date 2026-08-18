from dotenv import find_dotenv, load_dotenv 
import uuid
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from langchain.chat_models import init_chat_model 
from langgraph.graph import END, START, StateGraph
from prompt_toolkit import prompt 
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
# custom import 
from state import IntentClassifier, State
from nodes import *
from utils import *
from cli import show_banner

C = Console()
load_dotenv(find_dotenv(".env"))


llm = init_chat_model("groq:openai/gpt-oss-120b")


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



graph_builder = StateGraph(State)
graph_builder.add_node("classifier", classify_intent)
graph_builder.add_node("chat_agent", lambda state: prompt_llm_chat(llm, state))
graph_builder.add_node("RAG_agent", lambda state: prompt_llm_rag(llm, state))
graph_builder.add_node("Coding_agent", lambda state: prompt_llm_code(llm, state))
graph_builder.add_node("search_query_agent",lambda state: prompt_llm_search_query(llm,state))
graph_builder.add_node("web_search_agent",lambda state: prompt_llm_web_search(llm, state))

graph_builder.add_edge(START, "classifier")

# conditional nodes 

graph_builder.add_conditional_edges(
    "classifier",
    lambda state: state["message_intent"],
    {
        "chat": "chat_agent",
        "knowledge": "RAG_agent",
        "code": "Coding_agent",
        "websearch":"web_search_agent"
    },
)


# Nodes

graph_builder.add_edge("chat_agent", END)
graph_builder.add_edge("RAG_agent", END)
graph_builder.add_edge("Coding_agent", END)
graph_builder.add_edge("web_search_agent",END)

checkpoint = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpoint)
config = {
    "configurable": {"thread_id": str(uuid.uuid4())}
}

show_banner()


while True:
    user_input = prompt("> ")
    if user_input.strip().lower() == "q":
        C.print("see you soon")
        break

    response = graph.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config
    )

    if "__interrupt__" in response:
        interrupt_data = response["__interrupt__"][0]

        request = interrupt_data.value
        C.print(
            Panel(
                request["message"],
                title=request["type"]
            )

        )
        approvel = prompt("Continue? [y/n] ").lower().strip()
        response=graph.invoke(
            Command(resume="yes" if approvel =="y" else "no"),
            config=config
        )

    C.print(
        Markdown
            (
            response["messages"][-1].content
            )
        )

