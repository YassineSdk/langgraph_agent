from state import *
from utils import *
from rich.console import Console
from langgraph.types import interrupt
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv, find_dotenv
from tavily import TavilyClient
import os

C = Console()

load_dotenv(find_dotenv(".env"))

def prompt_llm_chat(llm, state: State):

    with C.status("[green][CHAT] thinking ...."):

        messages = [
            {"role": "system", "content": read_prompts("chat_prompt.yaml")}
        ] + state["messages"]

        response = llm.invoke(messages)
    return {"messages": [response]}


def prompt_llm_rag(llm, state: State):

    decision = interrupt(
        {
            "type":"rag_approval",
            "message":"The agent wants to use the RAG system. This may be an expensive operation. Do you want to continue?"
        }
    )
    if decision != "yes" :
        return {
            "messages":[
                {
                    "role":"assistant",
                    "content":"RAG search cancelled by the user."
                }
            ]
        }
    with C.status("[green][RAG] Searching docs ...."):
        query = state["messages"][-1].content
        docs = Retrieve_knowledge(query)

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        messages = [
            {
                "role": "system",
                "content": read_prompts("rag_prompt.yaml")
            },
            {
                "role": "user",
                "content": f"""
        Context:
        {context}

        Question:
        {query}
        """
            }
        ]
        response = llm.invoke(messages)

    return {"messages": [response]}


def prompt_llm_code(llm, state: State):
    with C.status("[green][Coding] thinking ...."):
        messages = [
            {"role": "system", "content": read_prompts("code_prompt.yaml")}
        ] + state["messages"]
        response = llm.invoke(messages)
    return {"messages": [response]}

def prompt_llm_web_search(llm,state:State,max_results=10):
    """
    this function takes a query and search it in the web via an API called
    tavily , a web search API that returns results
    args :
        - query (str) : query what are we searching for
    results :
        - list [dict]
        {
            { "url": .....,
            "title": .....,
            "content": .....,
            "score":........,                   
            "raw_content":......,
            },
        }
    """
    key = os.getenv("Tavily_APIKEY")
    opm_query = search_rewriter(llm,state)
    with C.status(f"[green] searching for {opm_query} ... [green ]") as status:

        if not opm_query:
            raise ValueError(
                "the query is empty"
            )
        if not key:
            raise ValueError(
                "Tavily API key not found. Check your .env file."
            )

        Tav_client = TavilyClient(api_key=key)
        response = Tav_client.search(
            query=opm_query,
            search_depth="advanced",
            max_results=max_results,
            include_domains = None,
            include_raw_content=False,
            include_answer=True ,
            exclude_domains=  [
            "pinterest.com",
            "facebook.com",
            "twitter.com",
            "reddit.com",
            "linkedin.com"
            ]
        )
        status.update("[yellow]Processing search results...[/yellow]")
        if response is None :
            C.print("[red] web search ended [red]")
            raise ValueError('the Response is None')

        results= {
            "answer": response.get("answer"),
            "results":response.get("results",[])
        }

        messages = [
            {
                "role": "system",
                "content": read_prompts("web_search_prompt.yaml")
            },
            {
                "role": "user",
                "content": f"""
        Context:
        {results}

        Question:
        {opm_query}
        """
            }
        ]
        llm_response = llm.invoke(messages)
    C.print("[bold green]✓ Web search completed[/bold green]")
    return {"messages": [llm_response]}



def search_rewriter(llm,state:State):
    messages = state["messages"][-1]

    prompt = """
    You are a web search query optimizer.

    Convert the user's request into a concise, precise search query
    that will retrieve the most relevant information from the web.

    Rules:
    - Keep the original intent.
    - Include important entities, dates, locations, or technical terms.
    - Remove conversational filler.
    - Do not answer the user's question.
    - Return only the optimized search query.
    """

    result = llm.invoke(
        [
        SystemMessage(content=prompt),
        messages
        ]
    )
    return result.content.strip()

