
from state import State
from utils import read_prompts, Retrieve_knowledge
from rich.console import Console
from langgraph.types import interrupt

C = Console()

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
    with C.status("[green][RAG] thinking ...."):
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

def prompt_llm_web_search(query,max_results=10):
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
    with console.status("[green] searching the web ... [green ]"):

        if not query:
            raise ValueError(
                "the query is empty"
            )
        if not key:
            raise ValueError(
                "Tavily API key not found. Check your .env file."
            )

        Tav_client = TavilyClient(api_key=key)
        response = Tav_client.search(
            query=query,
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

        if response is None :
            raise ValueError('the Response is None')
            console.print("[red] web search ended [red]")

        console.print("[green] web search ended [green]")
        return {
            "answer": response.get("answer"),
            "results":response.get("results",[])
        }
