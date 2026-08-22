from main import graph 
import chainlit as cl
from utils import Retrieve_knowledge
from pathlib import Path



SUPPORTED_EXTENTIONS = {".pdf"}


@cl.on_chat_start
async def start():
    cl.user_session.set(
        "thread_id",
        cl.context.session.id
    )


@cl.on_message
async def main(message: cl.Message):

    thread_id = cl.user_session.get("thread_id")
    query = message.content
    file_ = None 

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # checking if a file exist
    if message.elements:
        file = message.elements[0]

        # validating if the file exists
        extention = Path(file.name).suffix.lower()
        if extention not in SUPPORTED_EXTENTIONS:
            await cl.Message(
                content=f"I can't Read {file.name} currently support PDF documents only."
            ).send()
            return 
        
        file_ = file

    result = await graph.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query 
                }
            ],
            "filename":file_.path if file_ else None
        },
        config=config
    )
    answer = result["messages"][-1].content

    await cl.Message(
        content=answer
    ).send()


