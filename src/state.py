from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field 
from langgraph.graph.message import add_messages 



class IntentClassifier(BaseModel):
    message_intent: Literal["chat", "knowledge", "websearch"] = Field(
        ...,
        description="Classify the user task.",
        )


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_intent: str | None
    search_query : str | None 
    filename : str | None 

