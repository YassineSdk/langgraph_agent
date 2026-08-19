from dotenv import load_dotenv 
import os 
from langsmith import Client

load_dotenv()
TRACING=os.getenv("LANGSMITH_TRACING")
PROJECT=os.getenv("LANGSMITH_PROJECT")
API_KEY=os.getenv("LANGSMITH_API_KEY") 

    

