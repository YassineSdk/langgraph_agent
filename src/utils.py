
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
print(PROJECT_ROOT)


def load_embeddings():
    """
    this functions 
    - downloads the embeddings model if it does not exists in the local cache dir 
    - load the model as a langchain object 

    """

    MODEL_PATH = PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"

    if not MODEL_PATH.exists():
        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        model.save(str(MODEL_PATH))
    
    model_chain = HuggingFaceEmbeddings(model_name=str(MODEL_PATH))

    return model_chain


def read_knowledge(filename):
    """
    this function reads the knowledge base from a text file and return it's content 
    """
    loader = PyMuPDFLoader(filename)

    documents = loader.load()

    return "\n\n".join(
        doc.page_content
        for doc in documents
    )


def read_prompts(prompt_name):

    prompt_path = PROJECT_ROOT / "prompts"/ prompt_name
    with open(prompt_path,"r",encoding="utf-8") as f:
        prompt = yaml.safe_load(f)

        return prompt['system_prompt']

model_chain = load_embeddings()

def Retrieve_knowledge(query,filename):
    """
    this function retrives chunks with high semantic simularity with the a query 
    """
    MAX_TOKEN = 4500
    raw_doc = read_knowledge(filename)

    # if the document is small we will pass it 
    if len(raw_doc.split()) <= MAX_TOKEN:
        return [Document(page_content=raw_doc)]
    
    # splitting the document into chunks 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = splitter.create_documents([raw_doc])
    
    # creating the vectot store 
    vector_store = InMemoryVectorStore(model_chain)

    # adding the chunks 
    vector_store.add_documents(documents)

    # retrieve top 5 
    context = vector_store.similarity_search(
        query,
        k=5
    )

    return context







