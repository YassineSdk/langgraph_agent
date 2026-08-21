
import yaml
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader




embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={
        "device": "cpu"
    },
    show_progress=False
)


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
    DIR_NAME=Path("prompts")
    prompt_path = DIR_NAME.joinpath(prompt_name)
    with open(prompt_path,"r",encoding="utf-8") as f:
        prompt = yaml.safe_load(f)

        return prompt['system_prompt']


def Retrieve_knowledge(query,filename):
    """
    this function retrives chunks with high semantic simularity with the a query 
    """
    raw_doc = read_knowledge(filename)

    # splitting the document into chunks 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
    )
    documents = splitter.create_documents([raw_doc])

    # creating the vectot store 
    vector_store = InMemoryVectorStore(embeddings)

    # adding the chunks 
    vector_store.add_documents(documents)

    # retrieve top 5 

    return vector_store.similarity_search(
        query,
        k=5
    )






