import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings

Default_Temperature = 0

load_dotenv()

def get_api_key(model):
    return os.getenv(f"{model}_API_KEY")

def ChatViaGroq(model_name, temperature = Default_Temperature, api_key = get_api_key("GROQ")):
    return ChatGroq(model = model_name, temperature = temperature, api_key = api_key)

def get_OllamaEmbedding(model_name, temperature = Default_Temperature, base_url = os.getenv("OLLAMA_BASE_URL")  ):
    return OllamaEmbeddings(model= model_name,temperature= temperature, base_url = base_url)
