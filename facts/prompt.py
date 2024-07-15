from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

load_dotenv()

embedding_function = OpenAIEmbeddings()

db = Chroma(
    persist_directory="facts/db",
    embedding_function=embedding_function
)

retriever = db.as_retriever()
