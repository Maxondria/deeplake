from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


load_dotenv()


text_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator="\n"
)

loader = TextLoader(file_path=Path("facts/facts.txt"))
docs = loader.load_and_split(text_splitter=text_splitter)

embedding_fn = OpenAIEmbeddings()

db = Chroma.from_documents(
    documents=docs,
    embedding=embedding_fn,
    persist_directory="facts/db"
)

results = db.similarity_search(
    query="What is an interesting fact about the English language?", k=1)

for result in results:
    print("\n\n")
    print(result.page_content)
