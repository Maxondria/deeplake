from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory

load_dotenv()

prompt = ChatPromptTemplate(
    input_variables=["content", "chat_history"],
    messages=[
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)


llm = ChatOpenAI()

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=FileChatMessageHistory(file_path="chat_history.json")
)

chain = LLMChain(llm=llm, prompt=prompt, memory=memory)


while True:
    content = input(">> ")
    if len(content) > 0:
        result = chain.invoke({"content": content})
        print(result["text"])
