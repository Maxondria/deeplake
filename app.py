from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.sequential import SequentialChain
from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default="python")
parser.add_argument("--task", type=str, default="sort a list")

args = parser.parse_args()


llm = ChatOpenAI()

code_prompt = PromptTemplate.from_template(
    "Write a very short {language} function that will {task}."
)

test_prompt = PromptTemplate.from_template(
    "Write a test for the following {language} function: \n {code}"
)


code_chain = LLMChain(llm=llm, prompt=code_prompt, output_key="code")
test_chain = LLMChain(llm=llm, prompt=test_prompt, output_key="test")

chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["language", "task"],
    output_variables=["code", "test"],
)

result = chain.invoke({"language": args.language, "task": args.task})

print("<" * 40 + " Generated Code " + ">" * 40)
print(result["code"])
print("<" * 40 + " Generated Test " + "<" * 40)
print(result["test"])
