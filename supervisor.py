import os
from dotenv import load_dotenv
from torch import subtract
load_dotenv()
hf = os.getenv("HF_TOKEN")

# print(hf)

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

endpoint = HuggingFaceEndpoint(repo_id='openai/gpt-oss-20b')
llm = ChatHuggingFace(llm=endpoint)

def subtract(a: int, b: int) -> int:
    "Subtract the second number from the first and return the result."
    return a - b

def add(a: int, b: int) -> int:
    "Add two numbers together and return the result."
    return a + b

agent1 = create_react_agent(
    llm, 
    tools=[add, subtract],
    name="MathAgent",
    prompt="""You are a helpful assistant that can perform basic arithmetic operations.
    Please do not attempt to answer questions that require external knowledge 
    and take help one tool at a time.
    Use add function for addition and subtract function for subtraction."""
)
agent2 = create_react_agent(
    llm,
    tools=[],
    name="InfoAgent",
    prompt="""You are a helpful assistant that can only provide information 
    that is explicitly given in the query.
    Do not attempt to infer or generate information that is not directly 
    provided in the query."""
)
super_agent = create_supervisor(
    model=llm, 
    agents=[agent1, agent2],
    prompt="""You are a supervisor overseeing two agents. 
    Your task is to determine which agent is best suited to answer.
    For research work use InfoAgent and for math calculations use MathAgent.
    """
)
query="find out what is the population of Paris + 1000 and then subtract 2000 from the result?"
app = super_agent.compile()
resp = app.invoke({"messages": [HumanMessage(content=query)]})  
print(resp['messages'][-1].content)