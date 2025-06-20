from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool  # Assume tools are properly defined

import os

# Load environment variables
load_dotenv()

# Define structured output schema
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)

# Output parser
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a research assistant that will help generate a research paper.
Answer the user query and use necessary tools.
Wrap the output in this format and provide no other text:
{format_instructions}""",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Register tools
tools = [search_tool, wiki_tool, save_tool]

# Create agent
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)

# Agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# User input
query = input("What can I help you research? ")

# Run agent
raw_response = agent_executor.invoke({"query": query})

# Parse structured output
try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)
