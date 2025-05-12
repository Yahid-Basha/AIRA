import chainlit as cl
from langgraph.graph import StateGraph, END
from llm_wrapper import llm_call
from qdrant_utils import query_qdrant
import json

# State Object
class AgentState:
    def __init__(self):
        self.memory = []
        self.last_response = ""

# Node: Input
def input_node(state: AgentState, message: str) -> AgentState:
    state.memory.append({"user": message})
    return state

# Node: LLM Entity Extraction
def extraction_node(state: AgentState) -> AgentState:
    user_message = state.memory[-1]["user"]
    extraction_prompt = f"""
Extract the following information from the user's query:

1. Application Name as stored in the database (Example: SALES_APP).
2. Start Date in ISO format (YYYY-MM-DD).
3. End Date in ISO format (YYYY-MM-DD).

Rules:
- If only a single date is mentioned, start_date and end_date should both be that date.
- If a date range is provided, extract both start and end dates.
- If no date is mentioned, return null for both dates.
- Application names should be returned in UPPERCASE.

Respond strictly in this JSON format:
{{"app_name": "<APP_NAME>", "start_date": "<YYYY-MM-DD>", "end_date": "<YYYY-MM-DD>"}}

User Query: "{user_message}"
"""
    extraction_response = llm_call(extraction_prompt)
    try:
        extracted_data = json.loads(extraction_response)
    except json.JSONDecodeError:
        extracted_data = {"app_name": None, "start_date": None, "end_date": None}

    state.memory.append({"extracted_info": extracted_data})
    return state

# Node: Qdrant Query
def qdrant_node(state: AgentState) -> AgentState:
    extracted_info = state.memory[-1]["extracted_info"]
    app_name = extracted_info.get("app_name")
    start_date = extracted_info.get("start_date")
    end_date = extracted_info.get("end_date")

    if not app_name:
        result = "No valid application name found in the query."
    else:
        result = query_qdrant(app_name, start_date, end_date)

    state.memory.append({"qdrant_result": result})
    return state

# Node: Final LLM Response
def llm_response_node(state: AgentState) -> AgentState:
    user_message = state.memory[-1]["user"]
    extracted_info = state.memory[-1]["extracted_info"]
    qdrant_data = state.memory[-1].get("qdrant_result", "")

    prompt = f"""
You are a customer support analysis assistant.

User Query: {user_message}
Extracted Information: {json.dumps(extracted_info)}
Relevant Data from Database: {qdrant_data}

Generate a concise and informative response for the user.
"""
    response = llm_call(prompt)
    state.last_response = response
    state.memory.append({"assistant": response})
    return state

# LangGraph Workflow
graph = StateGraph(AgentState)
graph.add_node("Input", input_node)
graph.add_node("Extraction", extraction_node)
graph.add_node("QueryQdrant", qdrant_node)
graph.add_node("LLMResponse", llm_response_node)

graph.set_entry_point("Input")
graph.add_edge("Input", "Extraction")
graph.add_edge("Extraction", "QueryQdrant")
graph.add_edge("QueryQdrant", "LLMResponse")
graph.add_edge("LLMResponse", END)

app_graph = graph.compile()

# Chainlit Integration
@cl.on_chat_start
def start():
    cl.user_session.set("agent_state", AgentState())

@cl.on_message
async def main(message: cl.Message):
    state: AgentState = cl.user_session.get("agent_state")
    result = app_graph.invoke(state, {"message": message.content})
    final_response = result.last_response

    await cl.Message(content=final_response).send()

"""
"""
