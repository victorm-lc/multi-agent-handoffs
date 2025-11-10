"""
HANDOFFS PATTERN (LangChain Multi-Agent)

This implements the "Handoffs" pattern from LangChain's multi-agent documentation:
https://docs.langchain.com/oss/python/langchain/multi-agent.md#handoffs

In handoffs, agents can directly pass control to each other. The "active" agent changes, 
and the user interacts with whichever agent currently has control.

Flow:
1. The current agent decides it needs help from another agent
2. It passes control (and state) to the next agent  
3. The new agent interacts directly with the user until it decides to hand off again or finish

This implementation uses Command + Send objects for advanced handoff control:
- Uses structured output (Pydantic models) for routing decisions
- Command + Send pattern allows passing custom state to different agents
- More explicit control over what data each agent receives

Key characteristics:
- Decentralized control: agents can change who is active
- Agents can interact directly with the user
- Complex, human-like conversation between specialists
- **MAIN BENEFIT: Control context sent to agents** - agents receive only focused context instead of full conversation history

Benefits:
- Agents interact directly with users
- Support for complex, multi-domain conversations
- Specialist takeover capabilities
- Flexible state management per agent
- Structured decision making for routing

When to use (from LangChain docs):
- Need centralized control over workflow? ❌ No
- Want agents to interact directly with the user? ✅ Yes
- Complex, human-like conversation between specialists? ✅ Strong

Perfect for: Multi-domain conversations, specialist takeover, complex agent interactions
"""

from pydantic import BaseModel, Field
from typing import Literal
from utils import llm as model, State
from agents.invoice_agent import graph as invoice_information_subagent
from agents.music_agent import graph as music_catalog_subagent
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage

# STRUCTURED OUTPUT MODEL
# This Pydantic model defines the structure for routing decisions
class Step(BaseModel):
    subagent: Literal["music_catalog_subagent", "invoice_information_subagent", "END"] = Field(
        description="Name of the subagent that should execute this step, or END if there is no need for additional summary needed"
    )
    context: str = Field(description="Instructions for the subagent on their task to be performed")

# Create a model that outputs structured decisions instead of free-form text
router_model = model.with_structured_output(Step)

supervisor_prompt = """You are an expert customer support assistant for a digital music store. You can handle music catalog or invoice related question regarding past purchases, song or album availabilities. 
Your primary role is to serve as a supervisor/planner for this multi-agent team that helps answer queries from customers, and generate the next agent to route to. 

Your team is composed of two subagents that you can use to help answer the customer's request:
1. music_catalog_subagent: this subagent has access to user's saved music preferences. It can also retrieve information about the digital music store's music 
catalog (albums, tracks, songs, etc.) from the database. 
2. invoice_information_subagent: this subagent is able to retrieve information about a customer's past purchases or invoices 
from the database. 

Based on the existing steps that have been taken in the messages, your role is to generate the next subagent that needs to be called as well as the context they need to answer user queries. 
This could be one step in an inquiry that needs multiple sub-agent calls. 
If subagents are no longer needed to answer the user question or if a question is unrelated to music or invoice, return END. 
"""

from langgraph.types import Command, Send

def supervisor(state: State) -> Command[Literal["music_catalog_subagent", "invoice_information_subagent", END]]:
    """
    HANDOFFS PATTERN: Supervisor that decides which agent should have control.
    
    From LangChain docs: "The current agent decides to transfer control to another agent. 
    The active agent changes, and the user may continue interacting directly with the new agent."
    
    This demonstrates the Command + Send pattern for handoffs where:
    1. Current agent (supervisor) decides it needs help from another agent
    2. It passes control (and state) to the next agent using Command + Send
    3. The new agent will interact directly with the user until handoff or finish
    4. Different agents can receive different input data for focused context
    
    KEY BENEFIT: Instead of passing the full conversation history to agents,
    Send allows us to pass only the specific context each agent needs (result.context).
    This enables decentralized control where agents can change who is active.
    """
    # Get structured routing decision from the LLM
    result = router_model.invoke([SystemMessage(content=supervisor_prompt)] + state["messages"])
    
    if result.subagent: 
        subagent = result.subagent
        
        if subagent == "music_catalog_subagent": 
            # Create custom input state for the music catalog agent
            # KEY: This replaces the full conversation with just the focused context!
            # Agent only sees result.context, not the entire conversation history
            agent_input = {**state, "messages": [{"role": "user", "content": result.context}]}
            return Command(goto=[Send(subagent, agent_input)])
            
        elif subagent == "invoice_information_subagent": 
            # Create custom input state for the invoice agent  
            # KEY: Same pattern - agent gets only the specific context it needs
            # This prevents information overload and improves agent focus
            agent_input = {**state, "messages": [{"role": "user", "content": result.context}]}
            return Command(goto=[Send(subagent, agent_input)])
            
        elif subagent == "END": 
            # Handle the end case by generating a summary
            summary_prompt = """
            You are an expert customer support assistant for a digital music store. You can handle music catalog or invoice related question regarding past purchases, song or album availabilities. 
            Your primary role is to serve as a supervisor this multi-agent team that helps answer queries from customers. 
            Respond to the customer through summarizing the conversation, including individual responses from subagents. 
            If a question is unrelated to music or invoice, politely remind the customer regarding your scope of work. Do not answer unrelated answers. 
            """
            messages = model.invoke([SystemMessage(content=summary_prompt)] + state["messages"])
            update = {
                "messages": [messages]
            }
            return Command(goto=END, update=update)
    else:
        raise ValueError(f"Invalid step")

# GRAPH CONSTRUCTION
# This pattern uses Command + Send, so no ToolNode is needed
supervisor_workflow = StateGraph(State)

# Add nodes: supervisor and the subagent graphs
supervisor_workflow.add_node("supervisor", supervisor, destinations=["music_catalog_subagent", "invoice_information_subagent", "__end__"])
supervisor_workflow.add_node("music_catalog_subagent", music_catalog_subagent)
supervisor_workflow.add_node("invoice_information_subagent", invoice_information_subagent)

# Define the flow:
# 1. Start with supervisor
# 2. Supervisor uses structured output to decide routing
# 3. Command + Send objects handle navigation with custom state
# 4. Subagents return to supervisor when complete
supervisor_workflow.add_edge(START, "supervisor")
supervisor_workflow.add_edge("music_catalog_subagent", "supervisor")  # Return to supervisor after completion
supervisor_workflow.add_edge("invoice_information_subagent", "supervisor")

# Note: No conditional edges needed because the supervisor node handles all routing
# through Command objects. The Command + Send pattern provides more direct control
# over navigation than the conditional edge + tool pattern.

graph = supervisor_workflow.compile()