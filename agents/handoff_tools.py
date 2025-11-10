"""
HANDOFFS PATTERN - TOOLS IMPLEMENTATION (LangChain Multi-Agent)

This implements the "Handoffs" pattern from LangChain's multi-agent documentation:
https://docs.langchain.com/oss/python/langchain/multi-agent#handoffs

In handoffs, agents can directly pass control to each other. The "active" agent changes, 
and the user interacts with whichever agent currently has control.

Flow:
1. The current agent decides it needs help from another agent
2. It passes control (and state) to the next agent  
3. The new agent interacts directly with the user until it decides to hand off again or finish

This specific implementation uses "handoffs as tools" - a LangGraph pattern where:
- Handoff tools return Command objects that specify navigation to different agent nodes
- Tools can update state and navigate simultaneously 
- Each agent exists as a separate node in the graph (not just tool calls)

Key characteristics:
- Decentralized control: agents can change who is active
- Agents can interact directly with the user
- Complex, human-like conversation between specialists
- Tools return Command objects instead of simple strings
- Command objects specify destination nodes and state updates

Benefits:
- Agents interact directly with users
- Support for complex, multi-domain conversations  
- Specialist takeover capabilities
- More flexible routing than simple tool returns
- Native LangGraph handoff pattern with explicit control

When to use (from LangChain docs):
- Need centralized control over workflow? ❌ No
- Want agents to interact directly with the user? ✅ Yes
- Complex, human-like conversation between specialists? ✅ Strong

Perfect for: Multi-domain conversations, specialist takeover, explicit agent navigation control
Note: You can mix patterns - use handoffs for agent switching, and have each agent call subagents as tools.
"""

from agents.invoice_agent import graph as invoice_information_subagent
from agents.music_agent import graph as music_catalog_subagent
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import ToolMessage, SystemMessage
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langgraph.prebuilt import ToolNode
from utils import llm as model, State

# HANDOFFS PATTERN - TOOLS IMPLEMENTATION  
# From LangChain docs: "Agents can directly pass control to each other. The 'active' agent changes,
# and the user interacts with whichever agent currently has control."
# These tools demonstrate the "handoffs as tools" pattern where tools return Command objects
# instead of simple strings. This enables decentralized control and agent-to-agent navigation.

@tool("transfer-to-invoice-agent")
def transfer_to_invoice_agent(
    runtime: ToolRuntime,
    reason: str = "Invoice-related inquiry",
    context: str = "Context for the invoice agent"
) -> Command:
    """Transfer control to the invoice agent for invoice-related questions.
    
    HANDOFFS PATTERN: This demonstrates how the current agent passes control to another agent.
    From LangChain docs: "The current agent decides to transfer control to another agent.
    The active agent changes, and the user may continue interacting directly with the new agent."
    
    Flow:
    1. Current agent (via tool) decides it needs help from invoice agent
    2. Tool receives current state and creates handoff representation  
    3. Returns Command object that passes control (and state) to target agent
    4. Invoice agent becomes active and can interact directly with user
    
    Args:
        runtime: Tool runtime with access to state and tool_call_id (injected automatically)
        reason: Reason for the transfer
        context: Context information for the target agent
    """
    agent_name = "invoice_information_subagent"
    
    # Create a ToolMessage to represent the handoff in the conversation history
    # This follows LangGraph docs recommendation to add handoff representation
    tool_message = ToolMessage(
        content=f"Successfully transferred to {agent_name}. Reason: {reason}. Context: {context}",
        name="transfer-to-invoice-agent",
        tool_call_id=runtime.tool_call_id,
    )
    
    # Return Command object that specifies:
    # - goto: which node to navigate to
    # - update: how to update the state (add the handoff message)
    return Command(goto=agent_name, update={"messages": runtime.state["messages"] + [tool_message]})

@tool("transfer-to-music-catalog-agent")
def transfer_to_music_catalog_agent(
    runtime: ToolRuntime,
    reason: str = "Music catalog inquiry",
    context: str = "Context for the music catalog agent"
) -> Command:
    """Transfer control to the music catalog agent for music-related questions.
    
    Same pattern as invoice transfer but targeting the music catalog agent.
    Demonstrates how multiple handoff tools can coexist in the same graph.
    """
    agent_name = "music_catalog_subagent"
    
    tool_message = ToolMessage(
        content=f"Successfully transferred to {agent_name}. Reason: {reason}. Context: {context}",
        name="transfer-to-music-catalog-agent",
        tool_call_id=runtime.tool_call_id,
    )

    return Command(goto=agent_name, update={"messages": runtime.state["messages"] + [tool_message]})

# SUPERVISOR SETUP
tools = [transfer_to_invoice_agent, transfer_to_music_catalog_agent]
tool_node = ToolNode(tools)

supervisor_prompt = """You are an expert customer support assistant for a digital music store. You can handle music catalog or invoice related questions regarding past purchases, song or album availabilities.

Your primary role is to serve as a supervisor for this multi-agent team that helps answer queries from customers.

Your team is composed of two subagents:
1. music_catalog_subagent: Has access to user's saved music preferences and can retrieve information about the digital music store's music catalog (albums, tracks, songs, etc.) from the database.
2. invoice_information_subagent: Can retrieve information about a customer's past purchases or invoices from the database.

DECISION LOGIC:
- If the user's question needs specialist help AND no subagent has responded yet, use the appropriate handoff tool
- If a subagent has already provided a response, DO NOT call tools again - instead, synthesize and present their response to the user in a helpful, conversational way
- If the question is unrelated to music or invoices, respond directly without using tools

IMPORTANT: After a subagent completes their task, if there's no more help needed from your team, your job is to act as the final interface to the user - synthesize the subagent's response and present it conversationally, don't just add generic follow-ups."""

# Bind tools to model
model_with_tools = model.bind_tools(tools)

# Supervisor node that calls the model
def supervisor_node(state: State):
    """Supervisor node that decides whether to call handoff tools or respond directly."""
    messages = [SystemMessage(content=supervisor_prompt)] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

# Conditional edge function to route based on tool calls
def should_continue(state: State):
    """Route to tools if the model called tools, otherwise end."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# GRAPH CONSTRUCTION
# This pattern requires adding the subagent graphs as separate nodes
supervisor_workflow = StateGraph(State)

# Add all nodes: supervisor, tools, and the actual subagent graphs
supervisor_workflow.add_node("supervisor", supervisor_node)
supervisor_workflow.add_node("tools", tool_node, destinations=["music_catalog_subagent", "invoice_information_subagent"])
supervisor_workflow.add_node("music_catalog_subagent", music_catalog_subagent)  # Actual subagent graph
supervisor_workflow.add_node("invoice_information_subagent", invoice_information_subagent)  # Actual subagent graph

# Define the flow
supervisor_workflow.add_edge(START, "supervisor")
supervisor_workflow.add_conditional_edges("supervisor", should_continue, {
    "tools": "tools",
    END: END
})
# The tools node will return Command objects that specify which subagent to goto
# After subagents complete, they return to supervisor
supervisor_workflow.add_edge("music_catalog_subagent", "supervisor")  
supervisor_workflow.add_edge("invoice_information_subagent", "supervisor")

graph = supervisor_workflow.compile(name="supervisor")