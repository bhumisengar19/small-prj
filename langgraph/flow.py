import os
import sys
from typing import Annotated, Sequence, TypedDict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages 
from langgraph.checkpoint.memory import InMemorySaver 
from langchain_ollama import ChatOllama
from colorama import Fore, Style, init
from langgraph.prebuilt import ToolNode 
from tool import simple_screener, get_stock_details, get_stock_news

# Initialize colorama
init(autoreset=True)

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 1. Configuration & Model Initialization
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.2
)

# 2. Tool Registration
tools = [simple_screener, get_stock_details, get_stock_news]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

# 3. System Prompt for Financial Analysis
SYSTEM_PROMPT = SystemMessage(
    content="""You are an expert financial market analyst assistant equipped with real-time Yahoo Finance screening and ticker research tools.

Capabilities & Guidelines:
1. Screener Tool (simple_screener): Use this to discover stocks/funds/bonds matching macro market criteria (e.g., day_gainers, day_losers, most_actives, undervalued_large_caps, growth_technology_stocks, etc.).
2. Stock Details Tool (get_stock_details): Use this to fetch deep fundamentals (P/E, Market Cap, 52-week High/Low, Target Price, Business Summary) when discussing specific tickers.
3. Stock News Tool (get_stock_news): Use this to explain recent price movements, earnings catalysts, or company events.
4. Response Formatting:
   - Present market comparisons and multi-stock screener outputs in clean Markdown tables.
   - For Analyst Ratings: note that 1.0 = Strong Buy, 2.0 = Buy, 3.0 = Hold, 4.0 = Underperform, 5.0 = Sell.
   - Always state clearly that market data is for informational/educational purposes only and not individualized financial advice.
   - Be concise, analytical, and highlight standout metrics (e.g. low P/E, high dividend yield, near 52w high).
"""
)

# 4. State Definition using TypedDict
class AgentState(TypedDict): 
    messages: Annotated[Sequence[BaseMessage], add_messages]

# 5. Agent Nodes
def chatbot_node(state: AgentState) -> dict: 
    messages = list(state['messages'])
    # Inject system prompt at the beginning if not already present
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SYSTEM_PROMPT] + messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def router_node(state: AgentState) -> str: 
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls: 
        tool_names = [call.get('name', 'tool') for call in last_message.tool_calls]
        print(Fore.CYAN + f"[*] Executing tools: {', '.join(tool_names)}..." + Fore.RESET)
        return "tools" 
    return END

# 6. Assemble LangGraph Workflow
graph_builder = StateGraph(AgentState)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", router_node)
graph_builder.add_edge("tools", "chatbot")

# Compile with checkpoint memory
memory = InMemorySaver() 
graph = graph_builder.compile(checkpointer=memory)

# 7. Helper CLI functions
def print_help():
    print(Fore.CYAN + "\n=== Available Commands & Tips ===" + Fore.RESET)
    print("  " + Fore.YELLOW + "/help" + Fore.RESET + "      - Show this help menu")
    print("  " + Fore.YELLOW + "/clear" + Fore.RESET + "     - Reset conversation memory")
    print("  " + Fore.YELLOW + "/model" + Fore.RESET + "     - Show current LLM model configuration")
    print("  " + Fore.YELLOW + "exit/quit" + Fore.RESET + "  - Exit the application")
    print("\n" + Fore.CYAN + "Example Queries:" + Fore.RESET)
    print("  - 'Find the top 5 day gainers today and summarize them in a table'")
    print("  - 'Show me undervalued large caps and compare their P/E ratios'")
    print("  - 'What is the latest news and rating for NVDA?'")
    print("  - 'Screen growth technology stocks, then give details on the first one'\n")

# 8. Interactive Execution Loop
if __name__ == '__main__': 
    print(Fore.GREEN + "=============================================" + Fore.RESET)
    print(Fore.GREEN + "   📈 LangGraph Financial Screener Agent    " + Fore.RESET)
    print(Fore.GREEN + "=============================================" + Fore.RESET)
    print(f"Model: {Fore.YELLOW}{OLLAMA_MODEL}{Fore.RESET} | Endpoint: {Fore.YELLOW}{OLLAMA_BASE_URL}{Fore.RESET}")
    print("Type " + Fore.YELLOW + "/help" + Fore.RESET + " for tips, or " + Fore.YELLOW + "exit" + Fore.RESET + " to quit.\n")

    thread_id = "user_session_1"
    config = {"configurable": {"thread_id": thread_id}}

    while True: 
        try:
            prompt = input(Fore.WHITE + Style.BRIGHT + "Agent Prompt > " + Style.RESET_ALL)
        except (KeyboardInterrupt, EOFError):
            print("\n" + Fore.GREEN + "Goodbye!" + Fore.RESET)
            break

        clean_prompt = prompt.strip()
        if not clean_prompt:
            continue

        if clean_prompt.lower() in ["exit", "quit", "q"]:
            print(Fore.GREEN + "Goodbye!" + Fore.RESET)
            break

        if clean_prompt.lower() == "/help":
            print_help()
            continue

        if clean_prompt.lower() == "/clear":
            # Generate a new session thread
            import uuid
            thread_id = f"user_session_{uuid.uuid4().hex[:6]}"
            config = {"configurable": {"thread_id": thread_id}}
            print(Fore.MAGENTA + "[✓] Conversation history reset.\n" + Fore.RESET)
            continue

        if clean_prompt.lower() == "/model":
            print(Fore.CYAN + f"Current Model: {OLLAMA_MODEL} at {OLLAMA_BASE_URL}\n" + Fore.RESET)
            continue

        print(Fore.LIGHTBLACK_EX + "Analyzing request..." + Fore.RESET)
        
        try:
            # Stream execution
            events = graph.stream(
                {"messages": [HumanMessage(content=clean_prompt)]},
                config=config,
                stream_mode="values"
            )
            
            last_content = ""
            for event in events:
                if "messages" in event and event["messages"]:
                    last_msg = event["messages"][-1]
                    if isinstance(last_msg, AIMessage) and last_msg.content:
                        last_content = last_msg.content

            if last_content:
                print("\n" + Fore.LIGHTYELLOW_EX + last_content + Fore.RESET + "\n")
            else:
                print(Fore.RED + "[!] No response generated." + Fore.RESET + "\n")

        except Exception as e:
            print(Fore.RED + f"[!] Error during graph execution: {e}" + Fore.RESET)
            print(Fore.LIGHTBLACK_EX + "Ensure Ollama is running and the model is pulled." + Fore.RESET + "\n")

