# 1. Bring in dependencies 
from typing import Annotated 
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages 
from langgraph.checkpoint.memory import InMemorySaver 
from langchain_ollama import ChatOllama
from colorama import Fore 
from langgraph.prebuilt import ToolNode 
from tool import simple_screener

# 2. Create LLM
llm = ChatOllama(model='qwen2.5:14b')

# 8. Create tool
tools = [simple_screener]
# 9. Bind LLM with tools
llm_with_tools = llm.bind_tools(tools)
# 10. Create Tool Node 
tool_node = ToolNode(tools)

# 3. Create state
class State(dict): 
    messages: Annotated[list, add_messages]

import sys

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 4. Build LLM node 
def chatbot(state: State): 
    return {"messages": [llm_with_tools.invoke(state['messages'])]}

# 11. Create Router Node
def router(state: State): 
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls: 
        print(Fore.CYAN + "[*] Fetching screener data from Yahoo Finance..." + Fore.RESET)
        return "tools" 
    else: 
        return END 

# 5. Assemble Graph 
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "chatbot")
# 12. Update graph for Tools
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_conditional_edges("chatbot", router)
# 6. Add Memory and Compile Graph 
memory = InMemorySaver() 
graph = graph_builder.compile(checkpointer=memory)

# 7. Build call loop and run it
if __name__ == '__main__': 
    print(Fore.GREEN + "=== Stock Screener Agent Ready ===" + Fore.RESET)
    print("Type 'exit' or 'quit' to stop.\n")
    while True: 
        prompt = input(Fore.WHITE + "Agent Prompt > " + Fore.RESET)
        if not prompt.strip():
            continue
        if prompt.lower() in ["exit", "quit", "q"]:
            print(Fore.GREEN + "Goodbye!" + Fore.RESET)
            break
        print(Fore.LIGHTBLACK_EX + "Thinking..." + Fore.RESET)
        result = graph.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"configurable": {"thread_id": "1234"}}
        )
        print("\n" + Fore.LIGHTYELLOW_EX + result['messages'][-1].content + Fore.RESET + "\n")
