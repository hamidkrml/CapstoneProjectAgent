import streamlit as st
from typing import Annotated
from typing_extensions import TypedDict


from langchain_core.messages import AIMessage,HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
class State(TypedDict):
    messages:Annotated[list,add_messages]

graph_builder = StateGraph(State)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that helps the user to create a diet plan. "
            "You can ask the user for their preferences and dietary restrictions."
        ),
    MessagesPlaceholder(variable_name="messages"),

    ]
)

llm = ChatOpenAI( model="gpt-4o",api_key="")
generate = prompt | llm

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that helps the user to create a diet plan. "
            "You can ask the user for their preferences and dietary restrictions."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflect = reflection_prompt | llm
def generation_node(state: State) -> State:
    return ("messages", (generate.invoke(state["messages"])))

def reflection_node(state: State):

    cls_map = {
        "ai": HumanMessage, "human" : AIMessage }
    translated = [state["messages"][0]]+ [
        cls_map[msg.type](content=msg.content) for msg in state["messages"][1:]
    ]
    
    res = reflect.invoke(translated)
    return {"messages": [HumanMessage(content=res.content)]}


graph_builder.add_node("generate", generation_node)
graph_builder.add_node("reflect", reflection_node)
graph_builder.add_edge(START, "generate")
def should_continue(state: State):
    if len(state["messages"]) > 3:
       
            return END
    return "reflect"

graph_builder.add_conditional_edges("generate", should_continue)
graph_builder.add_edge("reflect", "generate")

memory = MemorySaver()

#graph_builder.add_memory(memory)

grap = graph_builder.compile(checkpointer=memory)

confiq = {"configurable":{"thread_id":"1"}}

def stream_graph_update(user_input: str):
    for event in grap.stream({"messages": [HumanMessage(content=user_input)]}, confiq):
        state = grap.get_state(confiq)
    last_message = state["messages"][-1]
    st.text(ChatPromptTemplate.from_messages([last_message]).pretty_repr())

def main():
    st.title("Diet Plan Assistant")
    st.write("Welcome to the Diet Plan Assistant! Please enter your preferences and dietary restrictions.")
    
    user_input = st.text_input("Enter your preferences and dietary restrictions:")
    
    if st.button("Submit"):
        stream_graph_update(user_input)
        st.write("Diet plan generated successfully!")
    
    if st.button("Reset"):
        grap.reset(confiq)
        st.write("Graph reset successfully!")

if __name__ == "__main__":
    main()