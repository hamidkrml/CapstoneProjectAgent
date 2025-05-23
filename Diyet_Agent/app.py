import streamlit as st
from typing import Annotated
from typing_extensions import TypedDict


from langchain_core.messages import AIMessage,HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_message
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class State(TypedDict):
    messages:Annotated[list,add_message]

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

llm = ChatOpenAI( model="gpt-4o")
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
    return ("messages", (llm.invoke(state["messages"])))

def reflection_node(state: State):

    cls_map = {
        "ai": HumanMessage, "human" : AIMessage }
    translated = [state["messages"][0]+ [
        cls_map[msg.type](content=msg.content) for msg in state["messages"][1:]
    ]]
    return translated
