import os
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

def get_response_from_ai_agent(llm_id, query, system_prompt):
    try:
        llm = ChatGroq(
            model_name=llm_id,
            temperature=0.7,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        agent = create_react_agent(
            model=llm,
            tools=[]
        )

        messages = [
            SystemMessage(content=system_prompt),
            *[HumanMessage(content=msg) for msg in query]
        ]

        result = agent.invoke({"messages": messages})
        # result genellikle bir dict ve "messages" anahtarında mesajlar olur
        if isinstance(result, dict) and "messages" in result:
            ai_messages = [m.content for m in result["messages"] if isinstance(m, AIMessage)]
            return ai_messages[-1] if ai_messages else "Cevap oluşturulamadı"
        return "Cevap oluşturulamadı"

    except Exception as e:
        raise RuntimeError(f"Groq Hatası: {str(e)}")