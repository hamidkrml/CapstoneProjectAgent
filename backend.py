from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, HTTPException
from ai_agent import get_response_from_ai_agent
import uvicorn

class UserDietData(BaseModel):
    age: int
    weight: float
    height: float
    gender: str
    activity_level: str
    goal: str = "maintain"
    dietary_restrictions: List[str] = []

app = FastAPI(title="Groq Diyet Asistanı")

@app.post("/generate-diet-plan")
async def generate_diet_plan(user_data: UserDietData):
    try:
        system_prompt = (
            f"Sen profesyonel bir diyetisyensin. {user_data.age} yaşında,\n"
            f"{user_data.weight} kg ağırlığında, {user_data.height} cm boyunda,\n"
            f"{user_data.gender} cinsiyetindeki, {user_data.activity_level} aktivite seviyesine sahip,\n"
            f"hedefi '{user_data.goal}' olan bir kişiye özel diyet planı hazırla.\n"
            f"Diyet kısıtlamaları: {', '.join(user_data.dietary_restrictions) or 'Yok'}"
        )

        response = get_response_from_ai_agent(
            llm_id="llama-3.3-70b-versatile",
            query=[f"Hedef: {user_data.goal}"],
            system_prompt=system_prompt
        )
        
        return {"diet_plan": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq hatası: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9949)