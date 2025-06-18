from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, HTTPException
from ai_agent import get_response_from_ai_agent
import uvicorn
import json

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
            f"Diyet kısıtlamaları: {', '.join(user_data.dietary_restrictions) or 'Yok'}\n"
            f"Lütfen çıktıyı sadece aşağıdaki formatta JSON olarak döndür:\n"
            '{ "meals": { "breakfast": "...", "lunch": "...", "dinner": "..." }, "total_calories": 2000 }'
        )

        response = get_response_from_ai_agent(
            llm_id="llama-3.3-70b-versatile",
            query=[f"Hedef: {user_data.goal}"],
            system_prompt=system_prompt
        )

        # Ensure response is a string before parsing
        if not isinstance(response, str):
            response_str = json.dumps(response)
        else:
            response_str = response

        try:
            parsed = json.loads(response_str)
            return parsed
        except Exception:
            return {"diet_plan": response_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq hatası: {str(e)}")

# Markdown tablosu için yardımcı fonksiyon
def meal_plan_to_markdown(meal_plan):
    md = "| Öğün | Yemek Adı | Malzemeler | Tarif |\n"
    md += "|------|-----------|------------|-------|\n"
    for meal, details in meal_plan.get("meals", {}).items():
        name = details.get("name", "")
        ingredients = ", ".join(details.get("ingredients", []))
        instructions = " ".join(details.get("instructions", []))
        md += f"| {meal} | {name} | {ingredients} | {instructions} |\n"
    return md

@app.post("/generate-diet-plan-markdown")
async def generate_diet_plan_markdown(user_data: UserDietData):
    try:
        system_prompt = (
            f"Sen profesyonel bir diyetisyensin. {user_data.age} yaşında,\n"
            f"{user_data.weight} kg ağırlığında, {user_data.height} cm boyunda,\n"
            f"{user_data.gender} cinsiyetindeki, {user_data.activity_level} aktivite seviyesine sahip,\n"
            f"hedefi '{user_data.goal}' olan bir kişiye özel diyet planı hazırla.\n"
            f"Diyet kısıtlamaları: {', '.join(user_data.dietary_restrictions) or 'Yok'}\n"
            f"Lütfen çıktıyı sadece aşağıdaki formatta JSON olarak döndür:\n"
            '{ "meals": { "breakfast": "...", "lunch": "...", "dinner": "..." }, "total_calories": 2000 }'
        )

        response = get_response_from_ai_agent(
            llm_id="llama-3.3-70b-versatile",
            query=[f"Hedef: {user_data.goal}"],
            system_prompt=system_prompt
        )

        # Ensure response is a string before parsing
        if not isinstance(response, str):
            response_str = json.dumps(response)
        else:
            response_str = response

        try:
            parsed = json.loads(response_str)
            markdown = meal_plan_to_markdown(parsed)
            return {"markdown": markdown}
        except Exception:
            return {"markdown": f"```\n{response_str}\n```"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq hatası: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9949)