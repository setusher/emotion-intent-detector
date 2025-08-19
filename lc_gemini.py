
import os
from typing import Literal
from dotenv import load_dotenv

from pydantic import BaseModel, Field, confloat
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

load_dotenv()


class EmotionPred(BaseModel):
    label: Literal["anger", "fear", "joy", "love", "sadness", "surprise", "neutral"] = Field(...)
    confidence: confloat(ge=0, le=1) = Field(..., description="Model self-reported confidence 0..1")

parser = PydanticOutputParser(pydantic_object=EmotionPred)

SYSTEM = (
    "You are an emotion classifier. "
    "Only consider the text content (do not infer from emojis beyond plain text). "
    "Return a single JSON object ONLY that matches the schema."
)


FEW_SHOTS = """
Examples (input → output):
- "I can't stop smiling today!!" → {{"label":"joy","confidence":0.90}}
- "You are the best, thank you so much" → {{"label":"love","confidence":0.85}}
- "I feel so empty and low" → {{"label":"sadness","confidence":0.88}}
- "This is terrifying..." → {{"label":"fear","confidence":0.86}}
- "How dare you lose my bag!" → {{"label":"anger","confidence":0.90}}
- "No way!! That's unbelievable!" → {{"label":"surprise","confidence":0.83}}
- "I was sad yesterday but I am slightly better today" → {{"label":"neutral","confidence":0.75}}
"""

INSTRUCTIONS = (
    "Choose the most likely single emotion from: anger, fear, joy, love, sadness, surprise, neutral.\n"
    "Estimate confidence between 0 and 1. Keep JSON strictly parsable.\n"
    "{format_instructions}\n"
    + FEW_SHOTS
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Text: {text}\n" + INSTRUCTIONS)
]).partial(format_instructions=parser.get_format_instructions())

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2,  
)


chain = prompt | llm | parser

def predict_with_gemini(text: str) -> EmotionPred:
    return chain.invoke({"text": text})

if __name__ == "__main__":
    print(predict_with_gemini("I am a little sad and a little happy"))
