# file: lc_gemini_intent.py
import os
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field, confloat
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

load_dotenv()

INTENTS = ["service_request","hotel_info","internal_experience","external_experience","booking","off_topic","feedback"]

class IntentPred(BaseModel):
    intent: Literal[tuple(INTENTS)] = Field(...)
    confidence: confloat(ge=0, le=1) = Field(...)

parser = PydanticOutputParser(pydantic_object=IntentPred)

SYSTEM = (
    "You are an intent classifier for a hotel concierge assistant. "
    "Choose exactly one intent from the allowed set and return ONLY one JSON object."
)

# IMPORTANT: escape braces {{ }} in examples
FEW_SHOTS = """
Examples (input → output):
- "Can I get two extra towels please?" → {{"intent":"service_request","confidence":0.92}}
- "What time does the pool close?" → {{"intent":"hotel_info","confidence":0.88}}
- "Is there a yoga session tomorrow morning?" → {{"intent":"internal_experience","confidence":0.86}}
- "We want a Delhi food walk this evening" → {{"intent":"external_experience","confidence":0.87}}
- "Book a couple spa at 5 PM" → {{"intent":"booking","confidence":0.91}}
- "haha you’re funny 😂" → {{"intent":"off_topic","confidence":0.7}}
- "this hotel is trash" → {{"intent":"feedback","confidence":0.7}}
"""

INSTRUCTIONS = (
    "Allowed intents: service_request, hotel_info, internal_experience, external_experience, booking, off_topic,, feedback.\n"
    "Return a single JSON object that matches the schema below.\n"
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

def predict_intent_with_gemini(text: str) -> IntentPred:
    return chain.invoke({"text": text})

# if __name__ == "__main__":
#     print(predict_intent_with_gemini("Please arrange room cleaning at 3 pm"))
