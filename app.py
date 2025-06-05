from fastapi import FastAPI
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sysmam.com.ar"],  # O ["*"] para desarrollo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


api_key = os.getenv('key')
model = "deepseek-r1-distill-llama-70b"
chat = ChatGroq(api_key=api_key, model_name=model)
parser = StrOutputParser()
chain = chat | parser

# Cargar contexto desde archivo
loader = TextLoader("contexto.txt", encoding="utf-8")
context_data = loader.load()

context = context_data[0].page_content if context_data else ""

template = """
Eres un chatbot con inteligencia artificial diseñado para proporcionar 
información y asistencia a los usuarios. 
Debes responder ÚNICAMENTE en español Argentino y basarte estrictamente 
en el contexto proporcionado. No inventes información.
NO INCLUYAS EL THINK O RAZONAMIENTO EN LA RESPUESTA.

Contexto: {context}
Pregunta: {question}
"""

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    prompt = template.format(context=context, question=query.question)
    response = chain.invoke(prompt)
    return {"response": response}
