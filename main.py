from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
import os
from typing import Dict
from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Google AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

# Pydantic models
class ChatMessage(BaseModel):
    message: str

class HealthData(BaseModel):
    bloodPressure: str
    bloodSugar: str
    cholesterol: str
    heartRate: str
    temperature: str

# Root endpoint
@app.get("/")
@app.head("/")
async def read_root():
    return {"message": "API is working!"}

# CORS pre-flight handlers
@app.options("/api/chat/")
async def handle_options_chat():
    return JSONResponse(content={}, status_code=200)

@app.options("/api/predict/")
async def handle_options_predict():
    return JSONResponse(content={}, status_code=200)

# Chat endpoint
@app.post("/api/chat")
async def chat(message: ChatMessage) -> Dict[str, str]:
    prompt = f"""
    You are a medical AI assistant. Provide helpful but general health information.
    Always include a disclaimer that this is not professional medical advice.
    
    User question: {message.message}
    """
    
    try:
        response = model.generate_content(prompt)
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in processing the request: {str(e)}")

# Health prediction endpoint
@app.post("/api/predict")
async def predict(data: HealthData) -> Dict[str, str]:
    prompt = f"""
    Analyze the following health metrics and provide general health insights:
    - Blood Pressure: {data.bloodPressure} mmHg
    - Blood Sugar: {data.bloodSugar} mg/dL
    - Cholesterol: {data.cholesterol} mg/dL
    - Heart Rate: {data.heartRate} bpm
    - Temperature: {data.temperature} °F
    
    Provide a general health assessment and suggestions for maintaining or improving health.
    Include a disclaimer about consulting healthcare professionals.
    """
    
    try:
        response = model.generate_content(prompt)
        return {"prediction": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in processing the request: {str(e)}")

# Document processing functions
def extract_text_from_pdf(pdf_file: BytesIO) -> str:
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(docx_file: BytesIO) -> str:
    doc = Document(docx_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_txt(txt_file: BytesIO) -> str:
    return txt_file.read().decode("utf-8")

# Document processing endpoint
@app.post("/api/process_document")
async def process_document(file: UploadFile = File(...)) -> Dict[str, str]:
    try:
        content = await file.read()
        file_extension = file.filename.split(".")[-1].lower()

        if file_extension == "pdf":
            text = extract_text_from_pdf(BytesIO(content))
        elif file_extension == "docx":
            text = extract_text_from_docx(BytesIO(content))
        elif file_extension == "txt":
            text = extract_text_from_txt(BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Only PDF, DOCX, and TXT are allowed.")

        prompt = f"""
        You are a medical AI assistant. Analyze the following medical document and provide a summary:
        {text}
        Always include a disclaimer that this is not professional medical advice.
        """

        try:
            response = model.generate_content(prompt)
            return {"summary": response.text}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error in processing the request: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
