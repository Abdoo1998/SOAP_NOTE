from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import wave
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import os
import time
from pydub import AudioSegment
import multiprocessing
import json
import re
import logging
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Constants
MAX_AUDIO_DURATION = 120  # 2 minutes
CHUNK_DURATION = 60  # 1 minute
ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.webm'}
TEMP_AUDIO_PATH = "temp_audio.wav"

# Differential diagnosis list
differential_diagnosis = [
    "Behavior", "Cardiology", "Dentistry", "Dermatology", "Endocrinology and Metabolism",
    "Gastroenterology", "Hematology/Immunology", "Hepatology", "Infectious Disease",
    "Musculoskeletal", "Nephrology/Urology", "Neurology", "Oncology", "Ophthalmology",
    "Respiratory", "Theriogenology", "Toxicology"
]

class AudioProcessingError(Exception):
    """Custom exception for audio processing errors"""
    pass

# FastAPI app initialization
app = FastAPI(title="SOAP Note Generator API")

# CORS settings
origins = [
    "https://paws.vetinstant.com",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_audio_file(filename: str) -> bool:
    """Validate audio file extension"""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS

async def save_and_convert_audio(audio_file: UploadFile) -> str:
    """Save uploaded file and convert to WAV if necessary"""
    try:
        file_extension = os.path.splitext(audio_file.filename)[1].lower()
        temp_input_path = f"temp_input{file_extension}"
        
        # Save uploaded file
        content = await audio_file.read()
        with open(temp_input_path, "wb") as temp_file:
            temp_file.write(content)

        # Convert to WAV if needed
        if file_extension != '.wav':
            audio = AudioSegment.from_file(temp_input_path)
            audio.export(TEMP_AUDIO_PATH, format="wav")
            os.remove(temp_input_path)
        else:
            os.rename(temp_input_path, TEMP_AUDIO_PATH)
        
        return TEMP_AUDIO_PATH
    except Exception as e:
        raise AudioProcessingError(f"Error processing audio file: {str(e)}")

def split_audio_and_translate(audio_path: str) -> str:
    """Split audio file into chunks and translate using OpenAI"""
    try:
        with wave.open(audio_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            frame_rate = wav_file.getframerate()
            total_duration = frames / float(frame_rate)

            if total_duration <= MAX_AUDIO_DURATION:
                with open(audio_path, "rb") as audio_file:
                    client = OpenAI(api_key=api_key)
                    translation = client.audio.translations.create(
                        model="whisper-1",
                        file=audio_file
                    )
                    return translation.text

            translated_text = ""
            chunk_size = int(CHUNK_DURATION * frame_rate)
            
            for i in range(0, frames, chunk_size):
                chunk_path = f"chunk_{i}.wav"
                try:
                    with wave.open(chunk_path, 'wb') as chunk_file:
                        chunk_file.setnchannels(wav_file.getnchannels())
                        chunk_file.setsampwidth(wav_file.getsampwidth())
                        chunk_file.setframerate(frame_rate)
                        wav_file.setpos(i)
                        chunk_file.writeframes(wav_file.readframes(chunk_size))
                    
                    chunk_translation = split_audio_and_translate(chunk_path)
                    translated_text += " " + chunk_translation
                finally:
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)

            return translated_text.strip()
    except Exception as e:
        raise AudioProcessingError(f"Error processing audio: {str(e)}")

def process_audio_and_translate(audio_path: str, result_queue: multiprocessing.Queue) -> None:
    """Process audio in a separate process"""
    try:
        translation = split_audio_and_translate(audio_path)
        result_queue.put({"success": True, "text": translation})
    except Exception as e:
        result_queue.put({"success": False, "error": str(e)})

def extract_section(text: str, section: str) -> str:
    """Extract a section from the medical note text"""
    pattern = rf"{section}:(.*?)(?:\n\n|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else "[Null]"

def process_medical_note(text: str) -> Dict[str, str]:
    """Process the medical note text into structured format"""
    sections = [
        "Subjective", "Objective", "Assessment", "Plan", 
        "Conclusion", "Differentialdiagnosis", "Preventive",
        "Prescription", "Dietrecommendations", "Diagnostics"
    ]
    return {section: extract_section(text, section) for section in sections}

@app.get("/")
async def read_root():
    """Root endpoint"""
    return {
        "message": "Welcome to the SOAP note generator API",
        "status": "active",
        "version": "2.0"
    }

@app.post("/soap_note/")
async def create_soap_note(
    audio_file: UploadFile = File(...),
    medical_history: str = Form(...)
) -> Dict[str, Any]:
    """Create a SOAP note from audio file and medical history"""
    start_time = time.time()
    logger.info(f"Starting SOAP note generation for file: {audio_file.filename}")

    try:
        # Validate file
        if not validate_audio_file(audio_file.filename):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Supported formats: WAV, MP3, WEBM"
            )

        # Save and convert audio
        temp_audio_path = await save_and_convert_audio(audio_file)

        # Translate audio
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=process_audio_and_translate,
            args=(temp_audio_path, result_queue)
        )
        process.start()
        process.join(timeout=300)  # 5 minutes timeout

        if process.is_alive():
            process.terminate()
            raise HTTPException(
                status_code=408,
                detail="Audio processing timeout"
            )

        result = result_queue.get()
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Audio processing error: {result['error']}"
            )

        # Combine medical history with translated text
        full_text = f"Medical History: {medical_history}\n\nConversation: {result['text']}"

        # Initialize OpenAI and create prompt
        llm = ChatOpenAI(model_name='gpt-4', api_key=api_key)
        
        # Create prompt template with proper escaping
        prompt = PromptTemplate.from_template("""
            You are a knowledgeable veterinary assistant. Convert this input into a SOAP note:
            
            Input: {full_text}
            
            Use these categories for differential diagnosis: {differential_diagnosis}
            
            Format the response with these sections:
            
            Subjective: (patient history and symptoms)
            Objective: (examination findings)
            Assessment: (diagnosis and evaluation)
            Plan: (treatment and recommendations)
            Conclusion: (summary)
            Differentialdiagnosis: (primary condition)
            Preventive: (vaccinations, deworming, etc.)
            Prescription: (medications)
            Dietrecommendations: (feeding guidelines)
            Diagnostics: (tests performed/needed)
            
            If no medical content is found, use "[Null]" for empty sections.
        """)

        # Generate SOAP note
        chain = LLMChain(llm=llm, prompt=prompt)
        medical_note_text = chain.run({
            "full_text": full_text,
            "differential_diagnosis": differential_diagnosis
        })

        # Process and structure the note
        medical_note = process_medical_note(medical_note_text)

        logger.info(f"SOAP note generated successfully. Time taken: {time.time() - start_time:.2f} seconds")
        return medical_note

    except Exception as e:
        logger.error(f"Error in SOAP note generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )
    finally:
        # Cleanup temporary files
        if os.path.exists(TEMP_AUDIO_PATH):
            os.remove(TEMP_AUDIO_PATH)
