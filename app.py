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
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the API key
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Differential diagnosis list
differential_diagnosis = [
    "Behavior", "Cardiology", "Dentistry", "Dermatology", "Endocrinology and Metabolism",
    "Gastroenterology", "Hematology/Immunology", "Hepatology", "Infectious Disease",
    "Musculoskeletal", "Nephrology/Urology", "Neurology", "Oncology", "Ophthalmology",
    "Respiratory", "Theriogenology", "Toxicology"
]

# Constants
MAX_AUDIO_DURATION = 120  # 2 minutes
CHUNK_DURATION = 60  # 1 minute
ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.webm'}
TEMP_AUDIO_PATH = "temp_audio.wav"

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

class AudioProcessingError(Exception):
    """Custom exception for audio processing errors"""
    pass

def validate_audio_file(filename: str) -> bool:
    """Validate audio file extension"""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def convert_to_wav(input_path: str, output_path: str) -> None:
    """Convert audio file to WAV format"""
    try:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
    except Exception as e:
        raise AudioProcessingError(f"Error converting audio: {str(e)}")

def split_audio_and_translate(audio_path: str) -> str:
    """
    Split audio file into chunks and translate using OpenAI
    """
    try:
        with wave.open(audio_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            frame_rate = wav_file.getframerate()
            total_duration = frames / float(frame_rate)

            if total_duration <= MAX_AUDIO_DURATION:
                # Audio is within limit, translate directly
                with open(audio_path, "rb") as audio_file:
                    client = OpenAI(api_key=api_key)
                    translation = client.audio.translations.create(
                        model="whisper-1",
                        file=audio_file
                    )
                    return translation.text

            # Split longer audio into chunks
            translated_text = ""
            chunk_size = int(CHUNK_DURATION * frame_rate)
            
            for i in range(0, frames, chunk_size):
                chunk_path = f"chunk_{i}.wav"
                try:
                    # Write chunk to temporary file
                    with wave.open(chunk_path, 'wb') as chunk_file:
                        chunk_file.setnchannels(wav_file.getnchannels())
                        chunk_file.setsampwidth(wav_file.getsampwidth())
                        chunk_file.setframerate(frame_rate)
                        wav_file.setpos(i)
                        chunk_file.writeframes(wav_file.readframes(chunk_size))
                    
                    # Translate chunk
                    chunk_translation = split_audio_and_translate(chunk_path)
                    translated_text += chunk_translation + " "
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
        result_queue.put(translation)
    except Exception as e:
        result_queue.put(f"Error: {str(e)}")

def create_soap_note_prompt() -> PromptTemplate:
    """Create the SOAP note prompt template"""
    template = """
    1. Role: You are a knowledgeable veterinarian assistant.

    2. Task: Convert the following doctor-patient dialogue and medical history into a SOAP note format.
       Doctor-patient dialogue and medical history: {full_text}
    
    3. Language: Ensure the use of correct medical terminology and formal language.
    
    4. Format: Use the SOAP note format (Subjective, Objective, Assessment, and Plan).
    
    5. Differential Diagnosis: Derive a differential diagnosis heading from the conversation in the format of "System-Condition".
    
    6. Comprehensiveness: Include all relevant diagnosis, treatment, and plan details from the conversation.
    
    7. Accuracy: Stick to the conversation transcribed and avoid any form of hallucination.
    
    8. Relevance: Focus on relevant medical aspects for veterinary SOAP notes.
    
    9. Professionalism: Follow professional veterinary documentation standards.
    
    10. Conclusion: Include a well-structured conclusion after the SOAP format.
    
    11. DifferentialDiagnosis: Select a primary condition/system from this list: {differential_diagnosis}
    
    Output Format:
    Subjective:
    [Subjective content here]

    Objective:
    [Objective content here]

    Assessment:
    [Assessment content here]

    Plan:
    [Plan content here]

    Conclusion:
    [Conclusion content here]

    DifferentialDiagnosis:
    [Differential Diagnosis here]
    """
    return PromptTemplate.from_template(template)

def extract_section(text: str, section: str) -> str:
    """Extract a section from the medical note text"""
    pattern = rf"{section}:(.*?)(?:\n\n|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def process_medical_note(text: str) -> Dict[str, str]:
    """Process the medical note text into structured format"""
    sections = ["Subjective", "Objective", "Assessment", "Plan", "Conclusion", "DifferentialDiagnosis"]
    return {section: extract_section(text, section) for section in sections}

@app.get("/")
async def read_root():
    """Root endpoint"""
    return {"message": "Welcome to the SOAP note generator API", "status": "active"}

@app.post("/soap_note/")
async def create_soap_note(
    audio_file: UploadFile = File(...),
    medical_history: str = Form(...)
) -> Dict[str, Any]:
    """
    Create a SOAP note from audio file and medical history
    """
    start_time = time.time()
    logger.info(f"Starting SOAP note generation for file: {audio_file.filename}")

    try:
        # Validate file
        if not validate_audio_file(audio_file.filename):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Supported formats: WAV, MP3, WEBM"
            )

        # Save and process audio file
        file_extension = os.path.splitext(audio_file.filename)[1].lower()
        temp_input_path = f"temp_input{file_extension}"
        
        try:
            # Save uploaded file
            with open(temp_input_path, "wb") as temp_file:
                temp_file.write(await audio_file.read())

            # Convert to WAV if needed
            if file_extension != '.wav':
                convert_to_wav(temp_input_path, TEMP_AUDIO_PATH)
            else:
                os.rename(temp_input_path, TEMP_AUDIO_PATH)

            # Translate audio
            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=process_audio_and_translate,
                args=(TEMP_AUDIO_PATH, result_queue)
            )
            process.start()
            process.join(timeout=300)  # 5 minutes timeout

            if process.is_alive():
                process.terminate()
                raise HTTPException(
                    status_code=408,
                    detail="Audio processing timeout"
                )

            translation = result_queue.get()
            if isinstance(translation, str) and translation.startswith("Error:"):
                raise HTTPException(
                    status_code=500,
                    detail=translation
                )

            # Combine medical history with translated text
            full_text = f"Medical History: {medical_history}\n\nConversation: {translation}"

            # Generate SOAP note
            llm = ChatOpenAI(model_name='gpt-4', api_key=api_key)
            prompt = create_soap_note_prompt()
            chain = LLMChain(llm=llm, prompt=prompt)
            
            medical_note_text = chain.run({
                "full_text": full_text,
                "differential_diagnosis": differential_diagnosis
            })

            # Process and structure the note
            medical_note = process_medical_note(medical_note_text)

            logger.info(f"SOAP note generated successfully. Time taken: {time.time() - start_time:.2f} seconds")
            return medical_note

        finally:
            # Cleanup temporary files
            for file_path in [temp_input_path, TEMP_AUDIO_PATH]:
                if os.path.exists(file_path):
                    os.remove(file_path)

    except Exception as e:
        logger.error(f"Error in SOAP note generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

