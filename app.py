from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import wave
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import os
import time
import wave
from pydub import AudioSegment

api_key = os.environ.get('OPENAI_API_KEY')
defrentail_daignossi = [
    "Behavior",
    "Cardiology",
    "Dentistry",
    "Dermatology",
    "Endocrinology and Metabolism",
    "Gastroenterology",
    "Hematology/Immunology",
    "Hepatology",
    "Infectious Disease",
    "Musculoskeletal",
    "Nephrology/Urology",
    "Neurology",
    "Oncology",
    "Ophthalmology",
    "Respiratory",
    "Theriogenology",
    "Toxicology"
]

max_duration = 120  # 2 minutes

app = FastAPI()

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

def split_audio_and_translate(audio_path):
    """
    Splits audio file into chunks (around 1 minute), translates each chunk using OpenAI,
    and concatenates translations. Handles short audio chunks gracefully.
    Returns the complete translated text.
    """
    with wave.open(audio_path, 'rb') as wav_file:
        frames = wav_file.getnframes()
        frame_rate = wav_file.getframerate()
        total_duration = frames / float(frame_rate)

        if total_duration <= max_duration:
            # Audio is within limit, translate directly
            audio_file = open(audio_path, "rb")
            try:
                translation = OpenAI(api_key=api_key).audio.translations.create(
                    model="whisper-1",
                    file=audio_file
                )
                return translation.text
            except Exception as e:  # Catch potential OpenAI errors
                print(f"OpenAI Error during translation: {e}")
                return ""  # Handle error by returning empty string

        else:
            # Split audio into chunks (around 1 minute) and translate each
            desired_duration = 60  # 1 minute (adjust for potential variations)
            chunk_size = int(desired_duration * frame_rate)
            translated_text = ""
            short_chunks = []  # List to store empty translations for short chunks
            with wave.open(audio_path, 'rb') as wav_file:
                for i in range(0, frames, chunk_size):
                    # Read chunk data
                    chunk_data = wav_file.readframes(chunk_size)
                    # Create temporary audio file for the chunk
                    with wave.open(f"chunk_{i}.wav", 'wb') as chunk_file:
                        chunk_file.setnchannels(wav_file.getnchannels())
                        chunk_file.setsampwidth(wav_file.getsampwidth())
                        chunk_file.setframerate(frame_rate)
                        chunk_file.writeframes(chunk_data)
                    # Translate the chunk
                    try:
                        chunk_translation = split_audio_and_translate(f"chunk_{i}.wav")
                        translated_text += chunk_translation
                    except Exception as e:  # Catch potential OpenAI errors
                        print(f"OpenAI Error during chunk translation: {e}")
                        short_chunks.append("")  # Store empty string for short chunk
                    # Remove temporary audio file
                    os.remove(f"chunk_{i}.wav")
            if short_chunks:
                print(f"Warning: Encountered {len(short_chunks)} short audio chunks.")
            return translated_text



import multiprocessing

def process_audio_and_translate(audio_path, result_queue):
    translation = split_audio_and_translate(audio_path)
    result_queue.put(translation)
@app.get("/")
async def read_root():
    return {"message": "Welcome to the SOAP note generator API"}

@app.post("/soap_note/")
async def create_soap_note(audio_file: UploadFile = File(...)):
    start_time = time.time()

    # Check file extension
    file_extension = os.path.splitext(audio_file.filename)[1].lower()

    # Save audio file
    temp_audio_path = "temp_audio.wav"
    if file_extension == ".mp3":
        with open("temp_audio.mp3", "wb") as temp_audio:
            temp_audio.write(await audio_file.read())
        # Convert MP3 to WAV
        audio = AudioSegment.from_mp3("temp_audio.mp3")
        audio.export(temp_audio_path, format="wav")
        os.remove("temp_audio.mp3")
        
    elif file_extension == ".webm":
        # Extract audio from webm using pydub
        
        try:
            audio = AudioSegment.from_file(audio_file.file)
            audio.export("temp_audio.wav", format="wav")
        except Exception as e:
            print(f"Error extracting audio from webm: {e}")
            return {"error": "Failed to process webm file. Please ensure it's a valid webm audio format."}
    
    else:
        with open(temp_audio_path, "wb") as temp_audio:
            temp_audio.write(await audio_file.read())

    audio_process_time = time.time() - start_time
    print(f"Audio processing time: {audio_process_time} seconds")

    # Translate audio
    translate_start_time = time.time()
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=process_audio_and_translate, args=(temp_audio_path, result_queue))
    process.start()
    process.join()
    translation = result_queue.get()
    translate_time = time.time() - translate_start_time
    print(f"Translation time: {translate_time} seconds")

    # OpenAI model initialization
    openai_time = time.time()
    openai = ChatOpenAI(model_name='gpt-4o', api_key=api_key)
    openai_init_time = time.time() - openai_time
    print(f"OpenAI model initialization time: {openai_init_time} seconds")

    # Generating SOAP note prompt
    prompt_time = time.time()
    conversation_prompt = PromptTemplate.from_template(f"""
                                               1. Role: You are a knowledgeable veterinarian assistant.

                                                2. Task: Convert the following doctor-patient dialogue into a SOAP note format.
                                                   Doctor-patient dialogue: {{translation}}
                                                
                                                3. Language: Ensure the use of correct medical terminology and formal language.
                                                
                                                4. Format: Use the SOAP note format (Subjective, Objective, Assessment, and Plan).
                                                
                                                5. Differential Diagnosis: Derive a differential diagnosis heading from the conversation in the format of "System-Condition". Example: Dermatology-Atopic Dermatitis.
                                                
                                                6. Comprehensiveness: Ensure that all aspects relevant to the diagnosis, treatment, and plan from the conversation are added to the medical documentation.
                                                
                                                7. Accuracy: Stick to the conversation transcribed and avoid any form of hallucination.
                                                
                                                8. Relevance: Include relevant medical aspects to veterinary SOAP notes and avoid all other general conversations.
                                                
                                                9. Professionalism: Ensure that the medical documentation follows a professional veterinary documentation standard.
                                                
                                                10. Conclusion: Include a well-structured conclusion after the SOAP format.
                                                
                                                11. System Instructions:
                                                    a. Ensure that the conclusion is always correctly derived and published.
                                                    b. Ensure that the differential diagnosis derived from the conversation is always in the form this list: {defrentail_daignossi}
                                                    c. Stick to the conversation transcribed and avoid any form of hallucination.
                                                
                                                
                                                13. Structure:
                                                    a. The SOAP note should be structured as paragraphs and contain all the necessary information.
                                                    b. Additional information should be structured as bullet points.
                                                    c. Do not include the conclusion in the additional information.
                                                
                                                Remember to maintain a professional tone throughout the document and ensure all information is relevant to veterinary practice.
                                                                                                Dont include conclusion in the additional information
                                                """)
    prompt_generation_time = time.time() - prompt_time
    print(f"Prompt generation time: {prompt_generation_time} seconds")

    # Generating medical note
    process_chain_time = time.time()
    process_conversation_chain = LLMChain(
    llm=openai, prompt=conversation_prompt)

    data = {"conversation": translation}

    medical_note = process_conversation_chain.run(data)
    process_chain_execution_time = time.time() - process_chain_time
    print(f"Process chain execution time: {process_chain_execution_time} seconds")

    os.remove(temp_audio_path)

    total_time = time.time() - start_time
    print(f"Total time taken: {total_time} seconds")

    return {"medical_note": medical_note}
