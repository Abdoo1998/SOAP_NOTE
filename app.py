from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Form
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

# Load the API key
api_key = os.environ.get('OPENAI_API_KEY')

# Differential diagnosis list
differential_diagnosis = [
    "Behavior", "Cardiology", "Dentistry", "Dermatology", "Endocrinology and Metabolism",
    "Gastroenterology", "Hematology/Immunology", "Hepatology", "Infectious Disease",
    "Musculoskeletal", "Nephrology/Urology", "Neurology", "Oncology", "Ophthalmology",
    "Respiratory", "Theriogenology", "Toxicology"
]

# Maximum duration for audio processing
max_duration = 120  # 2 minutes

# FastAPI app initialization
app = FastAPI()

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

def split_audio_and_translate(audio_path):
    """
    Splits audio file into chunks, translates each chunk using OpenAI,
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
            except Exception as e:
                print(f"OpenAI Error during translation: {e}")
                return ""

        else:
            # Split audio into chunks and translate each
            desired_duration = 60  # 1 minute
            chunk_size = int(desired_duration * frame_rate)
            translated_text = ""
            with wave.open(audio_path, 'rb') as wav_file:
                for i in range(0, frames, chunk_size):
                    chunk_data = wav_file.readframes(chunk_size)
                    with wave.open(f"chunk_{i}.wav", 'wb') as chunk_file:
                        chunk_file.setnchannels(wav_file.getnchannels())
                        chunk_file.setsampwidth(wav_file.getsampwidth())
                        chunk_file.setframerate(frame_rate)
                        chunk_file.writeframes(chunk_data)
                    try:
                        chunk_translation = split_audio_and_translate(f"chunk_{i}.wav")
                        translated_text += chunk_translation
                    except Exception as e:
                        print(f"OpenAI Error during chunk translation: {e}")
                    os.remove(f"chunk_{i}.wav")
            return translated_text

def process_audio_and_translate(audio_path, result_queue):
    translation = split_audio_and_translate(audio_path)
    result_queue.put(translation)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the SOAP note generator API"}

@app.post("/soap_note/")
async def create_soap_note(
    audio_file: UploadFile = File(...),
    medical_history: str = Form(...)
):
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
            audio.export(temp_audio_path, format="wav")
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

    # Combine medical history with translated text
    full_text = f"Medical History: {medical_history}\n\nConversation: {translation}"

    # OpenAI model initialization
    openai_time = time.time()
    openai = ChatOpenAI(model_name='gpt-4', api_key=api_key)
    openai_init_time = time.time() - openai_time
    print(f"OpenAI model initialization time: {openai_init_time} seconds")

    # Generating SOAP note prompt
    prompt_time = time.time()
    conversation_prompt = PromptTemplate.from_template("""
        1. Role: You are a knowledgeable veterinarian assistant.

        2. Task: Convert the following doctor-patient dialogue and medical history into a SOAP note format.
           Doctor-patient dialogue and medical history: {full_text}
        
        3. Language: Ensure the use of correct medical terminology and formal language.
        
        4. Format: Use the SOAP note format (Subjective, Objective, Assessment, Plan, DeferentialDiagnosis, and Conclusion).
        
        5. DeferentialDiagnosis: This section is crucial and must be included. Follow these guidelines:
           a. Provide at least one deferential diagnosis, and up to a maximum of five when possible.
           b. Format each diagnosis as "System - Condition". For example: "Dermatology - Atopic Dermatitis".
           c. List the deferential diagnoses in order of likelihood, with the most probable first.
           d. If the information is insufficient for multiple diagnoses, provide at least one potential diagnosis or area of concern.
           e. If no clear deferential diagnosis can be determined, explicitly state: "Insufficient information for a deferential diagnosis. Further diagnostics recommended."
           f. Ensure that the deferential diagnoses are relevant to the systems mentioned in this list: {differential_diagnosis}
        
        6. Comprehensiveness: Ensure that all aspects relevant to the diagnosis, treatment, and plan from the conversation are added to the medical documentation.
        
        7. Accuracy: Stick to the conversation transcribed and avoid any form of hallucination.
        
        8. Relevance: Include relevant medical aspects to veterinary SOAP notes and avoid all other general conversations.
        
        9. Professionalism: Ensure that the medical documentation follows a professional veterinary documentation standard.
        
        10. Conclusion: Include a well-structured conclusion after the DeferentialDiagnosis section.
        
        11. System Instructions:
            a. Ensure that the conclusion is always correctly derived and published.
            b. Stick to the conversation transcribed and avoid any form of hallucination.
            c. Always include a DeferentialDiagnosis section, even if it's to state that further diagnostics are needed.
        
        12. Output Format: Structure your response as follows:
            Subjective:
            [Subjective content here]

            Objective:
            [Objective content here]

            Assessment:
            [Assessment content here]

            Plan:
            [Plan content here]

            DeferentialDiagnosis:
            [DeferentialDiagnosis content here, following the guidelines in point 5]

            Conclusion:
            [Conclusion content here]
        
        Remember to maintain a professional tone throughout the document and ensure all information is relevant to veterinary practice.
    """)
    prompt_generation_time = time.time() - prompt_time
    print(f"Prompt generation time: {prompt_generation_time} seconds")

    # Generating medical note
    process_chain_time = time.time()
    process_conversation_chain = LLMChain(
        llm=openai, prompt=conversation_prompt
    )

    data = {
        "full_text": full_text,
        "differential_diagnosis": differential_diagnosis
    }

    medical_note_text = process_conversation_chain.invoke(data)
    process_chain_execution_time = time.time() - process_chain_time
    print(f"Process chain execution time: {process_chain_execution_time} seconds")

    # Post-process the output to ensure correct format
    def extract_section(text, section):
        pattern = rf"{section}:(.*?)(?:\n\n|\Z)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""

    medical_note = {
        "Subjective": extract_section(medical_note_text, "Subjective"),
        "Objective": extract_section(medical_note_text, "Objective"),
        "Assessment": extract_section(medical_note_text, "Assessment"),
        "Plan": extract_section(medical_note_text, "Plan"),
        "Conclusion": extract_section(medical_note_text, "Conclusion"),
        "DifferentialDiagnosis": extract_section(medical_note_text, "DifferentialDiagnosis")
    }

    # Ensure DifferentialDiagnosis is not empty
    if not medical_note["DifferentialDiagnosis"] or medical_note["DifferentialDiagnosis"].lower() in ["none", "n/a", ""]:
        # If DifferentialDiagnosis is empty, try to generate one based on the Assessment
        assessment = medical_note["Assessment"]
        if assessment:
            # Use GPT to generate a differential diagnosis based on the assessment
            diff_diag_prompt = PromptTemplate.from_template("""
                Based on the following assessment, provide a differential diagnosis 
                If no clear differential diagnosis can be determined, state "No clear differential diagnosis can be determined based on the given information."
                
                Assessment: {assessment}
                
                DifferentialDiagnosis:
            """)
            diff_diag_chain = LLMChain(llm=openai, prompt=diff_diag_prompt)
            diff_diag = diff_diag_chain.run({"assessment": assessment})
            medical_note["DifferentialDiagnosis"] = diff_diag.strip()
        else:
            medical_note["DifferentialDiagnosis"] = "No clear differential diagnosis can be determined based on the given information."

    os.remove(temp_audio_path)

    total_time = time.time() - start_time
    print(f"Total time taken: {total_time} seconds")

    return medical_note

