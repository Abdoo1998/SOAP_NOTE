from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os
import subprocess
import wave
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# Replace with your actual OpenAI API key
api_key = "sk-proj-UCEe8mKLL1NBsF2JGOSDT3BlbkFJFhLtjUUwsRQ7zUYEnEuA"

# Define maximum audio duration (in seconds)
max_duration = 120  # 2 minutes

app = FastAPI()

class AudioResponse(BaseModel):
    translated_text: str

def convert_to_wav(input_file, output_file):
    try:
        subprocess.run(["ffmpeg", "-i", input_file, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_file], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting audio to WAV: {e}")
        return False

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

@app.post("/translate_audio/")
async def translate_audio(audio_file: UploadFile = File(...)):
    # Save audio file
    with open("temp_audio", "wb") as temp_audio:
        temp_audio.write(await audio_file.read())

    # Convert audio to WAV format
    converted_audio_file = "temp_audio.wav"
    if not convert_to_wav("temp_audio", converted_audio_file):
        return {"error": "Failed to convert audio to WAV format"}

    # Translate audio
    conversation = split_audio_and_translate(converted_audio_file)
    openai = ChatOpenAI(model_name='gpt-4',api_key=api_key)

    conversation_prompt = PromptTemplate.from_template(f"""Convert the following doctor-patient dialogue ({conversation}) into a SOAP note format, ensuring correct medical terminology and including a well-structured conclusion.""")

    process_conversation_chain = LLMChain(
    llm=openai, prompt=conversation_prompt)

    data = {"conversation": conversation}

    medical_note = process_conversation_chain.run(data)

    return {"medical_note": medical_note}
