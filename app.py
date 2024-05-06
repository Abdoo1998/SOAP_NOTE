

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import wave
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import os
import wave
from pydub import AudioSegment

api_key = os.environ.get('OPENAI_API_KEY')


max_duration = 120  # 2 minutes

app = FastAPI()

class AudioResponse(BaseModel):
    translated_text: str

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

@app.post("/soap_note/")
async def create_soap_note(audio_file: UploadFile = File(...)):
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
    else:
        with open(temp_audio_path, "wb") as temp_audio:
            temp_audio.write(await audio_file.read())

    # Translate audio
    conversation = split_audio_and_translate(temp_audio_path)
    openai = ChatOpenAI(model_name='gpt-4',api_key=api_key)

    conversation_prompt = PromptTemplate.from_template(f"""Convert the following doctor-patient dialogue ({conversation}) into a SOAP note format, ensuring correct medical terminology and including a well-structured conclusion.""")

    process_conversation_chain = LLMChain(
    llm=openai, prompt=conversation_prompt)

    data = {"conversation": conversation}

    medical_note = process_conversation_chain.run(data)
    os.remove(temp_audio_path)

    return {"medical_note": medical_note}
