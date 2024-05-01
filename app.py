from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import wave
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import io
import os
from pydub import AudioSegment
from tempfile import NamedTemporaryFile

# Replace with your actual OpenAI API key
api_key = os.environ.get('APENAI_API_KEY')

# Check if the API key is available
if api_key is None:
    print("Error: Apenai API key is not set.")
    exit()

# Define maximum audio duration (in seconds)
max_duration = 120  # 2 minutes

app = FastAPI()

class AudioResponse(BaseModel):
    translated_text: str

def convert_to_wav(audio_data):
    """
    Converts audio data to WAV format.
    """
    with NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_file:
        temp_wav_filename = temp_wav_file.name
        audio = AudioSegment.from_file(io.BytesIO(audio_data))
        audio.export(temp_wav_filename, format="wav")
    return temp_wav_filename

def split_audio_and_translate(audio_data):
    """
    Splits audio data into chunks (around 1 minute), translates each chunk using OpenAI,
    and concatenates translations. Handles short audio chunks gracefully.
    Returns the complete translated text.
    """
    wav_filename = convert_to_wav(audio_data)
    with wave.open(wav_filename, 'rb') as wav_file:
        frames = wav_file.getnframes()
        frame_rate = wav_file.getframerate()
        total_duration = frames / float(frame_rate)

        if total_duration <= max_duration:
            # Audio is within limit, translate directly
            try:
                translation = OpenAI(api_key=api_key).audio.translations.create(
                    model="whisper-1",
                    file=wav_filename
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
            for i in range(0, frames, chunk_size):
                # Read chunk data
                chunk_data = wav_file.readframes(chunk_size)
                # Translate the chunk
                try:
                    chunk_translation = split_audio_and_translate(chunk_data)
                    translated_text += chunk_translation
                except Exception as e:  # Catch potential OpenAI errors
                    print(f"OpenAI Error during chunk translation: {e}")
                    short_chunks.append("")  # Store empty string for short chunk
            if short_chunks:
                print(f"Warning: Encountered {len(short_chunks)} short audio chunks.")
            os.remove(wav_filename)
            return translated_text

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/soap_note/")
async def translate_audio(audio_file: UploadFile = File(...)):
    # Read audio file data into memory
    audio_data = await audio_file.read()

    # Translate audio
    conversation = split_audio_and_translate(audio_data)
    openai = ChatOpenAI(model_name='gpt-4',api_key=api_key)

    conversation_prompt = PromptTemplate.from_template(f"""Convert the following doctor-patient dialogue ({conversation}) into a SOAP note json format, ensuring correct medical terminology and including a well-structured conclusion.""")

    process_conversation_chain = LLMChain(
    llm=openai, prompt=conversation_prompt)

    data = {"conversation": conversation}

    medical_note = process_conversation_chain.run(data)

    # Delete temporary WAV file
    # os.remove(wav_filename)

    return {"medical_note": medical_note}
