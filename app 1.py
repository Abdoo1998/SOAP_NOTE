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
    openai = ChatOpenAI(model_name='gpt-4o', api_key=api_key)
    openai_init_time = time.time() - openai_time
    print(f"OpenAI model initialization time: {openai_init_time} seconds")

    # Generating SOAP note prompt
    prompt_time = time.time()
    conversation_prompt = PromptTemplate.from_template("""
    Role:

    You are a knowledgeable veterinary assistant responsible for converting a pre-appointment complaint summary and a doctor-patient conversation into a professional SOAP note format.

    Input: {full_text}

    Your task is to extract relevant information from the provided materials and organize it into the SOAP notes (Subjective, Objective, Assessment, Plan) structure, ensuring accuracy and professionalism.

    Utilize a structured, step by step approach thought process and the SOAP note components to ensure all relevant medical details are accurately documented. Consider the pre-appointment complaint summary, if available, as a context for the conversation. If the conversation lacks medically relevant content, return a null SOAP note.

    Additionally, you need to identify and extract specific action items such as Vaccinations, deworming, flea and tick treatments, prescriptions, diet recommendations, and diagnostic tests from the conversation and present them separately.

    Step 1: Initial Content Verification

    Purpose: Determine if the transcription contains medically relevant information.

    Actions: Analyze the conversation: Review the doctor-patient conversation for medical history, symptoms, assessments, diagnoses, treatments, or any other medically relevant content.

    Decision Point:
    If medically relevant content is present in the source: Proceed to Step 2.

    If medically relevant content is absent in the source: Output a response for the SOAP note stating that: No medically relevant information was provided for each of the content sections.

    Exclusion: Do not consider the pre-appointment complaint summary for this step. Focus only on the transcription provided to determine whether medical relevant content is available.

    Step 2: Use the Pre-Appointment Complaint Summary, if available:

    Where the pre-appointment complaint summary is provided, integrate it as Context: Use the pre-appointment complaint summary to provide essential background and context in the SOAP note.

    Highlight Key Information: Pay special attention to initial symptoms, concerns, and observations noted by the pet owner before the appointment.

    Ensure Continuity: Refer to the pre-appointment information throughout the SOAP note where relevant, especially in the Subjective and Assessment sections.

    Step 3: Subjective Information Extraction

    Purpose: Identify and extract subjective information provided by the pet owner from both the pre-appointment complaint summary and the conversation.

    Actions: From Pre-Appointment Complaint Summary, Extract the pet owners reported symptoms, concerns, and observations.
    Note any relevant history or changes in behavior.

    From Doctor-Patient Conversation:

    Identify additional subjective information shared during the consultation.

    Include any clarifications or new symptoms mentioned.

    Combine Information:

    Integrate data from both sources into a cohesive Subjective section.

    Ensure all information is relevant to the current condition.

    Include: Owners observations, reported symptoms, duration of symptoms, changes in behavior or appetite.

    Exclude: Casual remarks, unrelated anecdotes, non-medical conversations.

    Step 4: Objective Data Identification

    Purpose: Extract objective findings observed by the veterinarian during the examination.

    Actions: Extract from Conversation: Note physical examination findings, vital signs, diagnostic test results, and observable clinical signs

    Use precise medical terminology.

    Include Measurable Data:

    Record specific metrics such as temperature, heart rate, weight, etc., as provided.

    Include: Physical examination findings, vital signs, laboratory results, imaging findings.

    Exclude: Veterinarians small talk, non-clinical observations.

    Step 5: Assessment Extraction

    Purpose: Extract the Assessment findings from the conversation.

    Actions: Analyze the Conversation: Analyze the veterinarians discussions, explanations, and any indications provided.

    Pay attention to the veterinarians observations, hypotheses, or considerations regarding the pets condition.

    Identify Implicit Assessments:

    Extract assessments that are implied through the veterinarians comments or discussions with the pet owner.

    Only include assessments that are directly supported by the veterinarians statements. Do not add interpretations or diagnoses not mentioned in the conversation.

    Ensure that the assessment is directly supported by the information in the conversation.

    Avoid Adding New Information:

    Do not introduce assessments that are not supported by the conversation.

    Do not make medical judgments beyond what is discussed.

    Assessment Section:

    Summarize the assessment, ensuring it reflects the veterinarians conclusions as derived from the conversation.

    Step 6: Plan of Action

    Purpose: Extract the recommended plan for diagnostics, treatment, and follow-up based on the conversation and summarize the recommended plan for diagnostics, treatment, and follow-up as discussed in the conversation.

    Actions:

    Document all Recommendations:

    Include treatments, medications, therapies, interventions, diagnostic tests as discussed in the conversation.

    Provide Owner Instructions:

    Extract all care instructions given to the pet owner.

    Ensure Accuracy:

    Only include actions explicitly mentioned in the conversation.

    Step 7: Conclusion and Summary

    Purpose: Provide a detailed, cohesive, professional conclusion summarizing the SOAP note.

    Actions:

    Summarize Key Points:

    Restate the main findings, assessments, and plans as derived from the conversation.

    Ensure the conclusion reflects the information provided.

    Maintain Professional Tone:

    Use formal language appropriate for veterinary documentation.

    Step 8: Final Differential Diagnosis Derivation

    Purpose: Derive the Differential Diagnosis from the conversation, even if the veterinarian does not explicitly state it.

    Actions: Analyze the Conversation: Identify any conditions, systems, or diagnoses that are discussed or implied.

    Select Appropriate Term:

    Based on the conversation, identify the most likely condition or system affected.

    Use the format System-Condition (e.g., Gastrointestinal-Gastroenteritis).

    Ensure it is directly supported by the conversation.

    Avoid Adding New Information:

    Do not introduce differential diagnoses not supported by the conversation.

    Do not add new information or speculate beyond the details provided.

    Differential Diagnosis Section:

    Provide the differential diagnosis in the format System-Condition (e.g., Gastrointestinal-Gastroenteritis). Use the System from this list differential_diagnosis}

    Step 9: Action Items Extraction:

    Purpose: Derive all the action items of vaccination, deworming, flea and tick treatment, medications, injections, procedures, diagnostic tests from the conversation.

    Identify and Extract Specific Action Items from the Conversation:

    
    Vaccinations: Vaccine Name, scheduled date in DD-MM-YYYY format.

    Deworming: Deworming Name, scheduled date in DD-MM-YYYY format.

    Flea and tick treatment: Product name, scheduled date in DD-MM-YYYY format.

    Prescription:

    List all medications, injections, or treatments prescribed or administered during the visit.

    It should follow a standard format of:

    Name:
    Duration:
    Dosage:
    Frequency: Morning, afternoon, night or custom
    Remarks: Before meal or after meal and any other additional remarks or recommendations

    Diet Recommendations:

    Note any food or diet recommendations provided by the veterinarian.

    Diagnostic Tests:

    Include any diagnostic tests that were performed or recommended, such as blood tests, radiographs, etc. It should follow a standard format of: Name, scheduled date in DD-MM-YYYY format.

    Present These Action Items in a Separate Section:

    Organize the extracted action items under appropriate headings.

    Ensure Accuracy:

    Only include action items explicitly mentioned in the conversation.

    Do not infer or add actions not discussed.

    Step 10: Verification Step:

    Review for Accuracy and Relevance:

    Before finalizing the SOAP note, carefully review all extracted information.

    Ask yourself the following questions:

    Does each piece of information directly relate to the pets medical condition?

    Have I excluded all non-medical and irrelevant content?

    Make Necessary Corrections:

    If any information is irrelevant or inaccurately included, remove or correct it accordingly.

    Ensure that only pertinent medical information is included in each section.

    Step 11: Final Review and Compliance Check

    Purpose: Ensure the SOAP note is accurate, complete, and adheres to all guidelines.

    Actions:

    Verify All Sections:

    Confirm that all information is directly derived from the pre-appointment complaint summary and the conversation.

    Check Terminology and Formatting:

    Use correct medical terminology.

    Follow the specified output format precisely.

    Maintain Professionalism:

    Ensure the document reflects professional veterinary standards.

    Step 12: Data Formatting and Output:

    Purpose: Organize all verified data into the final formatted output. The output has to be strictly in the following format against each of the headings.

    Instructions:

    Compile all the verified information into a neatly formatted output with the following headers:

    Subjective: [Subjective content here]
    Objective: [Objective content here]
    Assessment: [Assessment content here]
    Plan: [Plan content here]
    Conclusion: [Conclusion content here]
    Differentialdiagnosis: [Differential Diagnosis content here]
Vaccination:[Vaccination content here]
Prescription: [medications and treatments content here]
Diet Recommendations:[diet recommendations content here]
Diagnostic Tests: [diagnostic tests performed or recommended content here]

    Step 13: Final Reminders:

    Assessment and Differential Diagnosis:

    Extract them from the conversation, even if not explicitly stated by the veterinarian as Assessment and differential diagnosis.

    Ensure they are directly supported by the conversation content.

    Do not add new information or make medical judgments beyond the conversation.

    Avoid Hallucinations:

    Do not introduce information that is not present in the pre-appointment complaint summary or the audio conversation.

    Professional Tone:

    Use formal language and correct medical terminology. Ensure that the final output is professional and suitable for inclusion in the official clinic medical records.

    Consistency:

    Ensure that all sections align with the extracted information.

    Mandatory Fields:

    Follow the detailed instructions for all fields. If any fields are not filled, reread the Input again to identify and extract the correct details.

    Conclusion and Differential Diagnosis are mandatory fields.

    Differential Diagnosis should be in the format System-Condition. The system can be derived from this list {differential_diagnosis}. For example: Dermatology-Atopic Dermatitis. Strictly stick to this format and don't give additional statements.

    Handling Missing Information:

    If any of the information is missing for a particular field and cannot be found or inferred, leave that section blank to maintain data integrity.

    If medically relevant content is absent in the source: Output a response for the fields stating that: No medically relevant information was provided for each of the content sections
    """)
    prompt_generation_time = time.time() - prompt_time
    print(f"Prompt generation time: {prompt_generation_time} seconds")

    # Generating medical note
    process_chain_time = time.time()
    process_conversation_chain = LLMChain(
        llm=openai, prompt=conversation_prompt
    )

    data = {"full_text": full_text, "differential_diagnosis": differential_diagnosis}

    medical_note_text = process_conversation_chain.run(data)
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
        "Vaccination": extract_section(medical_note_text, "Vaccination")
"Deworming": extract_section(medical_note_text, "Deworming")
"Flea and tick treament": extract_section(medical_note_text, "Flea and tick treatment")
        "Prescription": extract_section(medical_note_text, "Prescription")
        "Dietrecommendations": extract_section(medical_note_text, "Dietrecommendations")
        "Diagnostics": extract_section(medical_note_text, "Diagnostics")
        
        }

    os.remove(temp_audio_path)

    total_time = time.time() - start_time
    print(f"Total time taken: {total_time} seconds")

    return medical_note
