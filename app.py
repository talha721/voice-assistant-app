import os
import scipy
import torch
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    VitsModel,
    pipeline
)

app = Flask(__name__)

# Folder to store generated response audio files
AUDIO_DIR = "assistant_voice"
os.makedirs(AUDIO_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 1. LOAD MODELS (STT, LLM, TTS)
# ---------------------------------------------------------------------
print("⏳ Loading Speech-to-Text Engine (Whisper)...")
stt_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
    torch_dtype=torch.float32
)

print("⏳ Loading LLM Engine (Qwen2.5-0.5B)...")
llm_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_id, torch_dtype=torch.float32)
llm_generator = pipeline("text-generation", model=llm_model, tokenizer=llm_tokenizer)

print("⏳ Loading Text-to-Speech Engine (MMS-TTS)...")
tts_model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")


# ---------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------------------
def generate_ai_reply(user_input: str) -> str:
    messages = [
        {"role": "system", "content": "You are a friendly voice assistant. Give a concise, direct response in 1-2 sentences."},
        {"role": "user", "content": user_input}
    ]
    prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = llm_generator(
        prompt,
        max_new_tokens=60,
        temperature=0.6,
        top_p=0.9,
        do_sample=True,
        return_full_text=False
    )
    raw_text = outputs[0]["generated_text"].strip()
    if raw_text.startswith("assistant"):
        raw_text = raw_text.replace("assistant", "", 1).strip()
    return raw_text


def convert_text_to_audio(text: str) -> str:
    inputs = tts_tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = tts_model(**inputs).waveform
    
    waveform = output.squeeze().cpu().numpy()
    filename = f"reply_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    filepath = os.path.join(AUDIO_DIR, filename)
    scipy.io.wavfile.write(filepath, rate=tts_model.config.sampling_rate, data=waveform)
    return filename


# ---------------------------------------------------------------------
# 3. FLASK ROUTES
# ---------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/assistant_voice/<filename>")
def get_audio(filename):
    """Serves generated WAV files to the frontend audio player."""
    return send_from_directory(AUDIO_DIR, filename)


@app.route("/api/text-chat", methods=["POST"])
def text_chat():
    """Text Input -> LLM -> TTS -> Audio URL"""
    data = request.get_json()
    user_text = data.get("prompt", "").strip()

    if not user_text:
        return jsonify({"error": "Empty input"}), 400

    ai_reply = generate_ai_reply(user_text)
    audio_filename = convert_text_to_audio(ai_reply)

    return jsonify({
        "response": ai_reply,
        "audio_url": f"/assistant_voice/{audio_filename}"
    })


@app.route("/api/speech-chat", methods=["POST"])
def speech_chat():
    """Web Audio File Upload -> Whisper STT -> LLM -> TTS -> Audio URL"""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    
    # Save temp input audio file
    temp_path = os.path.join(AUDIO_DIR, "temp_input.wav")
    audio_file.save(temp_path)

    # Transcribe via Whisper
    transcription = stt_pipeline(temp_path)["text"].strip()
    
    if not transcription:
        return jsonify({"error": "Could not recognize speech"}), 400

    # Generate AI answer & synthesized audio
    ai_reply = generate_ai_reply(transcription)
    audio_filename = convert_text_to_audio(ai_reply)

    return jsonify({
        "user_text": transcription,
        "response": ai_reply,
        "audio_url": f"/assistant_voice/{audio_filename}"
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)