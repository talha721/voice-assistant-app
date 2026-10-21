import os
from datetime import datetime
import torch
import scipy
import numpy as np
import sounddevice as sd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    VitsModel,
    pipeline
)

# =====================================================================
# 1. SETUP STORAGE & CONFIGURATION
# =====================================================================
AUDIO_DIR = "assistant"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Audio recording settings
SAMPLE_RATE = 16000  # Standard rate for Whisper
RECORD_SECONDS = 5   # Fixed recording length in seconds


# =====================================================================
# 2. LOAD ALL THREE AI MODELS (STT, LLM, TTS)
# =====================================================================
print("⏳ Loading Speech-to-Text Engine (openai/whisper-base)...")
stt_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
    torch_dtype=torch.float32
)

print("⏳ Loading AI Brain LLM (Qwen/Qwen2.5-0.5B-Instruct)...")
llm_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_id,
    torch_dtype=torch.float32
)
llm_generator = pipeline(
    "text-generation",
    model=llm_model,
    tokenizer=llm_tokenizer
)

print("⏳ Loading Text-to-Speech Engine (facebook/mms-tts-eng)...")
tts_model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")


# =====================================================================
# 3. HELPER FUNCTIONS
# =====================================================================
def record_audio_from_mic(duration: int = RECORD_SECONDS, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Captures raw audio array directly from your system microphone."""
    print(f"\n🎤 Listening... Speak into your mic now! ({duration} seconds)")
    # Record audio into numpy array
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32"
    )
    sd.wait()  # Wait until recording finishes
    print("⏳ Processing your audio...")
    return audio_data.squeeze()


def transcribe_audio(audio_array: np.ndarray) -> str:
    """Passes the raw audio array into Whisper STT to convert it to text."""
    result = stt_pipeline({"raw": audio_array, "sampling_rate": SAMPLE_RATE})
    return result["text"].strip()


def generate_ai_reply(user_input: str) -> str:
    """Generates an answer using the Qwen LLM."""
    messages = [
        {
            "role": "system",
            "content": "You are a friendly voice assistant. Give a concise, direct response in 1-2 sentences. Do not echo the question."
        },
        {"role": "user", "content": user_input}
    ]

    prompt = llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

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
    """Converts the AI response back into speech and saves it in 'assistant' folder."""
    inputs = tts_tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = tts_model(**inputs).waveform

    waveform = output.squeeze().cpu().numpy()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(AUDIO_DIR, f"reply_{timestamp}.wav")

    scipy.io.wavfile.write(
        filepath, rate=tts_model.config.sampling_rate, data=waveform
    )
    return filepath


# =====================================================================
# 4. FULL VOICE CONVERSATION LOOP
# =====================================================================
def main():
    print("\n" + "=" * 60)
    print("🎙️ Full Voice Assistant Active! Press ENTER to record, or Ctrl+C to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            input("👉 Press [ENTER] to start speaking...")

            # Step 1: STT - Listen to Microphone & Transcribe
            audio_array = record_audio_from_mic()
            transcribed_text = transcribe_audio(audio_array)

            if not transcribed_text:
                print("❌ Could not understand any audio. Try speaking again.")
                continue

            print(f"👤 You (Transcribed): \"{transcribed_text}\"")

            # Step 2: LLM - Process with Qwen
            print("🧠 Thinking...")
            ai_reply = generate_ai_reply(transcribed_text)
            print(f"🤖 Assistant: \"{ai_reply}\"")

            # Step 3: TTS - Synthesize AI reply back to audio file
            saved_audio_path = convert_text_to_audio(ai_reply)
            print(f"🔊 Response saved to: '{saved_audio_path}'\n")

        except KeyboardInterrupt:
            print("\n👋 Exiting Voice Assistant.")
            break


if __name__ == "__main__":
    main()