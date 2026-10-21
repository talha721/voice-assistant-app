from datetime import datetime
import os
import scipy
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    VitsModel,
    pipeline,
)

# =====================================================================
# 1. SETUP STORAGE DIRECTORY
# =====================================================================
AUDIO_DIR = "assistant_voice"
os.makedirs(AUDIO_DIR, exist_ok=True)


# =====================================================================
# 2. LOAD AI BRAIN (LLM)
# =====================================================================
print("⏳ Loading LLM (Qwen/Qwen2.5-0.5B-Instruct)...")
llm_model_id = "Qwen/Qwen2.5-0.5B-Instruct"

llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_id,
    torch_dtype=torch.float32,
)

generator = pipeline(
    "text-generation",
    model=llm_model,
    tokenizer=llm_tokenizer,
)


def generate_ai_reply(user_input: str) -> str:
    """Generates a concise conversational reply."""
    messages = [
        {
            "role": "system",
            "content": "You are a friendly voice assistant. Give a concise, direct response in 1-2 sentences. Do not echo the question.",
        },
        {"role": "user", "content": user_input},
    ]

    prompt = llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    outputs = generator(
        prompt,
        max_new_tokens=60,
        temperature=0.6,
        top_p=0.9,
        do_sample=True,
        return_full_text=False,
    )

    raw_text = outputs[0]["generated_text"].strip()

    # Clean residual tokens
    if raw_text.startswith("assistant"):
        raw_text = raw_text.replace("assistant", "", 1).strip()

    return raw_text


# =====================================================================
# 3. LOAD TEXT-TO-SPEECH (facebook/mms-tts-eng)
# =====================================================================
print("⏳ Loading MMS-TTS Engine...")
tts_model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")


def convert_text_to_audio(text: str) -> str:
    """Converts text into audio and saves it in the 'assistant' folder."""
    inputs = tts_tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = tts_model(**inputs).waveform

    waveform = output.squeeze().cpu().numpy()

    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(AUDIO_DIR, f"reply_{timestamp}.wav")

    scipy.io.wavfile.write(
        filepath, rate=tts_model.config.sampling_rate, data=waveform
    )
    return filepath


# =====================================================================
# 4. CONVERSATION LOOP
# =====================================================================
def main():
    print("\n" + "=" * 50)
    print("🤖 Voice Assistant Ready! Type 'exit' or 'quit' to stop.")
    print("=" * 50 + "\n")

    while True:
        try:
            user_text = input("👤 You: ").strip()

            if not user_text:
                continue

            if user_text.lower() in ["exit", "quit", "stop", "bye"]:
                print("\n👋 Goodbye! Ending conversation session.")
                break

            # Step 1: LLM Processing
            print("🧠 Thinking...")
            ai_reply = generate_ai_reply(user_text)
            print(f"🤖 Assistant: {ai_reply}")

            # Step 2: TTS Processing & File Saving
            audio_path = convert_text_to_audio(ai_reply)
            print(f"🔊 Audio saved to: '{audio_path}'\n")

        except KeyboardInterrupt:
            print("\n👋 Conversation interrupted. Exiting...")
            break


if __name__ == "__main__":
    main()