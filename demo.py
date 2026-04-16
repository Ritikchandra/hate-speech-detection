import torch
import soundfile as sf
import whisper
from transformers import RobertaTokenizer
from model import MultimodalIntentModel
import config

# LOAD MODELS
model = MultimodalIntentModel()
model.eval()

tokenizer = RobertaTokenizer.from_pretrained(config.MODEL_NAME_TEXT)

# LOAD WHISPER (use small model for speed)
whisper_model = whisper.load_model("tiny")

# INPUT AUDIO
audio_path = "audio1.wav"

# ===== STEP 1: AUDIO → TEXT =====
result = whisper_model.transcribe(audio_path)
text = result["text"].strip()

print("\n[Transcription]:", text)

# ===== STEP 2: TEXT → TOKENS =====
enc = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=config.MAX_LEN,
    return_tensors="pt"
)

input_ids = enc["input_ids"]
attention_mask = enc["attention_mask"]

# ===== STEP 3: AUDIO → FEATURES =====
waveform, sr = sf.read(audio_path)
audio = torch.tensor(waveform, dtype=torch.float32)

if len(audio.shape) > 1:
    audio = audio.mean(dim=1)

audio = audio.unsqueeze(0)

# LABEL (dummy for loss)
label = torch.tensor([config.label2id["PlayMusic"]])

# ===== STEP 4: MODEL =====
with torch.no_grad():
    loss, output = model(input_ids, attention_mask, audio, label)

pred = output.argmax(dim=1).item()

# ===== OUTPUT =====
print("\n===== FINAL OUTPUT =====")
print("Audio Input:", audio_path)
print("Transcribed Text:", text)
print("Predicted Intent:", config.id2label[pred])
print("Loss:", loss.item())