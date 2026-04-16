import torch
import soundfile as sf
from transformers import RobertaTokenizer
from model import MultimodalIntentModel
import config

# LOAD MODEL
model = MultimodalIntentModel()
model.eval()

tokenizer = RobertaTokenizer.from_pretrained(config.MODEL_NAME_TEXT)

# INPUT (MANUAL TEXT)
text = "Play some music"
audio_path = "audio1.wav"

# TEXT PROCESSING
enc = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=config.MAX_LEN,
    return_tensors="pt"
)

input_ids = enc["input_ids"]
attention_mask = enc["attention_mask"]

# AUDIO PROCESSING
waveform, sr = sf.read(audio_path)
audio = torch.tensor(waveform, dtype=torch.float32)

if len(audio.shape) > 1:
    audio = audio.mean(dim=1)

audio = audio.unsqueeze(0)

# LABEL (for loss)
label = torch.tensor([config.label2id["PlayMusic"]])

# RUN
with torch.no_grad():
    loss, output = model(input_ids, attention_mask, audio, label)

pred = output.argmax(dim=1).item()

print("\n===== DEMO OUTPUT =====")
print("Text Input:", text)
print("Predicted Class:", pred)
print("Predicted Intent:", config.id2label[pred])
print("Loss:", loss.item())