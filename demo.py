import torch
import soundfile as sf
from transformers import RobertaTokenizer
from model import MultimodalIntentModel
import config

# ===== LOAD MODEL =====
num_classes = 14
model = MultimodalIntentModel(num_classes)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

tokenizer = RobertaTokenizer.from_pretrained(config.MODEL_NAME_TEXT)

# ===== INTENT MAPPING =====
id2label = {
    0: "LocateBranch",
    1: "ActivateCard",
    2: "LastTransactionDetails",
    3: "CardDispatchStatus",
    4: "CreditCardOutstandingBalance",
    5: "CardIssue",
    6: "GetIFSCCode",
    7: "GenerateCardPIN",
    8: "ReportFraud",
    9: "LoanInquiry",
    10: "CheckAccountBalance",
    11: "ChangeCardLimit",
    12: "BlockRequest",
    13: "ReportLostCard"
}

# ===== INPUT =====
text = "Please tell me the IFSC Code"
audio_path = "sample1.wav"

# AUDIO
waveform, _ = sf.read(audio_path)
waveform = torch.tensor(waveform, dtype=torch.float32)

if len(waveform.shape) > 1:
    waveform = waveform.mean(dim=1)

waveform = waveform.unsqueeze(0)

# TEXT
enc = tokenizer(
    text,
    padding="max_length",
    truncation=True,
    max_length=config.MAX_LEN,
    return_tensors="pt"
)

input_ids = enc["input_ids"]
attention_mask = enc["attention_mask"]

# ===== PREDICT =====
with torch.no_grad():
    logits = model(input_ids, attention_mask, waveform)
    pred = logits.argmax(dim=1).item()

print("\n===== DEMO =====")
print("Text:", text)
print("Predicted Intent:", id2label[pred])