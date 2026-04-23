import torch
from torch.utils.data import Dataset
import soundfile as sf
import io
from transformers import RobertaTokenizer
import config

tokenizer = RobertaTokenizer.from_pretrained(config.MODEL_NAME_TEXT)

class MultimodalDataset(Dataset):
    def __init__(self, audio_data, texts, labels):
        self.audio_data = audio_data
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, idx):
        audio_sample = self.audio_data[idx]
        text = self.texts[idx]

        # AUDIO (from bytes)
        waveform, _ = sf.read(io.BytesIO(audio_sample["bytes"]))
        waveform = torch.tensor(waveform, dtype=torch.float32)

        if len(waveform.shape) > 1:
            waveform = waveform.mean(dim=1)

        # TEXT
        enc = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=config.MAX_LEN,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        label = torch.tensor(self.labels[idx])

        return {
            "audio": waveform,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label
        }