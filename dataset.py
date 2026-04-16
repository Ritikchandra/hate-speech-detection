import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
import torchaudio
import config

tokenizer = RobertaTokenizer.from_pretrained(config.MODEL_NAME_TEXT)


class MultimodalDataset(Dataset):
    def __init__(self, texts, audio_paths, labels):
        self.texts = texts
        self.audio_paths = audio_paths
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]

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

        # AUDIO
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "audio": waveform,
            "label": torch.tensor(label, dtype=torch.long)
        }