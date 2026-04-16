from model import MultimodalIntentModel
from dataset import MultimodalDataset
from train import train

texts = [
    "Play music",
    "Navigate to home",
    "Turn on AC"
]

audio_paths = [
    "audio1.wav",
    "audio2.wav",
    "audio3.wav"
]

labels = [1, 2, 3]

dataset = MultimodalDataset(texts, audio_paths, labels)
model = MultimodalIntentModel()

train(model, dataset)