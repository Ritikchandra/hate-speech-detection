from model import MultimodalIntentModel
from dataset import MultimodalDataset
from data_loader import load_skit
from train import train

train_audio, train_texts, train_labels, val_audio, val_texts, val_labels, num_classes = load_skit()

train_dataset = MultimodalDataset(train_audio, train_texts, train_labels)
val_dataset = MultimodalDataset(val_audio, val_texts, val_labels)

model = MultimodalIntentModel(num_classes)

train(model, train_dataset, val_dataset)