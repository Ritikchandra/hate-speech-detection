import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import config

def collate_fn(batch):
    input_ids = torch.stack([x["input_ids"] for x in batch])
    attention_mask = torch.stack([x["attention_mask"] for x in batch])
    labels = torch.stack([x["label"] for x in batch])

    audios = [x["audio"] for x in batch]
    max_len = max([a.shape[0] for a in audios])

    padded_audio = []
    for a in audios:
        pad = max_len - a.shape[0]
        padded_audio.append(F.pad(a, (0, pad)))

    audio = torch.stack(padded_audio)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "audio": audio,
        "label": labels
    }

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask, audio)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

def train(model, train_dataset, val_dataset):

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)

            loss, _ = model(input_ids, attention_mask, audio, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"\nEpoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

        val_acc = evaluate(model, val_loader, device)
        print(f"Validation Accuracy: {val_acc:.4f}")
        torch.save(model.state_dict(), "model1.pth")