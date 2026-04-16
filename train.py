import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
import torch.nn.functional as F


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


def train(model, dataset):

    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)

            loss, _ = model(input_ids, attention_mask, audio, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")