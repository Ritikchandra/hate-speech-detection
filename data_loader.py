from datasets import load_dataset, Audio

def load_skit():
    ds = load_dataset("skit-ai/skit-s2i")

    # disable HF decoding (avoid torchcodec issues)
    ds = ds.cast_column("audio", Audio(decode=False))

    train_data = ds["train"]
    val_data = ds["test"]

    train_labels_raw = train_data["intent_class"]
    val_labels_raw = val_data["intent_class"]

    # label normalization
    unique_labels = sorted(set(train_labels_raw))
    label_map = {old: new for new, old in enumerate(unique_labels)}

    train_labels = [label_map[x] for x in train_labels_raw]
    val_labels = [label_map[x] for x in val_labels_raw]

    train_audio = train_data["audio"]
    val_audio = val_data["audio"]

    train_texts = train_data["template"]
    val_texts = val_data["template"]

    print("NUM_CLASSES:", len(unique_labels))

    return train_audio, train_texts, train_labels, val_audio, val_texts, val_labels, len(unique_labels)