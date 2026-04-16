def accuracy(preds, labels):
    correct = (preds.argmax(dim=1) == labels).sum().item()
    return correct / len(labels)