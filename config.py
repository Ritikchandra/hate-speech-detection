MODEL_NAME_TEXT = "roberta-base"
MODEL_NAME_AUDIO = "facebook/wav2vec2-base"

NUM_CLASSES = 7
MAX_LEN = 77

LR = 2e-5
EPOCHS = 3
BATCH_SIZE = 2

ALPHA1 = 0.3
ALPHA2 = 0.3

# SNIPS labels
id2label = {
    0: "AddToPlaylist",
    1: "BookRestaurant",
    2: "GetWeather",
    3: "PlayMusic",
    4: "RateBook",
    5: "SearchCreativeWork",
    6: "SearchScreeningEvent"
}

label2id = {v: k for k, v in id2label.items()}