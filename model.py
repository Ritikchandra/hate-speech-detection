import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, Wav2Vec2Model
import config


class MultimodalIntentModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.text_model = RobertaModel.from_pretrained(config.MODEL_NAME_TEXT)
        self.audio_model = Wav2Vec2Model.from_pretrained(config.MODEL_NAME_AUDIO)

        # 🔥 PARTIAL UNFREEZING (KEY CHANGE)
        for name, param in self.text_model.named_parameters():
            if "encoder.layer.10" in name or "encoder.layer.11" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for name, param in self.audio_model.named_parameters():
            if "encoder.layers.10" in name or "encoder.layers.11" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Classifiers
        self.text_fc = nn.Linear(768, num_classes)
        self.audio_fc = nn.Linear(768, num_classes)

        self.fusion_fc1 = nn.Linear(768 * 2, 512)
        self.fusion_fc2 = nn.Linear(512, num_classes)

    def forward(self, input_ids, attention_mask, audio, labels=None):

        # TEXT
        text_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_out.last_hidden_state[:, 0, :]
        text_logits = self.text_fc(text_feat)

        # AUDIO
        audio_out = self.audio_model(audio)
        audio_feat = audio_out.last_hidden_state.mean(dim=1)
        audio_logits = self.audio_fc(audio_feat)

        # FUSION
        fused = torch.cat([text_feat, audio_feat], dim=1)
        fused = F.relu(self.fusion_fc1(fused))
        fusion_logits = self.fusion_fc2(fused)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()

            loss1 = loss_fn(text_logits, labels)
            loss2 = loss_fn(audio_logits, labels)
            loss3 = loss_fn(fusion_logits, labels)

            loss = (
                config.ALPHA1 * loss1 +
                config.ALPHA2 * loss2 +
                (1 - config.ALPHA1 - config.ALPHA2) * loss3
            )

            return loss, fusion_logits

        return fusion_logits