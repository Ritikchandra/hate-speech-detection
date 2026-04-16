# Multimodal Intent Recognition

## Description
Implements audio + text multimodal intent recognition using:
- RoBERTa (text)
- wav2vec2 (audio)
- Fusion + weighted loss

## Run
pip install -r requirements.txt
python main.py

## Architecture
- Text branch → loss1
- Audio branch → loss2
- Fusion → loss3
- Final loss = weighted sum

## Based on Paper
"New Method for Intent Recognition Based on Audio and Text Multimodality"