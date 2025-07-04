# app.py
import torch
import torch.nn as nn
import gradio as gr
import pickle
from nltk.tokenize import word_tokenize

# Tokenizer loading
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

word2idx = tokenizer["word2idx"]
idx2word = tokenizer["idx2word"]
vocab_size = len(word2idx)

# Model class matching your notebook
class LSTMModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 100)
        self.lstm = nn.LSTM(100, 150, batch_first=True)
        self.fc = nn.Linear(150, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        out = self.fc(hidden.squeeze(0))
        return out

# Load trained model
model = LSTMModel(vocab_size)
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

def text_to_indices(sentence, vocab):
    return [vocab.get(token, vocab['<unk>']) for token in sentence]

def predict_next_word(text):
    tokens = word_tokenize(text.lower())
    indices = text_to_indices(tokens, word2idx)
    if not indices:
        return "❌ Not enough known words to predict."
    max_len = 61
    padded = [0]*(max_len - len(indices)) + indices
    input_tensor = torch.tensor(padded, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = torch.argmax(output, dim=1).item()
        return text + " " + idx2word.get(predicted_idx, "<unk>")

# Gradio app
interface = gr.Interface(
    fn=predict_next_word,
    inputs=gr.Textbox(lines=2, placeholder="Enter a sentence..."),
    outputs="text",
    title="🧠 Next Word Predictor",
    description="LSTM-based Next Word Prediction Model using PyTorch"
)

interface.launch()