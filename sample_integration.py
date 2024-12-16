import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from lime.lime_text import LimeTextExplainer
from custom_lime import CustomLimeTextExplainer
from data import IMDBData
import argparse
from typing import List, Tuple

class IMDBDataset(Dataset):
    def __init__(self, examples: List[Tuple[List[str], int]]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text, label = self.examples[idx]
        return ' '.join(text), label

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.rnn(packed_embedded)
        hidden = hidden.squeeze(0)
        return self.fc(hidden)

class NeuralClassifier:
    def __init__(self, train_data: List[Tuple[List[str], int]], vocab_size: int = 10000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab_size = vocab_size
        self.embedding_dim = 100
        self.hidden_dim = 256
        self.output_dim = 1

        # Build vocabulary from training data
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self._build_vocab(train_data)

        # Initialize model
        self.model = SimpleRNN(
            vocab_size=len(self.word2idx),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim
        ).to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters())

    def _build_vocab(self, train_data: List[Tuple[List[str], int]]):
        word_freq = {}
        for text, _ in train_data:
            for word in text:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and take top vocab_size words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[:self.vocab_size - 2]:  # -2 for <pad> and <unk>
            self.word2idx[word] = len(self.word2idx)

    def _tokenize(self, text: List[str]) -> List[int]:
        return [self.word2idx.get(word, self.word2idx['<unk>']) for word in text]

    def train_model(self, train_data: List[Tuple[List[str], int]], num_epochs: int = 5, batch_size: int = 32):
        dataset = IMDBDataset(train_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0

            for batch_texts, batch_labels in dataloader:
                self.optimizer.zero_grad()

                # Prepare batch
                texts = [text.split() for text in batch_texts]
                lengths = torch.tensor([len(t) for t in texts])
                padded_texts = nn.utils.rnn.pad_sequence(
                    [torch.tensor(self._tokenize(t)) for t in texts],
                    batch_first=True,
                    padding_value=self.word2idx['<pad>']
                ).to(self.device)
                batch_labels = batch_labels.float().to(self.device)

                # Forward pass
                predictions = self.model(padded_texts, lengths).squeeze(1)
                loss = self.criterion(predictions, batch_labels)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader)}')

    def predict(self, text: List[str]) -> float:
        self.model.eval()
        with torch.no_grad():
            tokens = torch.tensor(self._tokenize(text)).unsqueeze(0).to(self.device)
            lengths = torch.tensor([len(text)])
            prediction = torch.sigmoid(self.model(tokens, lengths)).item()
        return prediction

    def save(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'word2idx': self.word2idx,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.word2idx = checkpoint['word2idx']
        self.vocab_size = checkpoint['vocab_size']
        self.embedding_dim = checkpoint['embedding_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        self.output_dim = checkpoint['output_dim']

        self.model = SimpleRNN(
            vocab_size=len(self.word2idx),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

def create_lime_explainer(classifier: NeuralClassifier):
    explainer = CustomLimeTextExplainer()

    def predict_fn(texts):
        predictions = []
        for text in texts:
            pred = classifier.predict(text.split())
            predictions.append([1 - pred, pred])
        return np.array(predictions)

    return explainer, predict_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to the data folder")
    parser.add_argument("--task", type=str, help="task (imdb or author-id)")
    parser.add_argument("--model", type=str, help="model type (baseline, neural, or tfidf)")
    parser.add_argument("--save", type=str, help="path to save model")
    parser.add_argument("--load", type=str, help="path to load model")
    parser.add_argument("--measure", type=str, help="evaluation measure (acc, precision, recall, f1)")
    parser.add_argument("--explain", type=str, help="explain prediction for this text")
    args = parser.parse_args()

    if args.task == "imdb":
        dataset = IMDBData(args.data)
        train_data = list(dataset.get_train_examples())
        dev_data = list(dataset.get_dev_examples())
        test_data = list(dataset.get_test_examples())

        if args.model == "neural":
            classifier = NeuralClassifier(train_data)

            if args.load:
                classifier.load(args.load)
            else:
                print("Training new model...")
                classifier.train_model(train_data)
                if args.save:
                    print(f"Saving model to {args.save}")
                    classifier.save(args.save)

            if args.measure == "acc":
                correct = 0
                total = 0
                for text, label in dev_data:
                    pred = classifier.predict(text)
                    if (pred > 0.5) == label:
                        correct += 1
                    total += 1
                print(f"Accuracy: {100 * correct / total:.2f}%")

            if args.explain:
                explainer, predict_fn = create_lime_explainer(classifier)
                explanation = explainer.explain_instance(
                    args.explain,
                    predict_fn,
                    6,
                    0
                )
                print("Explanation for prediction:")
                print(explanation)

if __name__ == "__main__":
    main()
