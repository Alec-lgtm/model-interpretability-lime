import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, LabelField, TabularDataset, BucketIterator
from torchtext.vocab import GloVe

# Define fields for text and labels
TEXT = Field(tokenize='spacy', lower=True, include_lengths=True)
LABEL = LabelField(dtype=torch.float)

# Load dataset
datafields = [('text', TEXT), ('label', LABEL)]
train_data, test_data = TabularDataset.splits(
    path='.', train='train.csv', test='test.csv', format='csv', fields=datafields
)

# Build vocabulary with pretrained embeddings
TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=100))
LABEL.build_vocab(train_data)

# Create iterators
train_iter, test_iter = BucketIterator.splits(
    (train_data, test_data), batch_size=32, sort_within_batch=True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# Define the model
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pretrained_embeddings):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(pretrained_embeddings)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True)
        packed_output, (hidden, _) = self.rnn(packed_embedded)
        hidden = hidden.squeeze(0)
        return self.fc(hidden)

# Instantiate the model
vocab_size = len(TEXT.vocab)
embedding_dim = 100
hidden_dim = 256
output_dim = 1
pretrained_embeddings = TEXT.vocab.vectors

model = SimpleRNN(vocab_size, embedding_dim, hidden_dim, output_dim, pretrained_embeddings)

# Define optimizer and loss
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

# Training loop
for epoch in range(5):
    model.train()
    epoch_loss = 0
    for batch in train_iter:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(train_iter)}')

torch.save(model.state_dict(), 'text_classifier.pt')

