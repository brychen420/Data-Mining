import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from collections import Counter

# Label emotions from 0 to 6
def label_map(label):
    if label == "neutral":
        return 0
    elif label == "anger":
        return 1
    elif label == "joy":
        return 2
    elif label == "surprise":
        return 3
    elif label == "sadness":
        return 4
    elif label == "disgust":
        return 5
    elif label == "fear":
        return 6

# Encode the text so that it can be fed into the RNN model
def encode(text, word2index, label, N):
    # text: sentences
    # word2index: dict of words and coresponding indices
    # label: label of emotion
    # N: all data should be padded to length N
    tokenized = word_tokenize(text)
    encoded = [0]*N
    enc1 = [word2index.get(word) for word in tokenized]
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    
    return (encoded, label)


def encode_test(text, word2index, N):
    
    tokenized = word_tokenize(text)
    for i, word in enumerate(tokenized):
        if word2index.get(word) == None:
            tokenized[i] = 'unk'

    encoded = [0]*N
    enc1 = [word2index.get(word) for word in tokenized]

    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    
    return encoded


def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0

    # Use model.train for training
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        text = batch[0].to(device)
        target = batch[1]
        target = target.type(torch.LongTensor)
        target = target.to(device)
        preds = model(text)
        loss = criterion(preds, target)
        _, pred = torch.max(preds, 1)
        acc = accuracy_score(pred.tolist(), target.tolist())

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0

    # Use model.eval in testing
    model.eval()
    for batch in iterator:
        text = batch[0].to(device)
        target = batch[1]
        target = target.type(torch.LongTensor)
        target = target.to(device)
        preds = model(text)
        loss = criterion(preds, target)
        _, pred = torch.max(preds, 1)
        acc = accuracy_score(pred.tolist(), target.tolist())

        epoch_loss += loss.item()
        epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


train_df = pd.read_csv('train_HW2dataset.csv')
dev_df = pd.read_csv('dev_HW2dataset.csv')

train_df = train_df[['Emotion', 'Utterance']]
dev_df = dev_df[['Emotion', 'Utterance']]

train_set = list(train_df.to_records(index=False))
dev_set = list(dev_df.to_records(index=False))

# Generate a dictionary of words with index
counts = Counter()
for ds in [train_set, dev_set]:
    for label, text in ds:
        counts.update(word_tokenize(text))

# Special word "unknown" as index 0
word2index = {'unk': 0}
for i, word in enumerate(counts.keys()):
    word2index[word] = i+1

# Encode train data and development data 
train_encoded = [(encode(Utterance, word2index, label_map(label), 12))
                 for label, Utterance in train_set]

dev_encoded = [(encode(Utterance, word2index, label_map(label), 12))
               for label, Utterance in dev_set]

train_x = np.array([tweet for tweet, _ in train_encoded])
train_y = np.array([label for _, label in train_encoded])
dev_x = np.array([tweet for tweet, _ in dev_encoded])
dev_y = np.array([label for _, label in dev_encoded])

batch_size = 32

train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
dev_ds = TensorDataset(torch.from_numpy(dev_x), torch.from_numpy(dev_y))

train_dl = DataLoader(train_ds, shuffle=True,
                      batch_size=batch_size, drop_last=True)

dev_dl = DataLoader(dev_ds, shuffle=True,
                    batch_size=batch_size, drop_last=True)


# Set hyper parameters
src_vocab_size = len(word2index)
dimension_model = 32
num_layers = 5
hidden_size = 30
linear_hidden_size = 10
classes = 7
dropout = 0.2
lr = 1e-3

# Define properties and functions for our LSTM model
class LSTM(torch.nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()
        self.embed = torch.nn.Embedding(src_vocab_size, dimension_model)
        self.lstm = torch.nn.LSTM(input_size=dimension_model, hidden_size=hidden_size,
                                  num_layers=num_layers, dropout=dropout)
        self.linear = torch.nn.Linear(hidden_size, linear_hidden_size)
        self.linear1 = torch.nn.Linear(linear_hidden_size, classes)

    def forward(self, data):
        x = self.embed(data)
        x, (h_n, c_n) = self.lstm(x.transpose(0, 1))

        x = self.linear(x[-1])
        x = self.linear1(x)

        return x

# Train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTM().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_acc = 0
for epoch in range(10):
    train_loss, train_acc = train(model, train_dl,
                                  optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, dev_dl,
                                     criterion)

    print(f'Epoch: {epoch+1:02}, ')
    print(f'Train Loss: {train_loss:.3f},Train Acc: {train_acc * 100:.2f}%,')
    print(f'Val. Loss: {valid_loss:.3f}, Val. Acc: {valid_acc * 100:.2f}%\n')
    
    # Save the model with highest accuracy
    if best_acc <= valid_acc:
        best_acc = valid_acc
        PATH = f"epoch{epoch+1}_val.accuracy{valid_acc*100:.1f}%.pt"
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': valid_loss,
        }, PATH)

# Test the performance of the best model with test dataset
test_df = pd.read_csv('test_HW2dataset.csv')
test_df = test_df[['Utterance']]
test_set = test_df.values.tolist()
test_encoded = []

for sentence in test_set:
    test_encoded += [encode_test(Utterance, word2index, 10)
                     for Utterance in sentence]

test_x = np.array(test_encoded)
test_ds = TensorDataset(torch.from_numpy(test_x))
test_dl = DataLoader(test_ds, shuffle=False)

# Load best model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTM().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']

# Use model.eval on testing
model.eval()

# Output our prediction mede by our best model into csv file
predict = []
for deta in test_dl:
    text = deta[0].to(device)
    preds = model(text)
    _, pred = torch.max(preds, 1)
    predict.append(pred.item())

indices = range(len(predict))
data = {"index": indices ,"emotion": predict}
out_df = pd.DataFrame(data, columns=["index", "emotion"])
out_df.to_csv('result.csv', index=False, header=True)
