!pip install -U -q PyDrive


from google.colab import drive
drive.mount('/content/drive')
%run /content/drive/MyDrive/NLP/transformers.ipynb
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

import gspread
from google.colab import auth
from google.auth import default

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
sheet = gc.open_by_url('https://docs.google.com/spreadsheets/d/1KTtv07rQkpTosYZJn9h2iF5PObJlkrjpF3OzT0Oby68/edit?usp=drive_link')



START_TOKEN = ''
PADDING_TOKEN = ''
END_TOKEN = ''

worksheet = sheet.get_worksheet(0)
data = worksheet.get_all_values()[1:]
english_sentences = [row[0] for row in data]
russian_sentences = [row[1] for row in data]


# Limit Number of sentences
total_sentences = 100000
english_sentences = english_sentences[:total_sentences]
russian_sentences = russian_sentences[:total_sentences]
#english_sentences = [sentence.rstrip('\n') if isinstance(sentence, str) else str(sentence) for sentence in english_sentences]
#russian_sentences = [sentence.rstrip('\n') if isinstance(sentence, str) else str(sentence) for sentence in russian_sentences]
english_sentences = [sentence.rstrip('\n') for sentence in english_sentences]
russian_sentences = [sentence.rstrip('\n') for sentence in russian_sentences]

unique_english_tokens = set()
unique_russian_tokens = set()
for sentence in english_sentences:
    for token in list(set(sentence)):
        unique_english_tokens.add(token)
for sentence in russian_sentences:
    for token in list(set(sentence)):
        unique_russian_tokens.add(token)

unique_english_tokens.add(START_TOKEN)
unique_english_tokens.add(END_TOKEN)
unique_english_tokens.add(PADDING_TOKEN)
unique_russian_tokens.add(START_TOKEN)
unique_russian_tokens.add(END_TOKEN)
unique_russian_tokens.add(PADDING_TOKEN)

english_vocabulary = list(unique_english_tokens)
russian_vocabulary = list(unique_russian_tokens)

index_to_russian = {k:v for k,v in enumerate(russian_vocabulary)}
russian_to_index = {v:k for k,v in enumerate(russian_vocabulary)}
index_to_english = {k:v for k,v in enumerate(english_vocabulary)}
english_to_index = {v:k for k,v in enumerate(english_vocabulary)}

print(f"English Vocabulary", english_vocabulary)
print(f"Russian Vocabulary", russian_vocabulary)

max_sequence_length = 200

def is_valid_tokens(sentence, vocab):
    for token in list(set(sentence)):
        if token not in vocab:
            return False
    return True

def is_valid_length(sentence, max_sequence_length):
    return len(sentence) > 0 and len(list(sentence)) < (max_sequence_length - 1)

valid_sentence_indicies = []
for index in range(len(russian_sentences)):
    russian_sentence, english_sentence = russian_sentences[index], english_sentences[index]
    if is_valid_length(russian_sentence, max_sequence_length) \
      and is_valid_length(english_sentence, max_sequence_length) \
      and is_valid_tokens(russian_sentence, russian_vocabulary):
        valid_sentence_indicies.append(index)

print(f"Number of sentences: {len(russian_sentences)}")
print(f"Number of valid sentences: {len(valid_sentence_indicies)}")

russian_sentences = [russian_sentences[i] for i in valid_sentence_indicies]
english_sentences = [english_sentences[i] for i in valid_sentence_indicies]

d_model = 512
batch_size = 30
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 1
max_sequence_length = 200
ru_vocab_size = len(russian_vocabulary)

transformer = Transformer(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, ru_vocab_size, english_to_index, russian_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)

class TextDataset(Dataset):
  def __init__(self, english_sentences, russian_sentences):
    self.english_sentences = english_sentences
    self.russian_sentences = russian_sentences

  def __len__(self):
    return len(self.english_sentences)

  def __getitem__(self, idx):
    return self.english_sentences[idx], self.russian_sentences[idx]

dataset = TextDataset(english_sentences, russian_sentences)

batch_size = 3
train_loader = DataLoader(dataset, batch_size)
iterator = iter(train_loader)
