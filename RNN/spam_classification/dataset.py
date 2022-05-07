import os

import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class Vocabulary:
    def __init__(self):        
        self.t2i = {}
        self.i2t = {idx:token for token, idx in self.t2i.items()}
            
    def _add_token(self, token):
        if token in self.t2i:
            idx = self.t2i[token]
        else:
            idx = len(self.t2i)
            self.t2i[token] = idx
            self.i2t[idx] = token
        return idx
    
    def add_tokens(self, tokens):
        return [self._add_token(token) for token in tokens]


class SpamDataset(Dataset):
    def __init__(self, root='data'):
        self.root = root
        self.tokenizer = spacy.load('en_core_web_sm').tokenizer
        self.vocabulary=Vocabulary()
        
        self.sentences, self.labels = self._get_data()
    
    def _get_data(self):
        dataset = os.path.join(self.root, 'SMSSpamCollection.txt')
        
        sentences = []
        labels = []
        with open(dataset, 'r') as f:
            for line in f.readlines():
                sentence, label = (line.split('\t')[1].strip()), line.split('\t')[0]
                sentence = [token.text for token in self.tokenizer(sentence)]
                sentences.append(sentence)
                labels.append([0.0] if label == 'ham' else [1.0])
        self._build_vocab(sentences, self.vocabulary)
        
        # sentences = [[self.vocabulary.t2i[token] for token in sentence] for sentence in sentences]
        sentences_t = []
        for sentence in sentences:
            temp = []
            for token in sentence:
                temp.append(self.vocabulary.t2i[token])
            sentences_t.append(torch.tensor(temp))
            temp = []
            
        sentences = pad_sequence(sentences_t, batch_first=True)
        
        
        return sentences, torch.tensor(labels)
                
    def _build_vocab(self, sentences, vocabulary):
        for sentence in sentences:
            vocabulary.add_tokens(sentence)
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]
