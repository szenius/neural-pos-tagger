# python3.5 run_tagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle as pk
from datetime import datetime


class POSTagger(nn.Module):
    '''
    POS Tagger using CNN and LSTM.
    '''

    def __init__(self, char_dict, word_dict, tag_dict):
        super(POSTagger, self).__init__()

        # Save dictionaries
        self.char_dict = char_dict
        self.word_dict = word_dict
        self.tag_dict = tag_dict

        # Hyperparameters
        self.char_embedding_dim = 10
        self.word_embedding_dim = 250
        self.conv_in_channels = 1
        self.conv_filters = 32
        self.conv_kernel = 3
        self.maxpool_kernel = 1
        self.lstm_hidden_dim = 250
        self.lstm_num_layers = 2
        self.dropout = 0.5

        # Embeddings
        self.char_embeddings = nn.Embedding(len(self.char_dict), self.char_embedding_dim).to(device)
        self.word_embeddings = nn.Embedding(len(self.word_dict), self.word_embedding_dim).to(device)
        self.lstm_hidden_embeddings = self.init_hidden_embeddings(batch_size)

        # Layers
        self.conv = nn.Conv1d(self.char_embedding_dim, self.conv_filters, self.conv_kernel, bias=True, padding=(self.conv_kernel // 2)).to(device)
        self.lstm = nn.LSTM(self.word_embedding_dim + self.conv_filters, self.lstm_hidden_dim, dropout=self.dropout, num_layers=self.lstm_num_layers).to(device)
        self.dense = nn.Linear(self.lstm_hidden_dim, len(self.tag_dict)).to(device)
    
    def init_hidden_embeddings(self, batch_size):
        return (torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_dim).to(device),
                torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_dim).to(device))
    
    def forward(self, char_indices_batch, word_indices_batch):
        '''
        Input expects a batch of sentences, represented character indices and word indices.
            char_indices_batch: (batch_size, num_words_in_sentence, num_characters_in_word)
            word_indices_batch: (batch_size, num_words_in_sentence)
        
        Runs 
            Embedding through the character indices to generate character embeddings
            CNN over character embeddings grouped as words to generate character level word embeddings
            Embedding through the word indices to generate word level word embeddings
            Concatenates char and word level word embeddings as final word embeddings
            Pad each sentence to max sentence length in batch
            Bi-directional LSTM over word embeddings to generate a probability set for the POS tag set

        Returns tag_scores (batch_size, num_words, tagset_probabilities) and the maximum sentence length for this batch
        '''
        ############ Character-level CNN ############
        batch_sentence_embedding = []
        max_sentence_length = 0
        for sent_index, sentence in enumerate(char_indices_batch):
            sentence_embedding = []
            for word_index, word in enumerate(sentence):
                # Generate character embeddings using Embedding layer
                word_embedding_char_level = torch.tensor(word, dtype=torch.long).to(device)
                word_embedding_char_level = self.char_embeddings(word_embedding_char_level)
                # Run through CNN to get word embedding
                word_embedding_char_level = word_embedding_char_level.permute(1, 0)
                word_embedding_char_level = torch.stack([word_embedding_char_level]).to(device)
                word_embedding_char_level = self.conv(word_embedding_char_level)
                word_embedding_char_level = torch.max(word_embedding_char_level, 2)[0][0]
                # Get word embedding from tokens
                word_embedding_word_level = torch.tensor(word_indices_batch[sent_index][word_index], dtype=torch.long).to(device)
                word_embedding_word_level = self.word_embeddings(word_embedding_word_level)
                # Concat word embeddings and add to sentence embedding list
                word_embedding = torch.cat((word_embedding_char_level, word_embedding_word_level), 0).to(device)
                sentence_embedding.append(word_embedding)
            # Update max sentence length, add sentence embedding to batch list
            max_sentence_length = max(max_sentence_length, len(sentence_embedding))
            batch_sentence_embedding.append(sentence_embedding)

        ############ Word-level LSTM ############
        # Pad each sentence to max length
        empty_word_embedding = [PAD_TARGET_INDEX for i in range(self.conv_filters + self.word_embedding_dim)]
        for sent_index, sentence in enumerate(batch_sentence_embedding):
            for i in range(len(sentence), max_sentence_length):
                sentence.append(torch.tensor(empty_word_embedding.copy(), dtype=torch.float).to(device))
            batch_sentence_embedding[sent_index] = torch.stack(sentence).to(device)
        batch_sentence_embedding = torch.stack(batch_sentence_embedding).to(device)
        # Run batch through LSTM
        lstm_out, self.lstm_hidden_embeddings = self.lstm(batch_sentence_embedding.view(batch_sentence_embedding.size()[1], len(char_indices_batch), -1), self.lstm_hidden_embeddings)
        tag_space = self.dense(lstm_out.view(len(char_indices_batch), lstm_out.size()[0], -1))
        tag_scores = F.log_softmax(tag_space, dim=2).to(device)
        return tag_space, max_sentence_length


def load_model(model_file):
    '''
    Loads model from model_file
    '''
    model = torch.load(model_file)
    return model

def tag_sentence(test_file, model_file, out_file):
    start = datetime.now()

    # Load model and dictionaries
    model = load_model(model_file)
    print(model.char_dict)

    end = datetime.now()
    print('Finished... Took {}'.format(end - start))

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)
