# python3.5 run_tagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>

import os
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle as pk
from datetime import datetime

torch.manual_seed(1)

# Parameters
use_gpu = torch.cuda.is_available()
device = torch.device("cpu")
if use_gpu:
    print("Running test with GPU...")
    device = torch.device("cuda:0")
batch_size = 1

# Constants
UNK_KEY = '<UNK>'
PAD_TARGET_INDEX = -1

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
    model = torch.load(model_file, map_location=device)
    return model

def preprocess(lines):
    '''
    Preprocess each sentence into list of tokens
    Return list of preprocessed sentences
    '''
    preprocessed_data = []
    for sentence in lines:
        preprocessed_data.append(sentence.split(" "))
    return preprocessed_data

def read_input(fname):
    '''
    Read file into list of lines
    '''
    with open(fname) as f:
        lines = f.readlines()
    return [x.strip() for x in lines]   

def batch_data(data, batch_size):
    '''
    Split dataset into batches
    '''
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def get_word_char_indices(sentence, char_dict, word_dict):
    '''
    Given a sentence, the character and word dictionaries, generate the sequences of 
    word indices and character indices for this particular sentence.
    Also return max_char_length, which is the length of the longest word.
    '''
    word_indices = []
    char_indices = []
    for token in sentence:
        if '/' in token: token = token.split('/')[0]        # TODO: Given word1/word2/tag, only add word1 indices

        # Read word index from dictionary
        if token in word_dict:
            word_indices.append(word_dict[token])
        else: word_indices.append(word_dict[UNK_KEY])

        # Read character indices from dictionary
        char_indices_for_word = []
        for c in list(token):
            if c in char_dict:
                char_indices_for_word.append(char_dict[c])
            else: char_indices_for_word.append(char_dict[UNK_KEY])
        char_indices.append(char_indices_for_word)

    return char_indices, word_indices

def sentences_to_indices(sentences, char_dict, word_dict):
    '''
    Expects sentences to be a list of list of tokens.
    Converts sentences to character indices and word indices respectively.
    '''
    char_indices_list = []
    word_indices_list = []
    for sentence in sentences:
        char_indices, word_indices = get_word_char_indices(sentence, char_dict, word_dict)
        char_indices_list.append(char_indices)
        word_indices_list.append(word_indices)
    return char_indices_list, word_indices_list

def add_tags_to_sentence(tokens, predicted_tags):
    '''
    Given a list of tokens (belonging to a sentence) and the predicted tags per token, collapse them into
    a single String in POS tagged sentence format and return
    '''
    for i in range(len(tokens)):
        tokens[i] = '/'.join([tokens[i], predicted_tags[i]])
    return ' '.join(tokens)

def save_answer(out_file, output):
    '''
    Expects output to be list of Strings. Writes output to out_file.
    '''
    with open(out_file, 'w') as f:
        for item in output:
            f.write("%s\n" % item)

def tag_sentence(test_file, model_file, out_file):
    start = datetime.now()

    # Load model
    model = load_model(model_file)
    model.lstm_hidden_embeddings = model.init_hidden_embeddings(batch_size)
    model.lstm.flatten_parameters()

    # Load dictionaries
    char_dict = model.char_dict
    word_dict = model.word_dict
    tag_dict_reversed = model.tag_dict_reversed

    # Load test dataset
    lines = read_input(test_file)
    sentences_as_tokens_lists = preprocess(lines)
    char_indices, word_indices = sentences_to_indices(sentences_as_tokens_lists, char_dict, word_dict)

    # Run test dataset through model
    output = []
    for i in range(len(char_indices)):
        out_probs, max_sentence_length = model([char_indices[i]], [word_indices[i]])
        out_probs = torch.squeeze(out_probs).to(device)
        predicted_tags = []
        for pset in out_probs:
            max_prob, predicted_index = torch.max(pset, 0)
            predicted_tags.append(tag_dict_reversed[predicted_index.item()])
        output.append(add_tags_to_sentence(sentences_as_tokens_lists[i], predicted_tags))
    
    # Save answer
    save_answer(out_file, output)

    end = datetime.now()
    print('Finished... Took {}'.format(end - start))

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)
