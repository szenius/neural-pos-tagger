# python3.5 build_tagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence

import numpy as np

torch.manual_seed(1)

# Parameters
use_gpu = torch.cuda.is_available()
device = torch.device("cpu")
if use_gpu:
    device = torch.device("cuda:0")
epochs = 10
batch_size = 32

# Constants
PAD_KEY = '<PAD>'

class POSTagger(nn.Module):
    def __init__(self, charset_size, vocab_size, tagset_size):
        super(POSTagger, self).__init__()

        # Hyperparameters
        self.char_embedding_dim = 10
        self.word_embedding_dim = 250
        self.conv_in_channels = 1
        self.conv_filters = 32
        self.conv_kernel = 3
        self.maxpool_kernel = 1
        self.lstm_hidden_dim = 250
        self.dropout = 0.5

        # Embeddings
        self.char_embeddings = nn.Embedding(charset_size, self.char_embedding_dim).to(device)
        self.word_embeddings = nn.Embedding(vocab_size, self.word_embedding_dim).to(device)
        self.lstm_hidden_embeddings = self.init_hidden_embeddings()

        # Layers
        self.conv = nn.Conv1d(self.char_embedding_dim, self.conv_filters, self.conv_kernel, bias=True, padding=(self.conv_kernel // 2)).to(device)
        self.lstm = nn.LSTM(self.word_embedding_dim + self.conv_filters, self.lstm_hidden_dim, dropout=self.dropout).to(device)
        self.dense = nn.Linear(self.lstm_hidden_dim, tagset_size).to(device)
    
    def init_hidden_embeddings(self):
        return (torch.zeros(1, batch_size, self.lstm_hidden_dim).to(device),
                torch.zeros(1, batch_size, self.lstm_hidden_dim).to(device))
    
    def forward(self, char_indices_batch, word_indices_batch):
        '''
        Input sentence should be a list of tokens.
        Runs Character level CNN + Word level bi-directional LSTM.
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
        # Pad each sentence to max length
        empty_word_embedding = [-1 for i in range(self.conv_filters + self.word_embedding_dim)]
        for sent_index, sentence in enumerate(batch_sentence_embedding):
            for i in range(len(sentence), max_sentence_length):
                sentence.append(torch.tensor(empty_word_embedding.copy(), dtype=torch.float))
            batch_sentence_embedding[sent_index] = torch.stack(sentence).to(device)
        batch_sentence_embedding = torch.stack(batch_sentence_embedding).to(device)

        ############ Word-level LSTM ############
        lstm_out, self.lstm_hidden_embeddings = self.lstm(batch_sentence_embedding.view(batch_sentence_embedding.size()[1], len(char_indices_batch), -1), self.lstm_hidden_embeddings)
        tag_space = self.dense(lstm_out.view(len(char_indices_batch), lstm_out.size()[0], -1))
        tag_scores = F.log_softmax(tag_space, dim=2).to(device)
        return tag_space, max_sentence_length
        

def preprocess(lines):
    '''
    Read list of sentences, extract characters, words and POS tags, and build a dictionary each for
    all three. Each dictionary is of the form:
    {
        'a': 0,
        'b': 1,
        'c': 2,
        ...
    }
    Also preprocess each sentence into the following form:
    (
        [tokens], [tags]
    )
    '''
    word_set = []
    tag_set = []
    char_set = []
    preprocessed_data = []
    for sentence in lines:
        sent_tokens = sentence.split(" ")
        preprocessed_sent_tokens = []
        sent_tags = []
        for sent_token in sent_tokens:
            tokens = sent_token.split("/")
            tag_set.append(tokens[-1])                                      # Add to tag set
            for token in tokens[:-1]:
                word_set.append(token)                                      # Add to word set
                char_set.extend(list(token))                                # Add to char set
            sent_tags.append(tokens[-1])
            preprocessed_sent_tokens.append(tokens[0])                      # TODO: currently, given word1/word2/tag, ignore word2
        preprocessed_data.append((preprocessed_sent_tokens, sent_tags))     # Add to preprocessed sentences
            
    return build_dictionary(set(char_set)), build_dictionary(set(word_set)), build_dictionary(set(tag_set)), preprocessed_data
    
def build_dictionary(item_set):
    '''
    Given a set of items, return a dictionary of the form
    {
        item: index,
        ...
    }
    where each index is unique and in increasing order.
    '''
    result = {}
    for item in item_set:
        result[item] = len(result)
    return result

def read_input(fname):
    '''
    Read file into list of lines
    '''
    with open(fname) as f:
        lines = f.readlines()
    return [x.strip() for x in lines]   

def get_word_char_indices(sentence, char_dict, word_dict):
    '''
    Given a sentence, the character and word dictionaries, generate the sequences of 
    word indices and character indices for this particular sentence.
    Also return max_char_length, which is the length of the longest word.
    '''
    word_indices = []
    char_indices = []
    for token in sentence:
        word_indices.append(word_dict[token])
        char_indices_for_word = []
        for c in list(token):
            char_indices_for_word.append(char_dict[c])
        char_indices.append(char_indices_for_word)
    return char_indices, word_indices

def get_tag_indices(tags, tag_dict):
    '''
    Given a list of tags, output the corresponding sequence of indices based on the tag dictionary
    '''
    tag_indices = [tag_dict[tag] for tag in tags]
    return tag_indices

def batch_data(data, batch_size):
    '''
    Split dataset into batches
    '''
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def get_indices_for_batch(batch, char_dict, word_dict, tag_dict):
    char_indices_batch = []
    word_indices_batch = []
    tag_indices_batch = []
    for sentence, tags in batch:
        # Get indices from dictionaries
        char_indices, word_indices = get_word_char_indices(sentence, char_dict, word_dict)
        tag_indices = get_tag_indices(tags, tag_dict)
        # Append indices to batch indices list
        char_indices_batch.append(char_indices)
        word_indices_batch.append(word_indices)
        tag_indices_batch.append(tag_indices)
    return char_indices_batch, word_indices_batch, tag_indices_batch

def train_model(train_file, model_file):
    # Prepare dataset
    lines = read_input(train_file)                 
    char_dict, word_dict, tag_dict, preprocessed_data = preprocess(lines)
    preprocessed_data = list(batch_data(preprocessed_data, batch_size))

    # Prepare model
    model = POSTagger(len(char_dict), len(word_dict), len(tag_dict))
    if use_gpu:
        model = model.cuda()
    loss_function = nn.CrossEntropyLoss().to(device)                               # TODO: Can try setting weight?
    optimizer = optim.Adam(model.parameters()) 

    # Train model
    for epoch in range(epochs):
        print("STARTING EPOCH {}/{}...".format(epoch + 1, epochs))
        for batch_index, batch in enumerate(preprocessed_data):
            # Clear gradients and hidden layer
            model.zero_grad()
            model.lstm_hidden_embeddings = model.init_hidden_embeddings()

            # Prepare input to model
            char_indices_batch, word_indices_batch, tag_indices_batch = get_indices_for_batch(batch, char_dict, word_dict, tag_dict)

            # Forward pass
            tag_scores_batch, max_sentence_length = model(char_indices_batch, word_indices_batch)

            # Pad output to max sentence length
            for idx, tag_indices in enumerate(tag_indices_batch):
                for i in range(len(tag_indices), max_sentence_length):
                    tag_indices.append(-1)
                tag_indices_batch[idx] = torch.tensor(tag_indices, dtype=torch.long).to(device)
            tag_indices_batch = torch.stack(tag_indices_batch).to(device)

            # Backward pass
            print(tag_scores_batch.size(), tag_indices_batch.size())
            loss = loss_function(tag_scores_batch, tag_indices_batch)
            loss.backward()
            optimizer.step()

            # Print loss and accuracy
            num_correct = 0
            num_predictions = 0
            for tag_scores in tag_scores_batch:
                for i in range(tag_scores.size()[0]):
                    max_prob, predicted_index = torch.max(tag_scores[i], 0)
                    num_predictions += 1
                    if predicted_index.item() == tag_indices[i].item():
                        num_correct += 1
            print("Epoch {}/{} | Batch {}/{}: Loss {:.3f} | Accuracy {:.3f}".format(epoch + 1, epochs, batch_index + 1, len(preprocessed_data), loss.data.item(), num_correct / num_predictions))

     
    print('Finished...')
		
if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
