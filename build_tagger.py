# python3.5 build_tagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
from random import shuffle
import pickle as pk

torch.manual_seed(1)

# Parameters
use_gpu = torch.cuda.is_available()
device = torch.device("cpu")
if use_gpu:
    print("Running train with GPU...")
    device = torch.device("cuda:0")
epochs = 5
batch_size = 16
debug = False

# Constants
UNK_KEY = '<UNK>'
PAD_KEY = '<PAD>'
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
        self.word_embedding_dim = 200
        self.conv_filters = 32
        self.conv_kernel = 2
        self.lstm_hidden_dim = self.word_embedding_dim
        self.lstm_num_layers = 2
        self.dropout = 0.5

        # Embeddings
        self.char_embeddings = nn.Embedding(len(self.char_dict), self.char_embedding_dim).to(device)
        self.word_embeddings = nn.Embedding(len(self.word_dict), self.word_embedding_dim).to(device)
        self.lstm_hidden_embeddings = self.init_hidden_embeddings(batch_size)

        # Layers
        self.conv = nn.Conv1d(self.char_embedding_dim, self.conv_filters, self.conv_kernel, bias=True, padding=(self.conv_kernel // 2)).to(device)
        self.lstm = nn.LSTM(self.word_embedding_dim + self.conv_filters, self.lstm_hidden_dim, dropout=self.dropout, num_layers=self.lstm_num_layers, batch_first=True).to(device)
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
        # Character-level Embedding
        word_embedding_char_level = char_indices_batch.view(-1, char_indices_batch.shape[-1])
        word_embedding_char_level = self.char_embeddings(word_embedding_char_level)
        # Run embeddings through CNN
        word_embedding_char_level = word_embedding_char_level.view(word_embedding_char_level.shape[0], -1, word_embedding_char_level.shape[1])
        word_embedding_char_level = self.conv(word_embedding_char_level)
        word_embedding_char_level = torch.max(word_embedding_char_level, 2)[0]
        # Word-level Embedding
        word_embedding_word_level = word_indices_batch.view(-1)
        word_embedding_word_level = self.word_embeddings(word_embedding_word_level)
        # Concatenate Character-level and Word-level Embeddings
        batch_sentence_embedding = torch.cat((word_embedding_word_level, word_embedding_char_level), dim=1).to(device)
        batch_sentence_embedding = batch_sentence_embedding.view(char_indices_batch.shape[0], -1, batch_sentence_embedding.shape[-1])
        # Run Embeddings through LSTM
        lstm_out, self.lstm_hidden_embeddings = self.lstm(batch_sentence_embedding, self.lstm_hidden_embeddings)
        tag_space = self.dense(lstm_out)
        tag_scores = F.log_softmax(tag_space, dim=2).to(device)
        return tag_space


class TextDataset(Dataset):
    def __init__(self, char_dict, word_dict, tag_dict, sentence_tags):
        '''
        Args:
            train_file: the input file with POS tagged sentences.
        '''
        self.char_dict = char_dict
        self.word_dict = word_dict
        self.tag_dict = tag_dict
        self.sentence_tags = sentence_tags
        
    def __len__(self):
        return len(self.sentence_tags)

    def __getitem__(self, idx):
        '''
        Args:
            idx: corresponds to which line in the data we want to retrieve
        Returns the sentence text, tags 
        '''
        sentence, tags = self.sentence_tags[idx]
        return ' '.join(sentence), ' '.join(tags)

    
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
            
    return build_dictionary(set(char_set), add_unknown=True, add_pad=True), build_dictionary(set(word_set), add_unknown=True, add_pad=True), build_dictionary(set(tag_set), include_reverse=True), preprocessed_data
    
def build_dictionary(item_set, add_unknown=False, include_reverse=False, add_pad=False):
    '''
    Given a set of items, return a dictionary of the form
    {
        item: index,
        ...
    }
    where each index is unique and in increasing order.
    '''
    result = {}
    result_reversed = {}
    if add_pad: result[PAD_KEY] = len(result)
    for item in item_set:
        result[item] = len(result)
        if include_reverse: result_reversed[len(result_reversed)] = item
    if add_unknown: result[UNK_KEY] = len(result)
    if include_reverse: return result, result_reversed
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
    '''
    word_indices = []
    char_indices = []
    max_word_length = 0
    for token in sentence:
        word_indices.append(word_dict[token])
        char_indices_for_word = []
        for c in list(token):
            char_indices_for_word.append(char_dict[c])
        char_indices.append(char_indices_for_word)
        max_word_length = max(max_word_length, len(char_indices_for_word))
    return char_indices, word_indices, max_word_length

def get_tag_indices(tags, tag_dict):
    '''
    Given a list of tags, output the corresponding sequence of indices based on the tag dictionary
    '''
    tag_indices = [tag_dict[tag] for tag in tags]
    return tag_indices

def batch_to_indices(batch, char_dict, word_dict, tag_dict):
    '''
    For each batch of sentences, get the list of char indices, word indices and tag indices
    '''
    char_indices_batch = []
    word_indices_batch = []
    tag_indices_batch = []
    max_sent_length = 0
    max_word_length = 0
    sentences_batch, tags_batch = batch
    for i in range(len(sentences_batch)):
        # Split each data row into list of tokens and tags
        sentence = sentences_batch[i].split(" ")
        tags = tags_batch[i].split(" ")
        # Get indices from dictionaries
        char_indices, word_indices, curr_max_word_length = get_word_char_indices(sentence, char_dict, word_dict)
        tag_indices = get_tag_indices(tags, tag_dict)
        # Append indices to batch indices list
        char_indices_batch.append(char_indices)
        word_indices_batch.append(word_indices)
        tag_indices_batch.append(tag_indices)
        # Update max lengths
        max_sent_length = max(max_sent_length, len(word_indices))
        max_word_length = max(max_word_length, curr_max_word_length)
    return char_indices_batch, word_indices_batch, tag_indices_batch, max_sent_length, max_word_length

def pad_indices(char_indices_batch, word_indices_batch, tag_indices_batch, char_dict, word_dict, max_sent_length, max_word_length):
    ''' 
    Pad each sentence to same sentence length and each word to same word length
    '''
    # Pad sentences to max sentence length
    for i in range(len(word_indices_batch)):
        for j in range(len(word_indices_batch[i]), max_sent_length):
            word_indices_batch[i].append(word_dict[PAD_KEY])
            tag_indices_batch[i].append(PAD_TARGET_INDEX)
            char_indices_batch[i].append([char_dict[PAD_KEY]])
        word_indices_batch[i] = torch.tensor(word_indices_batch[i], dtype=torch.long).to(device)
        tag_indices_batch[i] = torch.tensor(tag_indices_batch[i], dtype=torch.long).to(device)
    # Pad words to max word length
    for i in range(len(char_indices_batch)):
        for j in range(len(char_indices_batch[i])):
            for l in range(len(char_indices_batch[i][j]), max_word_length):
                char_indices_batch[i][j].append(char_dict[PAD_KEY])
            char_indices_batch[i][j] = torch.tensor(char_indices_batch[i][j], dtype=torch.long).to(device)
        char_indices_batch[i] = torch.stack(char_indices_batch[i]).to(device)
    return torch.stack(char_indices_batch).to(device), torch.stack(word_indices_batch).to(device), torch.stack(tag_indices_batch).to(device)

def save_model(model_file, model):
    '''
    Save model into model_file 
    '''
    torch.save(model, model_file)

def train_model(train_file, model_file):
    '''
    This is the main training method. Here we prepare the dataset, model, run the training, then save the model.
    '''
    start = datetime.now()

    # Prepare dataset
    lines = read_input(train_file)                 
    char_dict, word_dict, (tag_dict, tag_dict_reversed), data = preprocess(lines)
    dataset = TextDataset(char_dict, word_dict, tag_dict, data)

    # Prepare model
    model = POSTagger(char_dict, word_dict, tag_dict)
    if use_gpu:
        model = model.cuda()
    loss_function = nn.CrossEntropyLoss(ignore_index=PAD_TARGET_INDEX).to(device)                               # TODO: Can try setting weight?
    optimizer = optim.Adam(model.parameters()) 

    # Train model
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    for epoch in range(epochs):
        for batch_index, batch in enumerate(dataloader):
            start_batch = datetime.now()

            # Clear gradients and hidden layer
            model.zero_grad()
            model.lstm_hidden_embeddings = model.init_hidden_embeddings(len(batch[0]))

            # Prepare input to model
            char_indices_batch, word_indices_batch, tag_indices_batch, max_sent_length, max_word_length = batch_to_indices(batch, char_dict, word_dict, tag_dict)
            char_indices_batch, word_indices_batch, tag_indices_batch = pad_indices(char_indices_batch, word_indices_batch, tag_indices_batch, char_dict, word_dict, max_sent_length, max_word_length)

            # Forward pass
            tag_scores_batch = model(char_indices_batch, word_indices_batch)

            # Prepare output
            target = tag_indices_batch.view(-1).to(device)
            output = tag_scores_batch.view(-1, tag_scores_batch.size()[-1])

            # Backward pass
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()

            # if debug or batch_index + 1 == math.ceil(len(data)/batch_size): 
            #     # Print loss and accuracy
            #     num_correct = 0
            #     num_predictions = 0
            #     for i in range(output.size()[0]):
            #         if target[i].item() == PAD_TARGET_INDEX:
            #             continue
            #         max_prob, predicted_index = torch.max(output[i], 0)
            #         num_predictions += 1
            #         if predicted_index.item() == target[i].item():
            #             num_correct += 1
            #     end_batch = datetime.now()
            #     print("Epoch {}/{} | Batch {}/{} ||| Loss {:.3f} | Accuracy {:.3f} ||| {}".format(epoch + 1, epochs, batch_index + 1, math.ceil(len(data)/batch_size), loss.data.item(), num_correct / num_predictions, end_batch - start_batch))
            # else:
            #     end_batch = datetime.now()
            #     print("Epoch {}/{} ||| Batch {}/{} ||| Loss {:.3f} ||| {}".format(epoch + 1, epochs, batch_index + 1, math.ceil(len(data)/batch_size), loss.data.item(), end_batch - start_batch))

    # Save model
    model.tag_dict_reversed = tag_dict_reversed
    save_model(model_file, model)

    end = datetime.now()
    print('Finished... Took {}'.format(end - start))
		
if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
