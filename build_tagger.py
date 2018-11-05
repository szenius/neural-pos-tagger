# python3.5 build_tagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# Parameters
use_gpu = torch.cuda.is_available()
device = torch.device("cpu")
if use_gpu:
    device = torch.device("cuda:0")
epochs = 10
batch_size = 32

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
        return (torch.zeros(1, 1, self.lstm_hidden_dim).to(device),
                torch.zeros(1, 1, self.lstm_hidden_dim).to(device))
    
    def forward(self, char_indices, word_indices):
        '''
        Input sentence should be a list of tokens.
        Runs Character level CNN + Word level bi-directional LSTM.
        '''
        # Character-level CNN
        sentence_embedding = []
        for i in range(len(char_indices)):
            # Get word embedding by running character embeddings through CNN
            word_embedding_char_level = []
            for idx in char_indices[i]:
                char_embedding = self.char_embeddings(torch.tensor(idx, dtype=torch.long).to(device))
                word_embedding_char_level.append(char_embedding)
            # Prepare stack of character embeddings
            word_embedding_char_level = torch.stack(word_embedding_char_level).to(device)
            word_embedding_char_level = word_embedding_char_level.permute(1, 0)
            word_embedding_char_level = torch.stack([word_embedding_char_level]).to(device)
            # Convolution on character embeddings
            word_embedding_char_level = self.conv(word_embedding_char_level)
            word_embedding_char_level = torch.max(word_embedding_char_level, 2)[0][0]

            # Get word embedding from tokens
            word_embedding_word_level = self.word_embeddings(torch.tensor(word_indices[i], dtype=torch.long).to(device))

            # Concat word embeddings and add to sentence embedding
            word_embedding = torch.cat((word_embedding_char_level, word_embedding_word_level), 0).to(device)
            sentence_embedding.append(word_embedding)
        sentence_embedding = torch.stack(sentence_embedding).to(device)

        # Word-level LSTM
        lstm_out, self.lstm_hidden_embeddings = self.lstm(sentence_embedding.view(len(char_indices), 1, -1), self.lstm_hidden_embeddings)
        tag_space = self.dense(lstm_out.view(len(char_indices), -1))
        tag_scores = F.log_softmax(tag_space, dim=1).to(device)
        return tag_scores
        

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
    return torch.tensor(tag_indices, dtype=torch.long)

def batch_data(data, batch_size):
    '''
    Split dataset into batches
    '''
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

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
            char_indices_batch = []
            word_indices_batch = []
            tag_indices_batch = []
            for sentence, tags in batch:
                char_indices, word_indices = get_word_char_indices(sentence, char_dict, word_dict)
                tag_indices = get_tag_indices(tags, tag_dict)
                char_indices_batch.append(char_indices)
                word_indices_batch.append(word_indices)
                tag_indices_batch.append(tag_indices)

            # Forward pass
            tag_scores_batch = model(char_indices_batch, word_indices_batch)

            # Backward pass
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
