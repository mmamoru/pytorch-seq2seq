#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os




PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


import random
import itertools


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(" ")] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))




def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m



# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)  # Byte?
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len






class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            dropout=(0 if n_layers == 1 else dropout),
            bidirectional=True,
        )

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)

        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, : self.hidden_size] + outputs[:, :, self.hidden_size :]
        # [T x B x *]
        # T is the length of the longest sequence
        # B is the batch size
        # * is any number of dimensions
        
        # hidden.size()=[(layer x numDirection) x B x *]
        
        # Return output and final hidden state
        return outputs, hidden







class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method not in ["dot", "general", "concat"]:
            raise ValueError(self.method, "is not an appropriate attention method.")

        elif self.method == "general":
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)

        elif self.method == "concat":
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))
            print('self.vのパラメタNaNの数',torch.sum(torch.isnan(self.v)))

    def dot_score(self, hidden, encoder_outputs):
        return torch.sum(encoder_outputs * hidden, dim=2)

    def concat_score(self, hidden, encoder_outputs):
        attn_energies = self.attn(
            torch.cat(
                (hidden.expand(encoder_outputs.size(0), -1, -1), encoder_outputs), dim=2
            )
        ).tanh()
        
        return torch.sum(attn_energies * self.v, dim=2)

    def general_score(self, hidden, encoder_outputs):
        energy = self.attn(encoder_outputs)
        return torch.sum(energy * hidden, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == "general":
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == "concat":
            attn_energies = self.concat_score(hidden, encoder_outputs)

        elif self.method == "dot":
            attn_energies = self.dot_score(hidden, encoder_outputs)
            
        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()
        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)




class AttnDecoderRNN(nn.Module):
    def __init__(
        self, attn_model, embeding, hidden_size, output_size, n_layers=1, dropout=0.1
    ):
        super(AttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embeding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            dropout=(0 if n_layers == 1 else dropout),
        )
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # rnn_output.Size() = [1(sequenceLength) x B x *]
        # hidden.Size() = [layer x B x *]
        
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden



# パラメータカウント関数
def parameters_count(net):
    params = 0
    for p in net.parameters():
        #print(p)
        #print(p.numel())
        if p.requires_grad:
            params += p.numel()
        
    print(params)







