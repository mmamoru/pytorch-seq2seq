#!/usr/bin/env python
# coding: utf-8


import os
import torch
import torch.nn as nn
from Voc import *
from model_lstm import *

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        #decoder_hidden = encoder_hidden[: self.decoder.n_layers]
        decoder_hidden = (encoder_hidden[0][: self.decoder.n_layers], encoder_hidden[1][: self.decoder.n_layers])
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return [all_tokens], [-torch.sum(torch.log(all_scores).detach().cpu())/float(max_length - 1 + 1e-6)]
    
    
import sentencepiece as spm
segmentation_model_position = './data'
segmentation_model_name = 'train_model32000.model'

spp = spm.SentencePieceProcessor()
spp.Load(os.path.join(segmentation_model_position, segmentation_model_name))
def sentencePieceNormalizeString(input_sentence):
    splitSentence = ' '.join(spp.EncodeAsPieces(input_sentence))
    return splitSentence


import MeCab
tagger = MeCab.Tagger("-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd/")

def mecabNormalizeString(input_sentence):
    splitSentence = ' '.join([ele.split("\t")[0] for ele in tagger.parse(input_sentence).split("\n")[:-2]])
    return splitSentence




class BeamSearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, beam_width, n_best):
        super(BeamSearchDecoder, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.beam_width = beam_width
        self.n_best = n_best
        
    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
#         decoder_hidden = encoder_hidden[: self.decoder.n_layers]
        decoder_hidden = (encoder_hidden[0][: self.decoder.n_layers], encoder_hidden[1][: self.decoder.n_layers])
        # Initialize decoder input with SOS_token
        #decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        
        
        class BeamSearchNode(object):
            def __init__(self, hid, prevNode, wordId, logP, length):
                self.hid = hid
                self.prevNode = prevNode
                self.wordId = wordId
                self.logP = logP
                self.length = length
                
            def eval(self, alpha=1.0):
                reward = 0
                # Add here a function for shaping a reward

                return self.logP / float(self.length - 1 + 1e-6) + alpha * reward
          
        
        n_best_batch_list = []
        n_best_batch_score_list = []
        for batchNum in range(encoder_hidden[0].size(1)):
            
            decoder_input = torch.LongTensor([[SOS_token]]).to(device)
            # 一つ前の状態の隠れベクトル、単語をNodeを保持するNodeを生成
            node = BeamSearchNode(hid=(decoder_hidden[0][:,batchNum,:].unsqueeze(1), decoder_hidden[1][:,batchNum,:].unsqueeze(1)),
                                  prevNode=None, wordId=decoder_input, logP=0, length=1)
            nextNodes=[]
            
            nextNodes.append((-node.eval(), id(node), node))
            n_dec_steps = 0
            while True:
                
                nodes = [sorted(nextNodes)[inode] for inode in range(len(nextNodes)) if inode<self.beam_width]
                nextNodes = []
                end_node = []
                for beamNum in range(self.beam_width):
                    if len(nodes)<=0:
                        break
                    #今から探索するNodeを取得
                    score, _, n = nodes.pop(0)
                    decoder_input = n.wordId
                    decoder_hidden = n.hid
                    
                    if n.wordId[0][0].item()!=EOS_token and n.wordId[0][0].item()!=PAD_token and n.length<=max_length:
                        decoder_output, decoder_hidden = self.decoder(
                            decoder_input, decoder_hidden, encoder_outputs
                        )

                        topk_prob, topk_indexes = torch.topk(decoder_output, self.beam_width) 

                        for new_k in range(self.beam_width):
                            decoded_t = topk_indexes[0][new_k].view(1,1) # (1)
                            logp = torch.log(topk_prob[0][new_k]).item() # float log probability val

                            node = BeamSearchNode(hid=decoder_hidden,
                                                  prevNode=n,
                                                  wordId=decoded_t,
                                                  logP=n.logP+logp,
                                                  length=n.length+1)
                            nextNodes.append((-node.eval(), id(node), node))
                            
                            
                    else:
                        node = BeamSearchNode(hid=decoder_hidden,
                                                  prevNode=n,
                                                  wordId=torch.LongTensor([[PAD_token]]).to(device),
                                                  logP=n.logP,
                                                  length=n.length)
                        nextNodes.append((-node.eval(), id(node), node))
                        end_node.append(1)
                if len(end_node)>=self.beam_width:
                    break
            
            
        
            if len(nextNodes)!=self.beam_width:
                print('assert not match beam width and nextNode length')
                
            n_best_seq_list = []
            n_best_score_list = []
            for score, _id, n in sorted(nextNodes):
                sequence = [n.wordId.item()]
                n_best_score_list.append(score)
                # back trace from end node
                while n.prevNode is not None:
                    n = n.prevNode
                    sequence.append(n.wordId.item())
                    
                sequence = sequence[::-1] # reverse
                n_best_seq_list.append(sequence)
        
            n_best_seq_list = n_best_seq_list[::-1]
            n_best_score_list = n_best_score_list[::-1]
            
            n_best_batch_list.append(n_best_seq_list)
            n_best_batch_score_list.append(n_best_score_list)
            
        n_best_batch_list = torch.tensor(n_best_batch_list)
        n_best_batch_score_list = torch.tensor(n_best_batch_score_list)
            
            #batchを無視して今回は出力することにした
        return n_best_batch_list[0], n_best_batch_score_list[0]
    


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH, viewScore=False):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    seqs, scores = searcher(input_batch, lengths, max_length)
    
    for iseqs in range(len(seqs)):
        tokens = seqs[iseqs]
    
        # indexes -> words
        decoded_words = [voc.index2word[token.item()] for token in tokens]
        if viewScore:
            print(scores[iseqs],decoded_words)
    return decoded_words


def evaluateInput(encoder, decoder, searcher, searcher1, voc, mecab_or_sentencePiece, viewScore=True):
    input_sentence = ""
    while 1:
        try:
            # Get input sentence
            input_sentence = input("> ")
            # Check if it is quit case
            if input_sentence == "q" or input_sentence == "quit":
                break
            # Normalize sentence
            if mecab_or_sentencePiece=="mecab":
                input_sentence = mecabNormalizeString(input_sentence)
                
            elif mecab_or_sentencePiece=="sentencePiece":
                input_sentence = sentencePieceNormalizeString(input_sentence)
                
            print('\n入力を分割すると',input_sentence+'\n')
            
            # Evaluate sentence with seacher
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence, viewScore=viewScore)
            # Format and print response sentence
            output_words[:] = [
                x for x in output_words if not (x == "EOS" or x == "PAD" or x == "SOS")
            ]
            print("\nBot(BeamSearch):", " ".join(output_words))
            
            print()
            
            # Evaluate sentence with seacher1
            output_words = evaluate(encoder, decoder, searcher1, voc, input_sentence, viewScore=viewScore)
            # Format and print response sentence
            output_words[:] = [
                x for x in output_words if not (x == "EOS" or x == "PAD" or x == "SOS")
            ]
            print("\nBot(GreedySearch):", " ".join(output_words)+"\n")

        except KeyError:
            print("Error: Encountered unknown word.")



