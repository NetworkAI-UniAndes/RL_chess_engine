from unittest.util import _MAX_LENGTH
import pandas as pd
import numpy as np
import torch 
import chess.pgn
import os
import fnmatch
import math
from stockfish import Stockfish
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import DataLoader
from stockfish import Stockfish
from torchtext.vocab import build_vocab_from_iterator
from torch.optim import Adam
from ast import literal_eval


def get_moves_dict():
    letters=list('abcdefgh')
    pawn_crown=[]
    pieces=list('qnrb')
    for x in range(len(letters)):
        for piece in pieces:
            if letters[x]== 'a':
                pawn_crown.append(letters[x]+'2'+letters[x]+'1'+piece)
                pawn_crown.append(letters[x]+'2'+letters[x+1]+'1'+piece)
                pawn_crown.append(letters[x]+'7'+letters[x]+'8'+piece)
                pawn_crown.append(letters[x]+'7'+letters[x+1]+'8'+piece)
            elif letters[x]== 'h':

                pawn_crown.append(letters[x]+'2'+letters[x]+'1'+piece)
                pawn_crown.append(letters[x]+'2'+letters[x-1]+'1'+piece)
                pawn_crown.append(letters[x]+'7'+letters[x]+'8'+piece)
                pawn_crown.append(letters[x]+'7'+letters[x-1]+'8'+piece)

            else:
                pawn_crown.append(letters[x]+'2'+letters[x]+'1'+piece)
                pawn_crown.append(letters[x]+'2'+letters[x-1]+'1'+piece)
                pawn_crown.append(letters[x]+'2'+letters[x+1]+'1'+piece)
                pawn_crown.append(letters[x]+'7'+letters[x]+'8'+piece)
                pawn_crown.append(letters[x]+'7'+letters[x-1]+'8'+piece)
                pawn_crown.append(letters[x]+'7'+letters[x+1]+'8'+piece)
                
    board_spaces=[x+str(y) for x in letters for y in range(1,9)]
    possible_moves=[x+str(y) for x in board_spaces for y in board_spaces]+pawn_crown
    moves_dict={ key:label for label,key in enumerate(possible_moves)}
    moves_dict['start']=len(moves_dict.values())
    moves_dict['end']=len(moves_dict.values())
    inv_moves_dict = {v: k for k, v in moves_dict.items()}
    return moves_dict,inv_moves_dict 
class Tokenizer:
    def __init__(self,path=None, max_length=76,vocab_size=323):
        self.max_length=max_length
        self.dictionary_board= {
            "/":0, #change value, and keep zero to mask
            "r":9,
            "n":10,
            "b":11,
            "q":12,
            "k":13,
            "p":14,
            "R":15,
            "N":16,
            "B":17,
            "Q":18,
            "K":19,
            "P":20,
            " ":321,
            "<PAD>":322
        }
        self.detokenizer_dict={v: k for k, v in self.dictionary_board.items()}
        self.dictionary_active_color={
            "w":21,
            "b":22
        }
        self.detokenizer_dict.update({v: k for k, v in self.dictionary_active_color.items()})
        self.dictionary_castling_rights={
            "K":23,
            "Q":24,
            "k":25,
            "q":26,
            "-":27
        }
        self.detokenizer_dict.update({v: k for k, v in self.dictionary_castling_rights.items()})
        self.dictionary_en_passant_square={
            "-":28,
            "a3":29,
            "b3":30,
            "c3":31,
            "d3":32,
            "e3":33,
            "f3":34,
            "g3":35,
            "h3":36,
            "a6":37,
            "b6":38,
            "c6":39,
            "d6":40,
            "e6":41,
            "f6":42,
            "g6":43,
            "h6":44
        }
        self.detokenizer_dict.update({v: k for k, v in self.dictionary_en_passant_square.items()})
        self.dictionary_half_move_clock={
        }
        self.dictionary_full_move_number={}
        
        self.vocab_size=vocab_size
        
        
    def board_to_token_vector(self, board):
        #print(board)
        #takes every charater on the board and returns a token vector
        return [ int(char) if char.isnumeric() else self.dictionary_board[char] for char in board ]



    def tokenize(self, fen):
        token_vector =[]
        fen_components=fen.split(" ")

        token_vector+= self.board_to_token_vector(fen_components[0])
        token_vector+=[self.dictionary_board[" "]]
        token_vector+= [self.dictionary_active_color[fen_components[1]]] # tokenize the active color
        token_vector+=[self.dictionary_board[" "]]
        token_vector+= [self.dictionary_castling_rights[char] for char in fen_components[2] ] # tokenize the castling rights
        token_vector+=[self.dictionary_board[" "]]
        token_vector+= [self.dictionary_en_passant_square[fen_components[3]]] # tokenize the en passant square
        token_vector+=[self.dictionary_board[" "]]
        token_vector+= [int(fen_components[4])+ 45] # tokenize the half move clock
        token_vector+=[self.dictionary_board[" "]]
        token_vector+= [int(fen_components[5])+ 45+100] # tokenize the full move number
        if(len(token_vector)<self.max_length):
            token_vector+=[self.dictionary_board["<PAD>"]]*(self.max_length-len(token_vector))
        else:
            print("bigger:", len(token_vector))
            
    
        return torch.LongTensor(token_vector)
    def detokenize(self,token_vector):
        fen=[]
        for token in token_vector:
            token=token.item()
            if token in range(1,9):
                fen.append(str(token))
            elif token > 44 and token<45+100:
                fen.append(str(token- 45))
            elif token >= 45+100 and token<self.dictionary_board[" "]:
                fen.append(str(token-(45+100)))
            else:
                fen.append(self.detokenizer_dict[token])
        return ''.join(fen).replace("<PAD>", "")

class Dataset(torch.utils.data.Dataset):

    def __init__(self, fens,legal_moves,target_moves,prev_moves,moves_dict):
        self.labels_dict=moves_dict
        self.labels = [self.labels_dict[label] for label in target_moves]
        self.legal_moves=[[self.labels_dict[move] for move in list_moves] for list_moves in legal_moves]
        self.positions = [tokenizer.tokenize(fen) for fen in fens]
        self.prev_moves=[self.labels_dict[label] for label in prev_moves]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_positions(self, idx):
        # Fetch a batch of inputs
        return self.positions[idx]
    def get_batch_legal_moves(self, idx):
        # Fetch a batch of inputs
        return self.legal_moves[idx]
    def get_batch_previous_moves(self, idx):
        # Fetch a batch of inputs
        return np.array(self.prev_moves[idx])

    def __getitem__(self, idx):

        batch_positions = self.get_batch_positions(idx)
        batch_y = self.get_batch_labels(idx)
        batch_input_tgt=torch.LongTensor(self.get_batch_legal_moves(idx))
        batch_prev_moves= self.get_batch_previous_moves(idx)
        return batch_positions, batch_prev_moves,batch_y

class PositionalEncoding(torch.nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:x.size(0),:], 
                         requires_grad=False)
        return self.dropout(x)

class TransformerClassifier(torch.nn.Module):

    def __init__(self, vocab_size_fens,n_moves, dropout=0.1, d_model=512, n_labels=5, nhead=8, num_encoder_layers=4, dim_feedforward=2048):

        super(TransformerClassifier, self).__init__()
        self.d_model=d_model
        self.embedding_fens = torch.nn.EmbeddingBag(vocab_size_fens, d_model)
        self.embedding_moves = torch.nn.EmbeddingBag(n_moves, d_model)
        self.pos_encoder_fens = PositionalEncoding(d_model, dropout)
        self.pos_encoder_moves = PositionalEncoding(d_model, dropout)
        self.transformer_model = torch.nn.Transformer(nhead=nhead, num_encoder_layers=num_encoder_layers
                                                ,num_decoder_layers=num_encoder_layers,dim_feedforward=dim_feedforward,
                                               d_model=d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(d_model, n_labels)
        self.relu = torch.nn.ReLU()

    def forward(self, fens_vector, mask_fens, moves_vector,mask_moves):
        embeded_fens=self.embedding_fens(fens_vector)* np.sqrt(self.d_model)
        fens_encoded=self.pos_encoder_fens(embeded_fens)
        embeded_moves=self.embedding_moves(moves_vector)* np.sqrt(self.d_model)
        moves_encoded=self.pos_encoder_moves(embeded_moves)
        
        
        pooled_output = self.transformer_model(src=fens_encoded,tgt=moves_encoded)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


if __name__=="__main__":
    main_folder = os.path.join(os.getcwd().split("RL_chess_engine")[0], "RL_chess_engine")
    games_folder= os.path.join(main_folder, "src", "games")
    path_real_games = os.path.join(games_folder, "real_games") 
    data_folder=os.path.join(main_folder, "src", "notebooks",'data_scientist','data')

    """ We retrieve the movements for the game"""
    MAX_LENGTH_VALID_MOVES=100
    N_GAMES=100
    if (os.path.exists(data_folder+'/'+str(N_GAMES)+'_games.csv')):
        data= pd.read_csv(data_folder+'/'+str(N_GAMES)+'_games.csv')
        FENS=data['fens'].to_list()
        TARGET_MOVES=data['target_moves'].to_list()
        LAST_MOVES=data['moves'].to_list()
        LEGAL_MOVES=data['LEGAL_MOVES'].map(literal_eval).to_list()
    else:
        FENS=[]
        TARGET_MOVES=[]
        LAST_MOVES=[]
        LEGAL_MOVES=[]
        stockfish = Stockfish()
        for (dirpath, dirnames, filenames) in os.walk(path_real_games):
            for file_path in tqdm(filenames[:N_GAMES]):
                with open(path_real_games+"/"+file_path) as file:
                    game = chess.pgn.read_game(file)
                    if game is None:
                        continue
                    for _ in game.mainline():
                        fen=game.board().fen()
                        FENS.append(fen)
                        stockfish.set_fen_position(fen) 
                        TARGET_MOVES.append(stockfish.get_best_move())
                        LEGAL_MOVES.append([move.uci() for move in game.board().legal_moves])
                        if game.move==None:
                            LAST_MOVES.append('start')
                        else:
                            LAST_MOVES.append(game.move.uci())
                        game=game.next()
        LEGAL_MOVES=[moves+['end']*(MAX_LENGTH_VALID_MOVES-len(moves)) for moves in LEGAL_MOVES]
        data_pd=pd.DataFrame({'fens':FENS,'moves':LAST_MOVES,'target_moves':TARGET_MOVES,'LEGAL_MOVES':LEGAL_MOVES})
        data_pd.to_csv(data_folder+'/'+str(N_GAMES)+'_games.csv')
    ## Compute the moves tokenizer

    moves_dict,inv_moves_dict = get_moves_dict()
    tokenizer=Tokenizer()

    ## we create the model 
    transformer=TransformerClassifier(vocab_size_fens=tokenizer.vocab_size,n_moves=len(moves_dict.keys()),
                                  d_model=(len(moves_dict.keys()) -2)//2, 
                                  n_labels=len(moves_dict.keys()), dim_feedforward=(len(moves_dict.keys())-2))

    
    ## Now we train our machine 
    EPOCHS = 1000
    LR = 1e-4
    MOMENTUM = 0.6
    fens_train, fens_val = np.split(FENS, [int(.8*len(FENS))])
    last_moves_train, last_moves_val = np.split(LAST_MOVES, [int(.8*len(LAST_MOVES))])
    target_moves_train, target_val = np.split(TARGET_MOVES, [int(.8*len(TARGET_MOVES))])
    legal_moves_train, legal_moves_val = np.split(LEGAL_MOVES, [int(.8*len(LEGAL_MOVES))])
    train, val = Dataset(fens_train,legal_moves_train,target_moves_train,last_moves_train,
                        moves_dict),Dataset(fens_val,legal_moves_val,target_val,last_moves_val,moves_dict)

    BATCH_SIZE=200
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    transformer.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr= LR)

    for epoch_num in range(EPOCHS):
        total_acc_train = 0
        total_loss_train = 0
        for train_input, target_input, train_label, in tqdm(train_dataloader):
            
            train_label = train_label.to(device)
            target_input =target_input.to(device)
            target_input =torch.reshape(target_input,(target_input.size()[0],1))
            mask_target = target_input.to(device)
            mask_input = train_input.to(device)
            input_id = train_input.squeeze(1).to(device)
            
            output=transformer(input_id , mask_input,target_input,mask_target)
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            transformer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
        total_acc_val = 0
        total_loss_val = 0
        with torch.no_grad():

            for val_input, target_input,val_label in val_dataloader:

                val_label = val_label.to(device)
                target_input =target_input.to(device)
                target_input =torch.reshape( target_input,(target_input.size()[0],1))
                mask_target = target_input.to(device)
                mask = val_input.to(device)
                input_id = val_input.squeeze(1).to(device)
                

                output= transformer(input_id , mask_input,target_input,mask_target)
                
                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train): .3f} \
            | Train Accuracy: {total_acc_train / len(train): .3f} \
            | Val Loss: {total_loss_val / len(val): .3f} \
            | Val Accuracy: {total_acc_val / len(val): .3f}')
    ##And we save the trained model  
    torch.save(transformer.state_dict(), '../../models/data_scientist/chess_transformer_v2.pth')


