from stockfish import Stockfish
import chess
import numpy as np
import os

class ChessFormatConverter:
    def __init__(self,path=None):
        if path is not None:
            self.stockfish = Stockfish(os.path.expanduser(path))
        else:
            self.stockfish = Stockfish()
        self.replace_dict={".":0,"P":1,"N":3,"B":2,"R":4,"Q":5,"K":6,"p":-1,"n":-3,"b":-2,"r":-4,"q":-5,"k":-6}
        self.inv_replace_dict = {v: k for k, v in self.replace_dict.items()}
    def convert_fen_to_transformers(self, fen):
        board = chess.Board(fen)
        str_board=str(board)
        for key in self.replace_dict.keys():
            str_board=str_board.replace(key,str(self.replace_dict[key]))
        return np.array(str_board.split()).reshape(8,8).astype(int)
    def convert_transformers_to_fen(self,board_array):
        fen_board=""
        for i in range(8):
            for j in range(8):
                fen_board+=self.inv_replace_dict[board_array[i][j]]
            fen_board+="/"
        return fen_board[:-1]

if __name__== "__main__":
    converter = ChessFormatConverter()
    print(converter.convert_fen_to_transformers("rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"))
    print(converter.convert_transformers_to_fen(converter.convert_fen_to_transformers("rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2")))