import re
from stockfish import Stockfish
import chess
class Player:
    def __init__(self) -> None:
        self.stockfish = Stockfish()
        self.board = chess.Board()
    def play(self, fen_position):
        self.stockfish.set_fen_position(fen_position) 
        move=self.stockfish.get_best_move()
        self.stockfish.make_moves_from_current_position([move])
        return self.stockfish.get_fen_position()
    def is_valid(self, fen_position):
        try:
            self.board.set_fen(fen_position)
            if self.board.is_valid():
                return True
            else:
                return False
        except:
            # Now the board is guaranteed to not have changed
            return False