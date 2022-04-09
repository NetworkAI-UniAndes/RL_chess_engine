from stockfish import Stockfish
class Player:
    def __init__(self) -> None:
        self.stockfish = Stockfish()
    def play(self, fen_position):
        self.stockfish.set_fen_position(fen_position) 
        move=self.stockfish.get_best_move()
        self.stockfish.make_moves_from_current_position([move])
        return self.stockfish.get_fen_position()