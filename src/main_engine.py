from models.data_scientist.chess_transformer import ChessTransformer 
from utils.data_engineering.game_starter import GameStarter
from utils.mlops.reward_function import Scorer
from utils.stockfish.stockfish_engine import Player
import numpy as np



class Chess_Engine():
    def __init__(self) -> None:
        self.ChessTransformer=ChessTransformer()
        self.GameStarter= GameStarter()
        self.Scorer= Scorer(None)
        self.Player= Player()

    def train(self,games_to_play,type_of_game="real_games",verbose=False):

        for game in range(games_to_play):
            position=self.GameStarter.getRandomFenPosition(type_of_game) # Returns a position in the fen format and the color of the player 

            finished_game=0
            while not finished_game:
                list_of_moves=self.ChessTransformer.predict_move(position) # Returns a list of moves in the chess library format
                if("checkmate" in list_of_moves):
                    finished_game=1
                    break
                scores =[] # Returns a list of scores for the given moves
                validation=[] # Returns a list of validity for the given moves
                last_move=position
                for move in list_of_moves:
                    score, validity =self.Scorer.get_evaluation(last_move,move)
                    scores.append(score)
                    validation.append(validity)
                    last_move=move
                self.ChessTransformer.train(scores)
                if("invalid move" in validation):
                    finished_game=1
                position= self.Player.play(list_of_moves[0])
            if verbose:
                print(position)
                print("Game # {}, average score {}".format(game, np.mean(scores)))
        
        return None


    def play(self,color, position):

        move=self.ChessTransformer.play(color, position) # Returns a move in the chess library format

        return move 

if __name__=="__main__":
    chess_engine=Chess_Engine()
    chess_engine.train(games_to_play=1000,verbose=True)
        