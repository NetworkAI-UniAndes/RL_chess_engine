import os
import sys
import fnmatch
import random
import chess.pgn
from pathlib import Path


class GameStarter:
    """This is the class in charge of accessing the local database of games.
    It also serves as an entry point to generate random chess positions
    from the local database."""

    def __init__(self) -> None:
        """On initialization the `GameStarter` object will just have a path
        to the folder inside the project where the games are stored. Also this
        will count the number of games per folder."""

        # Current working directory and then the path to the games DB folder
        self.main_folder = os.path.join(
            os.getcwd().split("RL_chess_engine")[0], "RL_chess_engine"
        )
        self.games_folder = os.path.join(self.main_folder, "src", "games")
        # Path to each type of games DB
        self.path_real_games = os.path.join(self.games_folder, "real_games")
        self.path_engines_games = os.path.join(self.games_folder, "engine_games")
        self.path_played_engine_games = os.path.join(
            self.games_folder, "played_engine_games"
        )
        # Number of games stored per each type of games DB.
        self.len_real_games = len(
            fnmatch.filter(os.listdir(self.path_real_games), "*.pgn")
        )
        self.len_engine_games = len(
            fnmatch.filter(os.listdir(self.path_engines_games), "*.pgn")
        )
        self.len_played_engine_games = len(
            fnmatch.filter(os.listdir(self.path_played_engine_games), "*.pgn")
        )

    def getRandomGame(self, game_type="real_games"):
        """This function returns a `game` object from the `python_chess` library.
        To get the PGN string from this object just call the `__str__()` method on the returned objected.
        If the FEN is required, call the `fen()` method on the returned object.
        `game_type` is the folder where which to retrieve the games. By default this will get the random games from the 'real_games'
        folder. If StockFish vs AlphaZero games are required this parameter should be equal to 'engine_games'.
        If games from our own engine are required `game_type` should be equal to 'played_engine_games`
        """
        folder_path, folder_len = self.path_real_games, self.len_real_games

        if game_type == "engine_games":
            folder_path = self.path_engines_games
            folder_len = self.len_engine_games
        elif game_type == "played_engine_games":
            folder_path = self.path_played_engine_games
            folder_len = self.len_played_engine_games

        file_path = os.path.join(
            folder_path, f"{random.randint(0, folder_len - 1)}.pgn"
        )
        with open(file_path) as file:
            game = chess.pgn.read_game(file)

        return game

    def getRandomPositionFromGame(self, game):
        """Given a game object from the `python_chess` library, this method will return a random
        game position from the mainline of the game. The return type of this function is a GameNode."""

        number_of_plays = sum(1 for _ in game.mainline())
        position_number = random.randint(0, number_of_plays)
        res = game
        for _ in range(position_number):
            res = res.next()
        return res

    def getRandomFenPosition(self, game_type="real_games"):
        """This function will return a random FEN position from the type of game specified.
        By default this will return a game from the `real_games` folder.
        If StockFish vs AlphaZero games are required this parameter should be equal to 'engine_games'.
        If games from our own engine are required `game_type` should be equal to 'played_engine_games`."""

        game = self.getRandomGame(game_type=game_type)
        random_position = self.getRandomPositionFromGame(game)
        return random_position.board().fen()
