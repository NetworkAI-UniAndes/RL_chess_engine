
class Tokenizer:
    def __init__(self,path=None):
        self.dictionary_board= {
            "/":0,
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
            "P":20
        }
        self.dictionary_active_color={
            "w":21,
            "b":22
        }
        self.dictionary_castling_rights={
            "K":23,
            "Q":24,
            "k":25,
            "q":26,
            "-":27
        }
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
        self.dictionary_half_move_clock={
        }
        self.dictionary_full_move_number={}

    def board_to_token_vector(self, board):
        print(board)
        #takes every charater on the board and returns a token vector
        return [ int(char) if char.isnumeric() else self.dictionary_board[char] for char in board ]



    def convert_fen_to_transformers(self, fen):
        token_vector =[]
        fen_components=fen.split(" ")

        token_vector+= self.board_to_token_vector(fen_components[0])
        token_vector+= [self.dictionary_active_color[fen_components[1]]] # tokenize the active color
        token_vector+= [self.dictionary_castling_rights[char] for char in fen_components[2] ] # tokenize the castling rights
        token_vector+= [self.dictionary_en_passant_square[fen_components[3]]] # tokenize the en passant square
        token_vector+= [int(fen_components[4])+ 44] # tokenize the half move clock
        token_vector+= [int(fen_components[5])+ 44+50] # tokenize the full move number

        return token_vector

if __name__=="__main__":
    tokenizer = Tokenizer()
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    print(tokenizer.convert_fen_to_transformers(fen))
