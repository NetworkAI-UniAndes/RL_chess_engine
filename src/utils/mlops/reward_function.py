import numpy as np
import os 
from stockfish import Stockfish
import chess

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
class Scorer:
    '''
    Clase para evaluar una jugada a partir de dos tableros de ajedrez representados por
    arrays de numpy
    '''
    def __init__(self, pathStock:str, is_white=True, valor_mate=1, error_jugadada=-10):
        if pathStock is not None:
            self.stock = Stockfish(os.path.expanduser(path))
        else:
            self.stock = Stockfish()
        
        self.valor_mate = valor_mate
        self.error_jugada = error_jugadada
        self.ChessFormatConverter = ChessFormatConverter()

    def set_current_board(self,fen_pos):
        self.stock.set_fen_position(fen_pos)


    def get_move(self,fen_pos_initial,fen_pos_final):
        tab1=self.ChessFormatConverter.convert_fen_to_transformers(fen_pos_initial)
        tab2=self.ChessFormatConverter.convert_fen_to_transformers(fen_pos_final)
        abstab1 = np.zeros(tab1.shape)
        abstab2 = np.zeros(tab2.shape)
        # Si juegan las blancas se elimian las piezas negras
        # y si juegan las negras se eliminan las piezas blancas.
        if fen_pos_initial.split(" ")[1]=="w":
            for i in range(tab1.shape[0]):
                for j in range(tab1.shape[1]):
                    abstab1[i,j]=tab1[i,j] if tab1[i,j]>0 else 0
                    abstab2[i,j]=tab2[i,j] if tab2[i,j]>0 else 0
        else:
            for i in range(tab1.shape[0]):
                for j in range(tab1.shape[1]):
                    abstab1[i,j]=tab1[i,j] if tab1[i,j]<0 else 0
                    abstab2[i,j]=tab2[i,j] if tab2[i,j]<0 else 0
        abstab1 = np.abs(abstab1)
        abstab2 = np.abs(abstab2)

        # La resta de los tableros permite obtener la
        # casilla de origen y la casilla final
        res = abstab2-abstab1

        x1, y1 = np.where(res > 0)
        x0, y0 = np.where(res < 0)

        filas = "abcdefgh"
        # Se genera un map entre el nombre de las columnas
        # y el índice.
        num2pos = {i:j for i,j in zip(range(8),filas)}

        if len(x1)>1 and len(y1)>1:
            # Este caso se da cuando hay enroque.
            # Se busca la posición inicial y final del rey
            x1,y1 = np.where(abstab2==6)
            x0,y0 = np.where(abstab1==6)

        mov = f"{num2pos[y0[0]]}{abs(x0[0]-8)}{num2pos[y1[0]]}{abs(x1[0]-8)}"
        return [mov]

    @staticmethod
    def board_to_fen(tab1):
        num2piece = {-4.0: 'r',
                -3.0: 'n',
                -2.0: 'b',
                -5.0: 'q',
                -6.0: 'k',
                -1.0: 'p',
                1.0: 'P',
                4.0: 'R',
                3.0: 'N',
                2.0: 'B',
                5.0: 'Q',
                6.0: 'K'}
        fen = ""
        for i in range(8):
            curr = -1
            for j in range(8):
                if j > curr:
                    piece = tab1[i,j]
                    if piece == 0:
                        cont = 1
                        for k in range(j+1,8):
                            piece2 = tab1[i,k]
                            if piece2 == 0:
                                cont += 1
                                curr = k
                            else:
                                break
                        fen += str(cont)
                    else:
                        fen += num2piece[piece]
            fen += "/"
        fen = fen.strip("/")
        return fen


    def get_evaluation(self,fen_pos_initial,fen_pos_final):
        self.set_current_board(fen_pos_initial)
        move= self.get_move(fen_pos_initial,fen_pos_final)
        if self.stock.is_move_correct(move[0]):
            self.stock.make_moves_from_current_position(move)
            evalu = self.stock.get_evaluation()
            
            if evalu["type"] == "cp":
                sig = (1/(1+10**(-evalu["value"]/100/4))-0.5)*2
                val = sig
            else:
                if evalu["value"] == 0:
                    val = 0
                else:
                    val = ((1/evalu["value"])+0.01)-self.valor_mate if evalu["value"] < 0 else ((1/evalu["value"])-0.01)+self.valor_mate

            if fen_pos_initial.split(" ")[1]=="w":
                return val if val != 0 else self.valor_mate+1 , "legal move"
            else:
                return -val if val != 0 else self.valor_mate+1 , "legal move"

        else:
            return self.error_jugada, "invalid move"