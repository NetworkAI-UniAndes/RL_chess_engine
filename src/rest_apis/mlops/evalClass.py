import numpy as np
import stockfish

class evalMove:
    """
    Clase para evaluar una jugada a partir de dos tableros de ajedrez representados por
    arrays de numpy
    """
    stock = None
    

    def __init__(self, tab1: np.array, tab2: np.array, pathStock:str, is_white=True, valor_mate=1, error_jugadada=-10):
        self.tab1 = tab1
        self.tab2 = tab2
        self.stock = stockfish(path=pathStock)
        self.valor_mate = valor_mate
        self.error_jugada = error_jugadada
        self.is_white = is_white
        self.set_current_board()
        self.move = self.get_move()


    def set_current_board(self):
        fen_tab1 = self.board_to_fen(self.tab1)
        self.stock.set_fen_posisition(fen_tab1)

    
    def get_move(self):
        abstab1 = np.abs(self.tab1)
        abstab2 = np.abs(self.tab2)
        res = abstab2-abstab1

        x1, y1 = np.where(res > 0)
        x0, y0 = np.where(res < 0)

        filas = "abcdefgh"
        num2pos = {i:j for i,j in zip(range(8),filas)}

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


    def get_evaluation(self):
        if self.self.stock.is_move_correct(self.move):
            self.stock.make_moves_from_current_position(self.mov)
            evalu = self.stock.get_evaluation()
            
            if evalu["type"] == "cp":
                sig = (1/(1+10**(-evalu["value"]/100/4))-0.5)*2
                val = sig
            else:
                if evalu["value"] == 0:
                    val = 0
                else:
                    val = ((1/evalu["value"])+0.01)-self.valor_mate if evalu["value"] < 0 else ((1/evalu["value"])-0.01)+self.valor_mate

            if self.is_white:
                return val if val != 0 else self.valor_mate+1
            else:
                return -val if val != 0 else self.valor_mate+1

        else:
            return self.error_jugada


    


    

