from flask import Flask
from flask_restful import Resource, Api, reqparse
from stockfish import Stockfish
from os import path
import chess
import re


app = Flask(__name__)
api = Api(app)

class Eval(Resource):
    #Mejoras traducir el pgn a fen e inicializar el tablero y mirar la evaluación
    def get(self):
        stock = Stockfish(path=r"stockfish_14.1_win_x64_popcnt\stockfish_14.1_win_x64_popcnt.exe")
        parser = reqparse.RequestParser()
        parser.add_argument("fName",required=True)
        parser.add_argument("color",required=True)
        args = parser.parse_args()
        
        if path.exists(args["fName"]):
            with open(args["fName"]) as file:
                pgnF = file.read()
            #Reemplaza los los espacion antes de los numeros con un salto de linea
            val = re.sub("\s(?=\d{1,3}\.)","\n",pgnF)
            #Remueve el score final de la partida
            res = re.sub("  .*","",val)

            #Encuentran todos los pares de movidas realizados en la partida
            ls = re.findall("(?<=\.).{1,10}",res)

            #Se crea una lista con todos cada movimiento realizado
            pgn = []
            for elem in ls:
                for elem2 in elem.split(" "):
                    pgn.append(elem2)
            board = chess.Board()

            lan = [] 
            #Se transforman los movimientos de pgn a lan(long algebraic notation)
            for i in pgn:
                lan.append(str(board.push_san(i)))

            #Se realizan todos los movimientos en stockfish
            stock.make_moves_from_current_position(lan)
            
            #Se evalua la ultima jugada realizada
            evalM = stock.get_evaluation()

            return evalM
        else:
            return {"message":"El path no se encontro"},404
            
            """
            Se puede usar este codigo si se quiere todos la evaluacion de todos los movimientos
            lan = {}
            Se transforma cada movimienot de pgn as lan (long algebraic notation)
            for i,mpgn in enumerate(pgn):
                lan[i] = {str(board.push_san(mpgn)):mpgn}

            
            results={"move":[],"eval":[]}

            #
            for i in lan.values():
                for lanm,pgnm in i.items():
                    stock.make_moves_from_current_position(moves=[lanm])
                    eval = stock.get_evaluation()
                    results["move"].append(pgnm)
                    results["eval"].append(eval)
            """       

            


api.add_resource(Eval, "/eval")

if __name__ == "__main__":
    app.run()