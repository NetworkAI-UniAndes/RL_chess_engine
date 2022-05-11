import os
from flask import Flask,request
from dotenv import load_dotenv
from flask_mongoengine import MongoEngine

## Load enviroment variables
current_path = os.path.abspath(os.path.dirname(__file__))
dotenv_file = os.path.join(current_path, ".env")
load_dotenv(dotenv_path=dotenv_file)

## Video de referencia para crear esta API
#  https://www.youtube.com/watch?v=xxhRSWmsdVE&list=PLyHhqObotRTf23FuXMtOfJE5sp7OoUq-6

app = Flask(__name__)

# Flask-MongoEngine settings
MONGO_URI_BOOKODM = os.environ.get("MONGO_URI_BOOKODM")
app.config["MONGODB_SETTINGS"] = {
    'host': MONGO_URI_BOOKODM
}
db = MongoEngine(app)

## EP que determina que esta funcionando
@app.route("/")
def hello():
    return "<h1>Hello 👋</h1>"

## EP que determina que esta funcionando
@app.route("/main-engine")
def index():
    return "<h1>API is alive 🤖</h1>"

## EPs que determinan el CRUD de una entiedad de juego
@app.route("/main-engine/game", methods = ['POST'])
def createGame():
    print(request.json)
    return 'recieved'

@app.route("/main-engine/games", methods = ['GET'])
def getGames():
    return 'recieved'

@app.route("/main-engine/game/<id>", methods = ['GET'])
def getGame(id):
    print("Id recieved: ",id)
    return 'recieved'

@app.route("/main-engine/game", methods = ['UPDATE'])
def updateGame():
    return 'recieved'

@app.route("/main-engine/game", methods = ['DELETE'])
def deleteGame():
    return 'recieved'    

@app.route("/main-engine/game/movements/validate", methods = ['POST'])
def deleteGame():
    return 'recieved'    

if __name__ == "__main__":
    if os.environ.get("APPDEBUG") == "ON":
        app.run(host=os.environ.get("IP"),
                port=os.environ.get("PORT"), debug=True)
    else:
        app.run(host=os.environ.get("IP"),
                port=os.environ.get("PORT"), debug=False)
