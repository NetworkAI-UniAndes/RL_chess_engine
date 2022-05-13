import { useState, useEffect } from "react";
import Chess from "chess.js";
import { Chessboard } from "react-chessboard";
import Button from 'react-bootstrap/Button';
import axios from 'axios';
import './App.scss';

export default function PlayRandomMoveEngine() {
  const [game, setGame] = useState(new Chess());
  let currentMove = '';
  const body = { FEN: game.fen(), UCI: currentMove};

  useEffect(() => {
    // POST request using axios inside useEffect React hook
    axios.post('https://localhost:5000/main-engine/game/movements/validate', body)
        .then(response => console.log(response.data));

    // empty dependency array means this effect will only run once (like componentDidMount in classes)
    }, [currentMove]);

  function safeGameMutate(modify) {
    setGame((g) => {
      const update = { ...g };
      modify(update);
      return update;
    });
  }

  function makeRandomMove() {
    const possibleMoves = game.moves();
    if (game.game_over() || game.in_draw() || possibleMoves.length === 0)
      return; // exit if the game is over
    const randomIndex = Math.floor(Math.random() * possibleMoves.length);
    safeGameMutate((game) => {
      game.move(possibleMoves[randomIndex]);
    });
  }

  function onDrop(sourceSquare, targetSquare) {
    // eslint-disable-next-line no-const-assign
    currentMove = sourceSquare + targetSquare;
    console.log(currentMove, game.fen());
    let move = null;
    safeGameMutate((game) => {
      move = game.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: "q", // always promote to a queen for example simplicity
      });
    });
    if (move === null) return false; // illegal move
    setTimeout(makeRandomMove, 200);
    return true;
  }

  return <>
    <Chessboard 
        position={game.fen()} 
        onPieceDrop={onDrop} 
        customDarkSquareStyle={{ backgroundColor: '#014B62' }}
        customLightSquareStyle={{ backgroundColor: '#11A1BB' }}
        />
        <Button variant="primary">Button #1</Button>
    </>
  ;
}

// function App() {
  
//   return (
//     <div className="App">
//       <ChessBoardComponent/>
//    </div>

//   );
// }

// export default App;