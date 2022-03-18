import { useState } from "react";
import Chess from "chess.js";
import { Chessboard } from "react-chessboard";
import Button from 'react-bootstrap/Button';
import './App.scss';

export default function PlayRandomMoveEngine() {
  const [game, setGame] = useState(new Chess());

  const [show, setShow] = useState(false);

  var kind=false;

  const handleClose = () => setShow(false);

  function safeGameMutate(modify) {
    setGame((g) => {
      const update = { ...g };
      modify(update);
      return update;
    });
  }

  function clickpve(){
    kind=true;
    handleClose;
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
  
    <Modal.Dialog show={show} onHide={handleClose}>
      <Modal.Header closeButton>
        <Modal.Title>Choose mode</Modal.Title>
      </Modal.Header>

      <Modal.Body>
        <Button onClick={clickpve}>Player vs Computer</Button>
        <Button onClick={handleClose}>Player vs Player</Button>
      </Modal.Body>
    </Modal.Dialog>
    
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