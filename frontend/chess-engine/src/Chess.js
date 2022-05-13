// import { useState, React } from 'react';
// import { Chessboard } from "react-chessboard";
// import Chess from "chess.js";

// class ChessBoardComponent extends React.Component{
//      [game, setGame] = useState(new Chess());


//     safeGameMutate(modify) {
//             setGame((g) => {
//               const update = { ...g };
//               modify(update);
//               return update;
//             });
//           }

//     makeRandomMove() {
//             const possibleMoves = game.moves();
//             if (game.game_over() || game.in_draw() || possibleMoves.length === 0)
//               return; // exit if the game is over
//             const randomIndex = Math.floor(Math.random() * possibleMoves.length);
//             safeGameMutate((game) => {
//               game.move(possibleMoves[randomIndex]);
//             });
//           }

//     onDrop(sourceSquare, targetSquare) {
//         let move = null;
//         safeGameMutate((game) => {
//         move = game.move({
//             from: sourceSquare,
//             to: targetSquare,
//             promotion: "q", // always promote to a queen for example simplicity
//             });
//         });
//         if (move === null) return false; // illegal move
//         setTimeout(makeRandomMove, 200);
//         return true;
//     }

// 	render(){
//         const [game, setGame] = useState(new Chess());

// 		return(
// 			<div>
// 				<Chessboard position={game.fen()} onPieceDrop={onDrop} customDarkSquareStyle={{ backgroundColor: '#014B62' }} customLightSquareStyle={{ backgroundColor: '#11A1BB' }}/>
// 			</div>
// 		)
// 	}
// }

// export default ChessBoardComponent;