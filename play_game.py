from Board import Board
from Model import TorchModel
from typing import Optional
import numpy as np 
import torch
from ReplayBuffer import ReplayBuffer

def play_single_game(
    model: TorchModel,
    memory: Optional[ReplayBuffer] = None 
) -> int:
    
    model.eval()
    board = Board()
    continue_game = True

    while continue_game:
        state = board.board.copy()  # Текущее состояние

        with torch.no_grad():
            moves_dist = model(
                board.board[np.newaxis, :, :]
            )
            moves = (
                torch.argsort(
                    moves_dist, 
                    descending=True
                )
                .cpu()
                .numpy()
                .astype(np.int8)[0]
            )
        
        for i in moves:
            new_board = board.make_move(int(i + 1))
            if np.any(new_board != board.board):
                action = i
                next_state = new_board
                board.board = next_state  
                break

        if board.is_win():
            reward = 1  
            done = True

        elif all(
            np.all(board.board == board.make_move(i))
            for i in range(1, 5)
        ):
            reward = -1
            done = True 
            
        else:
            reward = 0  
            done = False
        
        if memory is not None:
            memory.add(state, action, reward, next_state, done)
        
        if done:
            continue_game = False
        

    model.train()

    return np.max(board.board)