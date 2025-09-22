import numpy as np 
from .move2right import move2right
from typing import Optional

RNG = np.random.default_rng(42)

class Board:
    def __init__(self):
        self._init_board()

    @staticmethod
    def _generate_new_number() -> int:
        if RNG.random() >= 0.9:
            return 4
        else:
            return 2
        
    def _init_board(self) -> None:
        self._board = np.zeros(
            (4, 4),
            dtype=np.int32
        ) 
        idx_2 = RNG.integers(0, 4, size=2)
        self._board[idx_2[0], idx_2[1]] = 2

        flag = True
        while flag:
            idx_4 = RNG.integers(0, 4, size=2)

            if any(idx_2 != idx_4):
                flag = False 

        self._board[idx_4[0], idx_4[1]] = 4
    
    def is_win(self) -> bool:
        """
        True если мы победили и False иначе 
        """
        return np.any(self._board == 2048)
    
    @property
    def board(self):
        return self._board

    @board.setter
    def board(self, x: np.ndarray):
        assert isinstance(x, np.ndarray)
        assert x.dtype == np.int32
        assert x.shape == (4, 4)
        self._board = x

        
    def make_move(
            self, 
            dir: int,
        ) -> np.ndarray:
        """
        По соглашению 
        1 - Вправо 
        2 - Влево 
        3 - Вверх 
        4 - Вниз 

        Возвращает новую доску, новая доска устанавливается пользователем класса извне        
        """
        assert 1 <= dir <= 4

        if dir == 1:
            new_board = move2right(self._board)

        elif dir == 2:
            temp_matrix = np.rot90(self._board, -2)
            new_board = move2right(temp_matrix)
            new_board = np.rot90(new_board, 2)

        elif dir == 3:
            temp_matrix = np.rot90(self._board, -1)
            new_board = move2right(temp_matrix)
            new_board = np.rot90(new_board, 1)

        elif dir == 4:
            temp_matrix = np.rot90(self._board, 1)
            new_board = move2right(temp_matrix)
            new_board = np.rot90(new_board, -1)
        else:
            assert False

        zero_indices = np.argwhere(new_board == 0)
        
        if len(zero_indices) != 0:
            random_index = RNG.choice(zero_indices)
            new_board[tuple(random_index)] = self._generate_new_number()


        return new_board
        



        







