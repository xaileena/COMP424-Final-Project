# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves, count_disc_count_change


@register_agent("greedy_heuristic")
class StudentAgent(Agent):
    """
    A class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.max_depth = 4   

    def step(self, chess_board, player, opponent):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (board_size, board_size)
        where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
        and 2 represents Player 2's discs (Brown).
        - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
        - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

        You should return a tuple (r,c), where (r,c) is the position where your agent
        wants to place the next disc. Use functions in helpers to determine valid moves
        and more helpful tools.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        start = time.time()
        time_limit = 1.90   

        valid_moves = get_valid_moves(chess_board, player)
        if not valid_moves:
            return None

        order_moves = self.get_moves(chess_board, valid_moves, player)

        best_move = None
        best_score = float("-inf")

        for move in order_moves:
            if time.time() - start > time_limit:
                break

            board_copy = np.copy(chess_board)
            execute_move(board_copy, move, player)

            score = self.greedy_approach(board_copy, player, opponent, depth=1, start=start, time_limit=time_limit)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def get_moves(self, chess_board, moves, player):
        """
        Returns an ordered list of moves from most promising to least.
        A move is more promising if it you gain more disks and if it makes you go closer to the center
        """
        result = []
        for move in moves:
            num_discs_gained = count_disc_count_change(chess_board, move, player)
            center_score = 1 / (1 + self.manhattan_distance(chess_board, move))
            result.append((move, num_discs_gained + 0.5 * center_score))

        result.sort(key=lambda x: x[1], reverse=True)

        return [m for (m, _) in result]

    def manhattan_distance(self, chess_board, move):
        """
        Returns manhattan distance calculation of the disc
        """
        row, column = move.get_dest()
        size = chess_board.shape[0]
        center = size // 2
        return abs(row - center) + abs(column - center)

    def greedy_approach(self, chess_board, player, opponent, depth, start, time_limit):
        """
        We play a move, the opponent plays the best possible move, then evaluate based on heuristic
        """
        if time.time() - start > time_limit:
            return self.get_scores(chess_board, player, opponent)

        #Endgame
        game_ended, p1, p2 = check_endgame(chess_board)
        if game_ended:
            if p1 > p2:
                return 100
            else:
                return -100

        #Mac depth(4) reached
        if depth >= self.max_depth:
            return self.get_scores(chess_board, player, opponent)

        #Opponent moves
        opponent_moves = get_valid_moves(chess_board, opponent)
        if not opponent_moves:
            return self.get_scores(chess_board, player, opponent)

        minimum = float("inf")   

        #Opponent uses same heuristic 
        opponent_heuristic = self.get_moves(chess_board, opponent_moves, opponent)

        #Assume opponent will put us in worst position
        for move in opponent_heuristic:
            if time.time() - start > time_limit:
                break

            board_copy = np.copy(chess_board)
            execute_move(board_copy, move, opponent)

            score = self.get_scores(board_copy, player, opponent)
            if score < minimum:
                minimum = score

        return minimum


    def get_scores(self, chess_board, player, opponent):
        """
        Determine scores based on importance of heursitic
        """
        #Distribution of disks on the board
        my_discs = np.sum(chess_board == player)
        opponent_discs = np.sum(chess_board == opponent)
        disc_difference = my_discs - opponent_discs

        #Moves that can be played 
        my_moves = len(get_valid_moves(chess_board, player))
        opponent_moves = len(get_valid_moves(chess_board, opponent))
        movement = my_moves - opponent_moves

        #Closeness of disc to the center of the board
        closeness_to_center = self.close_to_center(chess_board, player, opponent)

        total_score = 3 * disc_difference + 2 * movement + 1 * closeness_to_center

        return total_score

    def close_to_center(self, chess_board, player, opponent):
        score = 0
        chess_board_size = chess_board.shape[0]
        center = chess_board_size / 2
    
        for row in range(chess_board_size):
            for column in range(chess_board_size):
                dist = abs(row - center) + abs(column - center)
                if chess_board[row, column] == player:
                    score += (chess_board_size - dist)
                elif chess_board[row, column] == opponent:
                    score -= (chess_board_size - dist)

        return score
