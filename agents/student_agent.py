# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  
  To run:
  ./venv/bin/python simulator.py --player_1 student_agent --player_2 random_agent --display
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"
    # TODO: adjust max depth or implement iterative deepening
    self.max_depth = 3 

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
    start_time = time.time()
    
    valid_moves = get_valid_moves(chess_board, player)
    
    if not valid_moves:
      return None
    
    best_move = None
    best_score = float('-inf')
    alpha = float('-inf')
    beta = float('inf')
    
    for move in valid_moves:
      board_copy = deepcopy(chess_board)
      execute_move(board_copy, move, player)
      score = self.minimax(board_copy, False, alpha, beta, player, opponent, 1)
      
      if score > best_score:
        best_score = score
        best_move = move
      
      alpha = max(alpha, best_score)
      
    time_taken = time.time() - start_time
    if time_taken > 2.0:
      print("WARNING: Move took too long. Time taken: {:.4f} seconds".format(time_taken))

    # TODO: return the best move found after searching
    return best_move
  
  def minimax(self, board, is_maximizing, alpha, beta, player, opponent, depth):
    """
    Alpha Beta Pruning algorithm implementation.
    """
    
    is_endgame, p1_score, p2_score = check_endgame(board)
    
    if is_endgame or depth == self.max_depth:
      return self.get_scores(board, player, opponent)
    
    cur_player = player if is_maximizing else opponent
    
    valid_moves = get_valid_moves(board, cur_player)
    
    if not valid_moves:
      # early termination, continue to next depth
      return self.minimax(board, not is_maximizing, alpha, beta, player, opponent, depth + 1)
    
    if is_maximizing:
      max_score = float('-inf')
      
      for move in valid_moves:
        board_copy = deepcopy(board)
        execute_move(board_copy, move, cur_player)
        score = self.minimax(board_copy, False, alpha, beta, player, opponent, depth+1)
        max_score = max(max_score, score)
        alpha = max(alpha, max_score)
        
        if beta <= alpha:
          break
        
      return max_score
    
    else:
      min_score = float('inf')
      
      for move in valid_moves:
        board_copy = deepcopy(board)
        execute_move(board_copy, move, cur_player)
        score = self.minimax(board_copy, True, alpha, beta, player, opponent, depth+1)
        min_score = min(min_score, score)
        beta = min(beta, min_score)
        
        if beta <= alpha:
          break
        
      return min_score
    
  def get_scores(self, board, player, opponent):
    """
    Returns the score difference between the player and opponent.
    Amplify scores to favor winning moves.
    
    TODO: improve evaluation function with heuristics.
    """
    
    num_player_discs = np.sum(board == player)
    num_opponent_discs = np.sum(board == opponent)
    
    if num_player_discs == 0:
      return -1000
    if num_opponent_discs == 0:
      return 1000
    
    return num_player_discs - num_opponent_discs