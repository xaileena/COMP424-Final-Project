# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import execute_move, check_endgame, get_valid_moves, count_disc_count_change

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
    
    valid_moves = self.get_moves(chess_board, player)
    
    if not valid_moves:
      return None
    
    best_move = valid_moves[0]
    
    depth = 1
    while True:
      # break right before 2.0 seconds
      if time.time() - start_time > 1.98:
        break
      
      try:
      
        cur_best_move = None
        cur_best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for move in valid_moves:
          
          if time.time() - start_time > 1.98:
            break
          
          board_copy = deepcopy(chess_board)
          execute_move(board_copy, move, player)
          score = self.minimax(board_copy, False, alpha, beta, player, opponent, 1, depth, start_time)
          
          if score > cur_best_score:
            cur_best_score = score
            cur_best_move = move
          
          alpha = max(alpha, cur_best_score)
        
        if cur_best_move is not None:
          best_move = cur_best_move
        
        depth += 1
      
      except TimeoutError:
        break
      
    time_taken = time.time() - start_time
    print("Depth reached: {}, Time taken: {:.4f} seconds".format(depth, time_taken))
    if time_taken > 2.0:
      print("WARNING: Move took too long. Time taken: {:.4f} seconds".format(time_taken))

    return best_move
  
  def get_moves(self, board, player):
    """
    Returns an ordered list of moves from most promising to least.
    """
    
    moves = get_valid_moves(board, player)
    result = []
    
    for move in moves:
      num_discs_gained = count_disc_count_change(board, move, player)
      result.append((num_discs_gained, move))
    
    result.sort(reverse=True, key = lambda x: x[0])
    return [move for _, move in result]
  
  def minimax(self, board, is_maximizing, alpha, beta, player, opponent, depth, max_depth, start_time):
    """
    Alpha Beta Pruning algorithm implementation.
    """
    
    if time.time() - start_time > 1.98:
      raise TimeoutError("Took too long to search")
    
    is_endgame, _, _ = check_endgame(board)
    
    if is_endgame or depth >= max_depth:
      return self.get_scores(board, player, opponent)
    
    cur_player = player if is_maximizing else opponent
    
    valid_moves = get_valid_moves(board, cur_player)
    
    if not valid_moves:
      # early termination, continue to next depth
      return self.minimax(board, not is_maximizing, alpha, beta, player, opponent, depth + 1, max_depth, start_time)
    
    if is_maximizing:
      max_score = float('-inf')
      
      for move in valid_moves:
        board_copy = deepcopy(board)
        execute_move(board_copy, move, cur_player)
        score = self.minimax(board_copy, False, alpha, beta, player, opponent, depth+1, max_depth, start_time)
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
        score = self.minimax(board_copy, True, alpha, beta, player, opponent, depth+1, max_depth, start_time)
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
    - Consider tricking the opponent into making bad moves?
    - Corner the opponent by decreasing their legal moves?
    
    Draft 1: Value having more legal moves
    """
    
    num_player_discs = np.sum(board == player)
    num_opponent_discs = np.sum(board == opponent)
    
    if num_player_discs == 0:
      return -1000
    if num_opponent_discs == 0:
      return 1000
    
    num_moves_player = len(get_valid_moves(board, player))
    num_moves_opponent = len(get_valid_moves(board, opponent))
    move_diff = num_moves_player - num_moves_opponent
    
    discs_diff = num_player_discs - num_opponent_discs
    
    center_points = self.central_control(board, player, opponent)
    
    return 5 * discs_diff + 2 * center_points + move_diff
  
  
  def central_control(self, board, player, opponent):
    """
    With just the number of discs and moves, the agent plays too passively.
    The agent avoids conflict and duplicates in safe spaces (near edges and corners).
    
    Make the agent play more aggressive by taking advantage of the space in the center
    of the board.
    """
    
    board_size = board.shape[0]
    center = board_size // 2
    score = 0
    
    for row in range(board_size):
      for col in range(board_size):
        distance = abs(row - center) +  abs(col - center)
        if board[row, col] == player:
          score += (board_size - distance)
        elif board[row, col] == opponent:
          score -= (board_size - distance)
    
    return score