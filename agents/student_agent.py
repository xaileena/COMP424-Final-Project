# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
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
    start_time = time.time()
    time_limit = 1.90

    valid_moves = self.get_moves(chess_board, player)

    if not valid_moves:
      return None

    best_move = None
    best_score = float('-inf')
    alpha = float('-inf')
    beta = float('inf')
    
    self.best_move_so_far = None
    self.best_score_so_far = float('-inf')

    for move in valid_moves:
      
      if time.time() - start_time > time_limit:
        break
      
      board_copy = np.copy(chess_board)
      execute_move(board_copy, move, player)
      
      score = self.minimax(board_copy, False, alpha, beta, player, opponent, 1, start_time, time_limit, move)
      
      if score > best_score:
        best_score = score
        best_move = move
      
      if score > self.best_score_so_far:
        self.best_score_so_far = score
        self.best_move_so_far = move
      
      alpha = max(alpha, best_score)
      
    time_taken = time.time() - start_time
    
    if time_taken > 2.0:
      print("WARNING: Move took too long. Time taken: {:.4f} seconds".format(time_taken))

    if self.best_move_so_far is not None:
      return self.best_move_so_far
    
    return best_move

  def get_moves(self, board, player):
    """
    Returns an ordered list of moves from most promising to least.
    """
    
    moves = get_valid_moves(board, player)
    result = {}
    
    for move in moves:
      num_discs_gained = count_disc_count_change(board, move, player)
      result[move] = num_discs_gained
    
    result = sorted(result.items(), key=lambda x: x[1], reverse=True)[:5]
    
    moves = []
    for move, _ in result:
      moves.append(move)
    
    return moves
  
  def minimax(self, board, is_maximizing, alpha, beta, player, opponent, depth, start_time, time_limit, root_move=None):
    """
    Alpha Beta Pruning algorithm implementation.
    """

    if time.time() - start_time > time_limit:
      if root_move is not None:
        score = self.get_scores(board, player, opponent)
        
        if score > self.best_score_so_far:
          self.best_score_so_far = score
          self.best_move_so_far = root_move
          
      return self.get_scores(board, player, opponent)

    is_endgame, _, _ = check_endgame(board)

    if is_endgame or depth == self.max_depth:
      return self.get_scores(board, player, opponent)

    cur_player = player if is_maximizing else opponent

    valid_moves = self.get_moves(board, cur_player)

    if not valid_moves:
      opponent_moves = self.get_moves(board, opponent if is_maximizing else player)
      
      if not opponent_moves:
        # both players have no moves, endgame
        return self.get_scores(board, player, opponent)
      
      # early termination, continue to next depth
      return self.minimax(board, not is_maximizing, alpha, beta, player, opponent, depth + 1, start_time, time_limit, root_move)

    if is_maximizing:
      max_score = float('-inf')
      
      for move in valid_moves:
        
        if time.time() - start_time > time_limit:
          break
        
        board_copy = np.copy(board)
        execute_move(board_copy, move, cur_player)
        
        score = self.minimax(board_copy, False, alpha, beta, player, opponent, depth+1, start_time, time_limit, root_move)
        
        max_score = max(max_score, score)
        alpha = max(alpha, max_score)
        
        if beta <= alpha:
          break
        
      return max_score
    
    else:
      min_score = float('inf')
      
      for move in valid_moves:
        
        if time.time() - start_time > time_limit:
          break
        
        board_copy = np.copy(board)
        execute_move(board_copy, move, cur_player)
        
        score = self.minimax(board_copy, True, alpha, beta, player, opponent, depth+1, start_time, time_limit, root_move)
        
        min_score = min(min_score, score)
        beta = min(beta, min_score)
        
        if beta <= alpha:
          break
        
      return min_score
    
  def get_scores(self, board, player, opponent):
      """
      Returns the score difference between the player and opponent.
      Amplify scores to favor winning moves.
      """
      
      num_player_discs = np.sum(board == player)
      num_opponent_discs = np.sum(board == opponent)
      
      if num_player_discs == 0:
        return -1000
      if num_opponent_discs == 0:
        return 1000
      
      num_moves_opponent = len(get_valid_moves(board, opponent))
      num_moves_player = len(get_valid_moves(board, player))
      
      if num_moves_opponent == 0 and num_moves_player > 0:
        return 500
      if num_moves_player == 0 and num_moves_opponent > 0:
        return -500

      move_diff = len(get_valid_moves(board, player)) - len(get_valid_moves(board, opponent))
      
      discs_diff = num_player_discs - num_opponent_discs
      
      center_points = self.central_control(board, player, opponent)
      
      game_progress = self.get_game_progress(board)
      
      if game_progress <= 0.5:
        return 2 * discs_diff + 6 * center_points + 4 * move_diff
      
      else:
        return 12 * discs_diff + 4 * move_diff
  
  
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
  
  def get_game_progress(self, board):
    """
    Get the game progress with 0 being the start
    and 1 being the end.
    """
    
    total_tiles = board.shape[0] * board.shape[1] - np.sum(board == 3)
    filled_tiles = np.sum((board == 1) + (board == 2))
    
    return filled_tiles / total_tiles if total_tiles > 0 else 0