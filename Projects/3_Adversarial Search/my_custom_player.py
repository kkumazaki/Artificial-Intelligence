
from sample_players import DataPlayer
import logging
import pickle
import random
import math

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    # add init function
    def __init__(self, player_id):
      self.player_id = player_id
      self.timer = None
      self.queue = None
      self.context = None
      self.data = None


    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        #import random
        #self.queue.put(random.choice(state.actions()))

        if state.ply_count < 2:
          self.queue.put(random.choice(state.actions()))
        else:
          #self.queue.put(self.minimax(state, depth = 3)) # Try1: minimax (same as the opponent minimax)
          #self.queue.put(self.alpha_beta_search(state, depth = 3)) # Try2: alpha beta pruning (same depth with the opponent mimimax)
          self.queue.put(self.alpha_beta_search(state, depth = 4)) # Try3: alpha beta pruning (deeper depth than the opponent mimimax)
          #self.queue.put(self.alpha_beta_search(state, depth = 5)) # Try4: alpha beta pruning (deeper depth than the opponent mimimax) --> very slow...

    # add minimax function
    def minimax(self, state, depth):

        def min_value(state, depth):
          if state.terminal_test():
            return state.utility(self.player_id)
          if depth <= 0:
            return self.score(state)
          value = float("inf")
          for action in state.actions():
            value = min(value, max_value(state.result(action), depth-1))
          return value

        def max_value(state, depth):
          if state.terminal_test():
            return state.utility(self.player_id)
          if depth <= 0:
            return self.score(state)
          value  = float("-inf")
          for action in state.actions():
            value = max(value, min_value(state.result(action), depth-1))
          return value

        return max(state.actions(), key=lambda x: min_value(state.result(x), depth-1))

    # add alpha-beta pruning function
    def alpha_beta_search(self, state, depth):
        alpha = float("-inf")
        beta = float("inf")
        best_score = float("-inf")
        best_move = None

        def min_value(state, alpha, beta, depth):
          if state.terminal_test():
            return state.utility(self.player_id)
          if depth <= 0:
            #return self.score(state)
            return self.score3(state)
          value = float("inf")
          for action in state.actions():
            value = min(value, max_value(state.result(action), alpha, beta, depth-1))
            if value <= alpha:
              return value
            beta = min(beta, value)
          return value

        def max_value(state, alpha, beta, depth):
          if state.terminal_test():
            return state.utility(self.player_id)
          if depth <= 0:
            #return self.score(state)
            return self.score3(state)
          value  = float("-inf")
          for action in state.actions():
            value = max(value, min_value(state.result(action), alpha, beta, depth-1))
            if value >= beta:
              return value
            alpha = max(alpha, value)
          return value

        for a in state.actions():
          v = min_value(state.result(a), alpha, beta, depth)
          alpha = max(alpha, v)
          if v > best_score:
            best_score = v
            best_move = a
        return best_move

    # add score function
    def score(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        #return len(own_liberties) - len(opp_liberties) # heuristic 1 (original)
        #return len(own_liberties) - 2 * len(opp_liberties) # heuristic 2
        #return 2 * len(own_liberties) - len(opp_liberties) # heuristic 3
        #return len(own_liberties) - 1.5 * len(opp_liberties) # heuristic 4
        #return 1.5 * len(own_liberties) - len(opp_liberties) # heuristic 5
        #return len(own_liberties) - 1.2 * len(opp_liberties) # heuristic 6
        #return 1.2 * len(own_liberties) - len(opp_liberties) # heuristic 7
        return 1.1 * len(own_liberties) - len(opp_liberties) # heuristic 8

    def score2(self, state):
        _WIDTH = 11
        own_loc = state.locs[self.player_id]
        center_loc = 57
        x, y = own_loc % (_WIDTH + 2), own_loc // (_WIDTH + 2)
        cx, cy =  center_loc % (_WIDTH + 2), center_loc // (_WIDTH + 2)
        center_distance = math.sqrt((cx-x)*(cx-x) + (cy-y)*(cy-y)) 
        return  math.sqrt(41) - center_distance # how close from center point (farest point is 0) # heuristic 9

    def score3(self, state):
        _WIDTH = 11
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        center_loc = 57
        x, y = own_loc % (_WIDTH + 2), own_loc // (_WIDTH + 2)
        cx, cy =  center_loc % (_WIDTH + 2), center_loc // (_WIDTH + 2)
        center_distance = math.sqrt((cx-x)*(cx-x) + (cy-y)*(cy-y)) 
        #return 1*(len(own_liberties) - len(opp_liberties)) + 1*(math.sqrt(41) - center_distance) # combination # heuristic 10
        #return 2*(len(own_liberties) - len(opp_liberties)) + 1*(math.sqrt(41) - center_distance) # combination # heuristic 11
        return 1*(len(own_liberties) - len(opp_liberties)) + 2*(math.sqrt(41) - center_distance) # combination # heuristic 12
        #return 1*(len(own_liberties) - len(opp_liberties)) + 3*(math.sqrt(41) - center_distance) # combination # heuristic 13
        #return 1*(len(own_liberties) - len(opp_liberties)) + 1.5*(math.sqrt(41) - center_distance) # combination # heuristic 14