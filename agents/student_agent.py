# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
from time import time
from random import choice, random, randrange
from math import sqrt, log
from copy import deepcopy
import numpy as np


class Node:
    """
    Nodes of a tree that will contain states of the game.
    
    Parameters:
    -----------
        state
            a given state of the game
        parent
            the parent node of the current node
        children
            a list of child nodes of the current node
    """
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        # always create an empty list of children
        self.children = []
        
        
    # set the state of a node
    def set_state(self, state):
        self.state = state
        
        
    # set the parent of a node
    def set_parent(self, parent):
        self.parent = parent
        
        
    # set the children of a node
    def set_children(self, children):
        self.children = children
        
        
    # add a child to the child nodes
    def add_child(self, child):
        self.children.append(child)
        
        
    # return a random child in the child nodes
    def get_random_child(self):
        return choice(self.children)
    
    
    # return the child node with the best score (for the final move)
    # in case of a tie, the last child with the max value is chosen
    def get_best_child(self):
        best_child = None
        best_score = -float('inf')
        
        for c in self.children:
            # the typical heuristic for final selection is the highest visit count
            #score = c.state.win_count
            #score = c.state.win_count + c.state.visit_count
            score = c.state.visit_count
            
            # update the child with the best score
            if score > best_score:
                best_score = score
                best_child = c
        
        return best_child
    

class Tree:
    """
    A tree whose nodes will contain states of the game.
    
    Parameters:
    -----------
        root
            the root node of the tree
    """
    def __init__(self, root):
        self.root = root
        
        
    # set the root of the tree
    def set_root(self, root):
        self.root = root
       
        
class State:
    """
    A current state of the game.
    Unfortunately, it required some of the functions from the original 'world.py'.
    
    Parameters:
        player_num
            the player who made the move in the given state
        visit_count
            the number of visits to the node with this state
        win_count
            the score generated from this state
        chess_board
            the board in the current state
        my_pos
            the position of the player (always the same)
        adv_pos
            the position of the adversary (always the same)
        max_step
            the maximum number of steps that the current player can take
        dir
            the direction in which the barrier was placed for the current move
        moves
            the set of possible moves for a single step (fixed)
        opposites
            the table of opposing directions (fixed)
        board_size
            the size of the board (fixed, for the current game)
        untried_actions
            the possible moves from the current state which have not been explored in the tree
    """
    def __init__(self, player_num, visit_count, win_count, chess_board, my_pos, adv_pos, max_step, dir=0):
        self.player_num = player_num
        self.visit_count = visit_count
        self.win_count = win_count
        self.chess_board = chess_board
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.max_step = max_step
        self.dir = dir
        
        # moves (up, right, down, left)
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        # opposite directions
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        # size of the board
        self.board_size = len(self.chess_board[0])
        self.untried_actions = []
        
    
    # equality operator
    def __eq__(self, o):
        if isinstance(o, State):
            return ((self.player_num == o.player_num)
                    and (self.visit_count == o.visit_count)
                    and (self.win_count == o.win_count)
                    and (np.array_equal(self.chess_board, o.chess_board))
                    and (self.my_pos == o.my_pos)
                    and (self.adv_pos == o.adv_pos)
                    and (self.max_step == o.max_step)
                    and (self.dir == o.dir))
        else:
            return False
        
        
    # ALL SETTER FUNCTIONS (some are not necessary)
    
    def set_player_num(self, player_num):
        self.player_num = player_num
        
        
    def set_visits(self, visit_count):
        self.visit_count = visit_count
        
        
    def set_score(self, win_count):
        self.win_count = win_count
        
        
    def set_board(self, chess_board):
        self.chess_board = chess_board
        
        
    def set_my_pos(self, my_pos):
        self.my_pos = my_pos
        
        
    def set_adv_pos(self, adv_pos):
        self.adv_pos = adv_pos
        
        
    def set_max_step(self, max_step):
        self.max_step = max_step
        
        
    def set_dir(self, dir):
        self.dir = dir
        
        
    def set_board_size(self, board_size):
        self.board_size = board_size
        
    
    def set_untried_actions(self, untried_actions):
        self.untried_actions = untried_actions
        
        
    # increment the visit count by 1
    # not necessary, but helps with identification
    def increment_visits(self):
        self.visit_count += 1
        
        
    # add a value to the win count
    # this can be adjusted to be other things, like how many squares you won
    def add_score(self, score):
        self.win_count += score
       
    
    # return the player number of the current adversary
    def get_opponent(self):
        return (1 - self.player_num) % 2
    
    
    # switch the current player to the current adversary
    def toggle_opponent(self):
        self.player_num = (1 - self.player_num) % 2
        
    """
    NOTE: The following functions are modified from 'world.py'.
    If there are more economical algorithms, feel free to put them here.
    """
    
    # simulate a random movement on the current player
    def random_walk(self, my_pos, adv_pos):
        # copy the original position
        ori_pos = deepcopy(my_pos)
        steps = np.random.randint(0, self.max_step + 1)
        
        for _ in range(steps):
            r, c = my_pos
            dir = np.random.randint(0, 4)
            m_r, m_c = self.moves[dir]
            # take one step
            my_pos = (r + m_r, c + m_c)

            k = 0
            # if i am hitting a barrier or on top of my adversary
            while self.chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                # the limit of adjustments (adjust this number and the one below for time)
                if k > 100:
                    break
                # find another direction to place a barrier
                dir = np.random.randint(0, 4)
                m_r, m_c = self.moves[dir]
                # find a new step to take
                my_pos = (r + m_r, c + m_c)

            # if we hit the limit, don't move
            if k > 100:
                my_pos = ori_pos
                break

        # place barrier
        dir = np.random.randint(0, 4)
        r, c = my_pos
        # if i am hitting a barrier
        while self.chess_board[r, c, dir]:
            # find another direction to place a barrier
            dir = np.random.randint(0, 4)

        return my_pos, dir
        
    
    # check if the game has reached a final state
    def check_endgame(self):
        # Union-Find
        father = dict()
        for r in range(self.board_size):
            for c in range(self.board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        # i barely understand any of this
        for r in range(self.board_size):
            for c in range(self.board_size):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if self.chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(self.board_size):
            for c in range(self.board_size):
                find((r, c))
        
        my_r = find(self.my_pos)
        adv_r = find(self.adv_pos)
        my_score = list(father.values()).count(my_r)
        adv_score = list(father.values()).count(adv_r)
        
        # the game has not finished
        if my_r == adv_r:
            return 2, my_score, adv_score

        # i have more squares than the adversary
        if my_score > adv_score:
            return 0, my_score, adv_score
        # i have less squares than the adversary
        elif my_score < adv_score:
            return 1, my_score, adv_score
        # i tied with the adversary
        else:
            return -1, my_score, adv_score
 
    
    # check if i have hit a boundary
    def check_boundary(self, pos):
        r, c = pos
        return 0 <= r < self.board_size and 0 <= c < self.board_size
       
    
    # check if the step i am taking is an acceptable step
    def check_valid_step(self, start_pos, end_pos, barrier_dir):
        # get the coordinates for the start and end position
        r, c = end_pos
        
        # reject the step if there is already a barrier placed
        if self.chess_board[r, c, barrier_dir]:
            return False
        # accept the step if there is no barrier placed and you did not move
        if np.array_equal(start_pos, end_pos):
            return True
        
        # get position of the adversary
        if self.player_num == 1:
            # the current player is 1, so the adversary in the next turn is 1
            adv_pos = self.adv_pos
        elif self.player_num == 0:
            # the current player is 0, so the adversary in the next turn is 0
            adv_pos = self.my_pos

        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == self.max_step:
                break
            for dir, move in enumerate(self.moves):
                if self.chess_board[r, c, dir]:
                    continue

                next_pos = cur_pos + move
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        # return whether or not 'end' can be reached from 'start'
        return is_reached


    # place a barrier
    def set_barrier(self, chess_board, r, c, dir):
        # set the barrier to True
        chess_board[r, c, dir] = True
        # set the opposite barrier to True
        move = self.moves[dir]
        chess_board[r + move[0], c + move[1], self.opposites[dir]] = True
  
    
    # return a list of all valid moves for the current positions
    def get_all_possible_moves(self):
        possible_moves = []
        
        if self.player_num == 1:
            # the current player is 1, so get starting position for 0
            start_array = np.asarray(self.my_pos)
        elif self.player_num == 0:
            # the current player is 0, so get starting position for 1
            start_array = np.asarray(self.adv_pos)
        
        for i in range(self.board_size + 1):
            for j in range(self.board_size + 1):
                # get all coords on the board
                new_pos = (i, j)
                # continue if the position does not hit a boundary
                if self.check_boundary(np.asarray(new_pos)):
                    # get all directions to place a barrier in, for all coords
                    for dir in range(4):
                        # one of the possible positions
                        end_array = np.asarray(new_pos)
                        # the position is accepted if the step is valid
                        if self.check_valid_step(start_array, end_array, dir):
                            possible_moves.append((i, j, dir))
               
        return possible_moves
    
    
    # return a list of states for all valid moves for the current position
    def get_all_possible_states(self):
        possible_states = []
        # get all valid moves for the current position
        possible_moves = self.get_all_possible_moves()
        
        # go over all moves
        for move in possible_moves:
            i, j, dir = move
            # create a new board which is the result of performing the move
            new_chess_board = deepcopy(self.chess_board)
            self.set_barrier(new_chess_board, i, j, dir)
            
            if self.player_num == 1:
                # the current player is 1, so the next state is a move by 0
                s = State(0, 0, 0, new_chess_board, (i, j), self.adv_pos, self.max_step, dir)
            elif self.player_num == 0:
                # the current player is 0, so the next state is a move by 1
                s = State(1, 0, 0, new_chess_board, self.my_pos, (i, j), self.max_step, dir)
                
            # extra check to prevent bad moves (auto loss) from being played (pruning)
            board_status, my_score, adv_score = s.check_endgame()
            # if the state is not an auto loss
            if board_status != 1:
                # add the new state to the list of possible states
                possible_states.append(s)

        return possible_states
    
    
    # return a list of all untried actions from the current state
    def get_untried_actions(self):
        # get all possible states
        possible_states = self.get_all_possible_states()
        self.set_untried_actions(possible_states)
    
      
    # make a random move
    def random_play(self):
        if self.player_num == 1:
            # the current player is 1, so 0 will make the next random move
            my_new_pos, dir = self.random_walk(self.my_pos, self.adv_pos)
        elif self.player_num == 0:
            # the current player is 0, so 1 will make the next random move
            my_new_pos, dir = self.random_walk(self.adv_pos, self.my_pos)
        i, j = my_new_pos
        
        # modify the chess board to reflect the move
        self.set_barrier(self.chess_board, i, j, dir)
        if self.player_num == 1:
            # update the move for 0
            self.my_pos = (i, j)
        elif self.player_num == 0:
            # update the move for 1
            self.adv_pos = (i, j)
        # update the direction (might be unnecessary?)
        self.dir = dir


class MonteCarloTreeSearch:
    """
    Where the magic happens.
    An instance of the entire search tree and search algorithm
    leading to a final outcome.
    """
    def __init__(self):
        pass
    
    
    class UCT:
        """
        Implementation of the upper confidence tree.
        """
        def __init__(self):
            pass
        
        
        # return the UCT value of a given node
        # the value 'c' is adjustable (default sqrt(2))
        def get_uct_value(self, total_visits, node_wins, node_visits, c=1.41):
            # node is not visited yet, prioritize the node
            if node_visits == 0:
                return 1000000

            return (node_wins / node_visits) + c * sqrt(log(total_visits) / node_visits)


        # find the node with the highest UCT value
        def find_best_node(self, node):
            best_child = None
            best_uct = -float('inf')
            # the visits to the current node
            parent_visits = node.state.visit_count
            
            for c in node.children:
                child_wins = c.state.win_count
                child_visits = c.state.visit_count
                # get the UCT value of all children of the current node
                uct = self.get_uct_value(parent_visits, child_wins, child_visits)
                
                # update the child with the highest UCT value
                if uct > best_uct:
                    best_uct = uct
                    best_child = c
            
            return best_child
            
    
    # select the leaf node from the root with the highest UCT value
    def select_promising_node(self, node):
        UCT = self.UCT()
        # while we have not reached a leaf node
        while len(node.children) != 0:
            # find the highest UCT value in the children
            node = UCT.find_best_node(node)
        
        return node
    
    
    # expand a node and its states
    def expand_node(self, node):
        # if there aren't any untried actions
        if len(node.state.untried_actions) == 0:
            # try to get all the untried actions
            node.state.get_untried_actions()
        
        # if there are untried actions
        if len(node.state.untried_actions) != 0:
            # pull out one of the untried actions
            state = node.state.untried_actions.pop()
            #state = node.state.untried_actions.pop(randrange(len(node.state.untried_actions)))
            # wrap the state in a node
            new_node = Node(state, node)
            # add the node to the children
            node.add_child(new_node)
      
        
    # propagate values up the tree
    def backpropagate(self, node, board_status, my_score=1, adv_score=1):
        # while we have not reached the root node
        while node is not None:
            # increment the visit count of the node
            node.state.increment_visits()
            
            if board_status == 1:
                # opponent won, add a penalty score or add nothing
                #node.state.add_score(-adv_score)
                pass
            elif board_status == 0:
                # player won, add a reward
                node.state.add_score(my_score)
            elif board_status == -1:
                # player tied, add a less strong reward
                node.state.add_score(my_score / 2)
            
            # move up the tree
            node = node.parent
    
    
    # simulate a full randomized game from the given state
    def rollout(self, node):
        # get a deep copy of the given node to run the simulation
        temp = deepcopy(node)
        temp_state = temp.state
        # check the final result of the initial game
        board_status, my_score, adv_score = temp_state.check_endgame()
        
        # if the opponent won, set a penalty score (unlikely to occur)
        if board_status == 1:
            node.state.set_score(-1000000)
            return board_status, my_score, adv_score
        
        # if the game is not finished, keep making moves
        while board_status == 2:
            # take one random step in the game
            temp_state.random_play()
            # change the player number
            # we do this after changing the board because the original player num was needed
            temp_state.toggle_opponent()
            # re-evaluate the result of the game at each step
            board_status, my_score, adv_score = temp_state.check_endgame()
        
        # the game ends with states 0, 1, or -1
        return board_status, my_score, adv_score
    
    
    def find_next_move(self, current_board, player_num, my_pos, adv_pos, max_step, run_length=10):
        """
        Given information on the state of the game,
        we run a randomized simulation and return the state
        with the suggested highest win rate.

        Parameters
        ----------
            current_board
                the current board state
            player_num
                the player who is currently being considered
            my_pos, adv_pos, max_step
                taken from the 'step' function in 'world.py'
            run_length
                how long the simulation can run for
        """
        # start the timer
        start = time()
        # get the number of the opponent, which is normally 1
        # we initialize as if the opponent has finished their move
        opponent = (1 - player_num) % 2
        
        # set up the root node of the tree
        # the root has the current board, current moves, and a default direction
        root_state = State(opponent, 0, 0, current_board, my_pos, adv_pos, max_step)
        # initialize a node without parents or children
        root_node = Node(root_state)
        # generate a tree with the root node
        tree = Tree(root_node)
        
        # generate all children of the root at the beginning
        root_states = tree.root.state.get_all_possible_states()
        
        # if there are no movable states from the root
        if len(root_states) == 0:
            # perform a random walk and return the state
            my_new_pos, dir = tree.root.state.random_walk(my_pos, adv_pos)
            r, c = my_new_pos
            tree.root.state.set_barrier(tree.root.state.chess_board, r, c, dir)
            err_state = State(0, 0, 0, tree.root.state.chess_board, my_new_pos, adv_pos, max_step, dir)
            return err_state
        
        # go through each child state of the root
        for state in root_states:
            new_node = Node(state, tree.root)
            board_status, my_score, adv_score = state.check_endgame()
            # aggressive heuristics, auto end the game if there is a win/tie
            if board_status == 0:
                return new_node.state
            elif board_status == -1:
                return new_node.state

            # add the child state to the children of the root
            tree.root.add_child(new_node)
            
        # keep iterating under the time limit
        while time() - start < run_length:
            """
            SELECTION
            """
            # occasionally re-select one of the root's children (epsilon-greedy)
            if random() < 0.05:
                root = tree.root.get_random_child()
                promising_node = self.select_promising_node(root)
            else:
                # select the leaf node with the highest UCT value
                promising_node = self.select_promising_node(tree.root)
            # check the outcome of the leaf node
            board_status, my_score, adv_score = promising_node.state.check_endgame()
            
            """
            EXPANSION
            """
            # if the game is not finished at the node
            if board_status == 2:
                # expand the current node
                self.expand_node(promising_node)
            
            """
            SIMULATION
            """
            # initially, the node is the "root" state for the simulation
            node_to_explore = promising_node
            if len(promising_node.children) > 0:
                # if there are children, choose one at random
                # otherwise, keep the current node
                node_to_explore = promising_node.get_random_child()
            
            # simulate a random game on the current state
            new_board_status, new_my_score, new_adv_score = self.rollout(node_to_explore)
            
            """
            UPDATE
            """
            # propagate the reward value up the tree
            #self.backpropagate(node_to_explore, new_board_status, new_my_score, new_adv_score)
            self.backpropagate(node_to_explore, new_board_status)
            
        # get the child node of the root with the best score
        winner_node = tree.root.get_best_child()
        
        #end = time()
        #print(end - start)
        return winner_node.state


@register_agent("student_agent")
class StudentAgent(Agent):
    def __init__(self):
        # inherit methods from superclass
        super(StudentAgent, self).__init__()
        # the current 'iteration' of the game
        self.turn = 0
        # agent name
        self.name = "ShallowBlue"
        # set autoplay
        self.autoplay = True
        # unnecessary?
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        # initialize the tree search
        self.MCTS = MonteCarloTreeSearch()


    def step(self, chess_board, my_pos, adv_pos, max_step):
        # always load the current agent as '0' (first player)
        init_player_num = 0
        
        # if it is the agent's first turn, run 30 seconds
        if self.turn == 0:
            state = self.MCTS.find_next_move(chess_board, init_player_num, my_pos, adv_pos, max_step, 30)
            self.turn += 1
        # if it is any turn after the first, run 2 seconds
        else:
            state = self.MCTS.find_next_move(chess_board, init_player_num, my_pos, adv_pos, max_step)
            self.turn += 1
        
        # send my_pos and dir as the final move
        return state.my_pos, state.dir
            