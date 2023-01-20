import chess
import chess.pgn
import chess.engine
import chess.svg

from random import choice
from time import time
from math import log, sqrt, e, inf


class Node():
    def __init__(self):
        self.state = chess.Board()
        self.children = set()
        self.parent = None
        self.parent_visits = 0
        self.visits = 0
        self.score = 0


def uct(node, c=1.41):
    return node.score / (node.visits + (10**-10)) + c * (sqrt(log(node.parent_visits + e) / (node.visits + (10**-10))))


def rollout(node, score=1):
    board = node.state
    if board.is_game_over():
        if board.result() == '1-0':
            return (node, score)
        elif board.result() == '0-1':
            return (node, -score)
        else:
            return (node, score / 2)

    possible_moves = [node.state.san(move) for move in list(node.state.legal_moves)]
    for move in possible_moves:
        temp_state = chess.Board(node.state.fen())
        temp_state.push_san(move)
        child = Node()
        child.state = temp_state
        child.parent = node
        node.children.add(child)
        
    random_state = choice(list(node.children))
    return rollout(random_state)


def expansion(node, is_white):
    if len(node.children) == 0:
        return node
    
    if is_white:
        best_uct = -inf
        best_child = None
        for c in node.children:
            tmp = uct(c)
            if tmp > best_uct:
                best_uct = tmp
                best_child = c

        return expansion(best_child, 0)

    else:
        best_uct = inf
        best_child = None
        for c in node.children:
            tmp = uct(c)
            if tmp < best_uct:
                best_uct = tmp
                best_child = c

        return expansion(best_child, 1)


def backpropagation(node, score):
    node.visits += 1
    node.score += score
    while node.parent != None:
        node.parent_visits += 1
        node = node.parent
    return node


def find_next_move(node, is_over, is_white, steps=30):
    if is_over:
        return -1
    
    possible_moves = [node.state.san(move) for move in list(node.state.legal_moves)]
    map_state_moves = dict()

    for move in possible_moves:
        tmp_state = chess.Board(node.state.fen())
        tmp_state.push_san(move)
        child = Node()
        child.state = tmp_state
        child.parent = node
        node.children.add(child)
        map_state_moves[child] = move

    while steps > 0:
        if is_white:
            best_uct = -inf
            best_child = None
            for c in node.children:
                tmp = uct(c)
                if tmp > best_uct:
                    best_uct = tmp
                    best_child = c
                    
            expanded_state = expansion(best_child, 0)
            state, reward = rollout(expanded_state)
            node = backpropagation(state, reward)
            steps -= 1
        else:
            best_uct = inf
            best_child = None
            for c in node.children:
                tmp = uct(c)
                if tmp < best_uct:
                    best_uct = tmp
                    best_child = c

            expanded_state = expansion(best_child, 1)
            state, reward = rollout(expanded_state)
            node = backpropagation(state, reward)
            steps -= 1

    if is_white:
        best_uct = -inf
        best_move = ''
        for c in node.children:
            tmp = uct(c)
            if tmp > best_uct:
                best_uct = tmp
                best_move = map_state_moves[c]
                
        return best_move
    else:
        best_uct = inf
        best_move = ''
        for c in node.children:
            tmp = uct(c)
            if tmp < best_uct:
                best_uct = tmp
                best_move = map_state_moves[c]
                
        return best_move


def setGameHeader(property, value):
    game.headers[str(property)] = str(value)


board = chess.Board()
is_white = 1
pgn = []
game = chess.pgn.Game()
time_elapsed = 0
count = 0

while not board.is_game_over():
    possible_moves = [board.san(move) for move in list(board.legal_moves)]
    root = Node()
    start = time()
    root.state = board
    result = find_next_move(root, board.is_game_over(), is_white)
    time_elapsed += (time() - start)
    board.push_san(result)
    pgn.append(result)
    is_white ^= 1
    count += 1
    
    print("Move:", result)
    print(board)
    print()
    
print()
print("Average time for a move =", time_elapsed / count , "s")
print()
print(board)
print()
print(" ".join(pgn))
print()
print(board.result())
setGameHeader("Result", board.result())
print(game)
