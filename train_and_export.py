"""Train two independent tabular Q-learning tic-tac-toe agents by self-play.
Export learned action-value tables as JSON for a standalone browser game.
"""
import argparse
import json
import random
from collections import defaultdict, Counter
from pathlib import Path

LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

def legal(board):
    return [i for i, c in enumerate(board) if c == '.']

def result(board):
    for a,b,c in LINES:
        if board[a] != '.' and board[a] == board[b] == board[c]:
            return board[a]
    return 'Draw' if '.' not in board else None

def train(episodes=200_000, alpha=0.2, gamma=0.95, seed=42):
    rng = random.Random(seed)
    q = {p: defaultdict(lambda: [0.0]*9) for p in ('X','O')}
    counter = Counter()
    for ep in range(episodes):
        epsilon = max(0.05, 1.0 - 0.95*ep/episodes)
        board = ['.']*9
        pending = {'X': None, 'O': None}
        turn='X'
        for _ in range(9):
            s = ''.join(board)
            if pending[turn] is not None:
                old_s, old_action = pending[turn]
                q_old = q[turn][old_s][old_action]
                max_future = max(q[turn][s][j] for j in legal(board))
                q[turn][old_s][old_action] += alpha*(gamma*max_future-q_old)
                pending[turn] = None
            moves=legal(board)
            if rng.random()<epsilon:
                action = rng.choice(moves)
            else:
                best=max(q[turn][s][j] for j in moves)
                action=rng.choice([j for j in moves if abs(q[turn][s][j]-best)<1e-12])
            board[action]=turn
            pending[turn]=(s, action)
            outcome = result(board)
            if outcome is not None:
                for player in ('X','O'):
                    if pending[player] is not None:
                        old_s, old_action = pending[player]
                        reward=0 if outcome=='Draw' else (1 if player==outcome else -1)
                        value=q[player][old_s][old_action]
                        q[player][old_s][old_action] += alpha*(reward-value)
                counter[outcome] += 1
                break
            turn = 'O' if turn=='X' else 'X'
    # Make model compact: unplayed/zero valued moves omitted as zeros in arrays.
    model = {
        player: {state:[round(v,4) for v in values] for state,values in q[player].items()}
        for player in ('X','O')
    }
    print('training episodes:', episodes)
    print('result distribution:', dict(counter))
    print('states:', {p:len(model[p]) for p in model})
    return model

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=200_000)
    parser.add_argument('--output', type=Path, default=Path('tic_tac_toe_q_model.json'))
    args=parser.parse_args()
    data=train(episodes=args.episodes)
    args.output.write_text(json.dumps({'episodes':args.episodes,'seed':42,'alpha':0.2,'gamma':0.95,'q':data},separators=(',',':')), encoding='utf-8')
    print('saved model:',args.output, args.output.stat().st_size,'bytes')
