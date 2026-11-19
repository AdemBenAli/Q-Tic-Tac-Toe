# Q / ARENA — Human vs Q-Learning AI

A standalone, offline, polished Tic-Tac-Toe game powered by two Q-learning agents trained against each other for 200,000 self-play games.

## Play

1. Double-click `index.html` to open it in your browser (Chrome, Edge, Firefox or Safari).
2. Choose to play as X (first) or O (second).
3. Select **Casual**, **Balanced**, or **Focused**, then click the board to move.
4. Inspect the **Inside the AI** panel to see whether the AI explored or exploited, its selected cell, and Q-values for legal cells.

No notebook, Python installation, server, account, or internet connection is required to **play**.

## How it works

- Two independent tabular Q-learning agents (one for X, another for O).
- Each agent learns from self-play, using terminal rewards: victory +1, draw 0, defeat -1.
- Training uses epsilon-greedy exploration decaying from 1.0 to 0.05.
- During play, the difficulty setting controls the *inference-time* exploration probability:
  - Casual: epsilon = 0.40
  - Balanced: epsilon = 0.15
  - Focused: epsilon = 0.00
- The Focused mode selects one of the actions with the maximum learned Q-value, breaking ties at random.
- **No heuristic win/block rules, minimax, or external API.**

The browser game includes the frozen trained model directly in the HTML. The separate `model.json` is provided for transparency and use in other projects.

## Re-train the model (optional)

Install Python 3.9+; the trainer uses only the standard library.

```bash
python train_and_export.py --episodes 200000 --output model.json
```

Re-training exports an independent `model.json`. The original `index.html` includes the already-trained Q-values; replacing them requires rebuilding or updating its embedded `TRAINED` object with the exported JSON.

## Notes

- The AI's move explanations are accurate descriptions of the epsilon-greedy action selection and learned Q-values, not symbolic tactical explanations like "fork" or "block".
- Q-learning can learn strong play but does **not** guarantee perfect minimax play, even in Focused mode.
- Learning occurs during the Python training process; browser matches use the frozen learned Q-tables (the AI does not update from your moves in the browser).
