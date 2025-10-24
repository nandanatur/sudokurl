import random
import time
import os
import sys
import glob
import random
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sudoku import start as ORIGINAL_START, reward_goal, is_complete, is_valid, get_possible_actions, get_sudoku_grid
from tensorflow.keras.callbacks import ModelCheckpoint


def _grid_to_onehot(grid):
    """Return (9,9,10) one-hot with channel 0 = empty, 1..9 values."""
    oh = np.zeros((9, 9, 10), dtype=np.float32)
    for r in range(9):
        for c in range(9):
            v = int(grid[r, c])
            oh[r, c, v] = 1.0
    return oh

def predict_best_action_from_model(model, grid, legal_only=True, return_q=False):
    """
    Given a trained model and a grid (9x9), return best action (r,c,v).
    If legal_only=True restricts choices to get_possible_actions(grid).
    """
    oh = _grid_to_onehot(grid).reshape(1, -1).astype(np.float32)
    preds = model.predict(oh, verbose=0)[0]  # flattened 729
    # unnormalize if training used normalization
    preds = preds * float(max(1.0, abs(reward_goal)))

    # reshape to (9,9,9)
    qvals = preds.reshape(9, 9, 9)  # channels value=1..9 -> index 0..8

    best = None
    best_val = -float('inf')
    possible = get_possible_actions(grid) if legal_only else [(r, c, v) for r in range(9) for c in range(9) for v in range(1,10)]

    for (r, c, v) in possible:
        val = qvals[r, c, v-1]
        if val > best_val:
            best_val = val
            best = (r, c, v)

    if return_q:
        return best, best_val
    return best


def fill_grid_greedy_with_model(model, initial_grid, max_steps=200):
    """
    Greedily apply model's best legal action until complete or stuck.
    Returns list of grids (states).
    """
    grid = initial_grid.copy().astype(int)
    path = [grid.copy()]
    for _ in range(max_steps):
        if is_complete(grid):
            break
        action = predict_best_action_from_model(model, grid, legal_only=True)
        if action is None:
            break
        r, c, v = action
        grid[r, c] = v
        # if invalid (shouldn't happen if we used legal_only) break
        if not is_valid(grid):
            grid[r, c] = 0
            break
        path.append(grid.copy())
    return path

if __name__ == "__main__":
    if len(sys.argv) == 3:
        try:
            start = int(sys.argv[1])
            if start < 0:
                raise Exception("Invalid start index")
            end = int(sys.argv[2])
            if end < 0:
                raise Exception("Invalid end index")
            samples = []
            counter = -1
            with open("sudoku_extreme_dataset/train.csv") as fh:
                for line in fh:
                    if counter >= start and counter <= end:
                        question = line.split(",")[1]
                        grid = np.array(get_sudoku_grid(question))
                        samples.append((counter, grid))
                    elif counter > end:
                        break
                    counter += 1
                else:
                    raise Exception("Invalid sample number")

            model = tf.keras.models.load_model("sudoku_mlp.keras")
            for counter, item in samples:
                start = time.time()
                print(f"Sample: {counter}")
                print("Question:")
                print(item)
                path = fill_grid_greedy_with_model(model, grid, max_steps=200)
                print("Solution:")
                print(path[-1])
                num_zeros = np.count_nonzero(path[-1] == 0)
                print(f"Number of cells unfilled: {num_zeros}")
                print(f"Accuracy: {((81-num_zeros)/81)*100}")
        except:
            raise
    else:
        print("Please provide 2 arguments to this program which is start index and end index to train.csv sample")

