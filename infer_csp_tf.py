#!/usr/bin/env python3
"""Inference utility for the trained CSP TF model.

Usage examples:
  # predict on the first example inside a merged NPZ
  python3 infer_csp_tf.py --model snapshots/csp_tf_model --npz snapshots/dataset_csp_merged_10.npz --index 0 --topk 3

  # predict from an 81-char puzzle string (use . for empty)
  python3 infer_csp_tf.py --model snapshots/csp_tf_model --pstring "53..7....6..195....98....6.8...6...34..8.3..17...2...6....6..28....419..5....8..79"
"""
import argparse
import numpy as np
import os
import tensorflow as tf

def preprocess_single(grid):
    # grid: (9,9) ints 0..9
    X_chan = np.zeros((1, 9, 9, 10), dtype=np.float32)
    for i in range(1, 10):
        X_chan[0, :, :, i-1] = (grid == i).astype(np.float32)
    X_chan[0, :, :, 9] = (grid == 0).astype(np.float32)
    return X_chan


def parse_pstring(s):
    s = s.strip()
    if len(s) < 81:
        raise ValueError('puzzle string must be 81 characters')
    grid = np.zeros((9,9), dtype=np.int8)
    idx = 0
    for r in range(9):
        for c in range(9):
            ch = s[idx]
            grid[r,c] = 0 if ch == '.' else int(ch)
            idx += 1
    return grid


def show_topk_for_grid(model, grid, topk=3, save_filled=None):
    X = preprocess_single(grid)
    p_pred, v_pred = model.predict(X)
    p = p_pred[0]  # (9,9,9)
    print(f'Value head (solvability prob): {float(v_pred[0,0]):.4f}')
    empties = np.argwhere(grid == 0)
    if empties.size == 0:
        print('Grid already complete')
        return
    for (r, c) in empties:
        probs = p[r, c, :]
        order = np.argsort(probs)[::-1]
        tops = order[:topk]
        vals = [(int(v+1), float(probs[v])) for v in tops]
        print(f'Cell ({r},{c}) top-{topk}:', vals)

    if save_filled:
        # greedily fill each empty cell with top-1 prediction
        filled = grid.copy()
        for (r, c) in empties:
            v = int(np.argmax(p[r, c, :]) + 1)
            filled[r, c] = v
        try:
            import sudoku
            sudoku.plot_sudoku_solution(filled, filename=save_filled)
            print('Saved filled-grid image to', save_filled)
        except Exception:
            print('Could not import sudoku.plot_sudoku_solution; filled grid printed below:')
            print(filled)


def find_mrv_cell(grid):
    """Return the MRV cell (r,c) and its candidate domain list for the provided grid.
    If multiple cells tie, return the first encountered. If no empty cells, return None.
    """
    empties = np.argwhere(grid == 0)
    if empties.size == 0:
        return None
    best_cell = None
    best_domain = None
    best_size = 999
    for (r, c) in empties:
        used = set(grid[r, :]) | set(grid[:, c]) | set(grid[(r//3)*3:(r//3)*3+3, (c//3)*3:(c//3)*3+3].flatten())
        used.discard(0)
        domain = [v for v in range(1, 10) if v not in used]
        if len(domain) < best_size:
            best_size = len(domain)
            best_cell = (int(r), int(c))
            best_domain = domain
            if best_size == 1:
                break
    return best_cell, best_domain


def show_mrv_topk_for_grid(model, grid, topk=3, save_filled=None, iterative_fill=False, max_steps=81):
    """Show top-k predictions only for the MRV cell. If iterative_fill is True,
    greedily fill MRV cells with the model's top-1 prediction until solved or max_steps.
    """
    working = grid.copy()
    steps = 0
    while True:
        mrv = find_mrv_cell(working)
        if mrv is None:
            print('Grid complete')
            break
        (r, c), domain = mrv
        X = preprocess_single(working)
        p_pred, v_pred = model.predict(X)
        p = p_pred[0]
        print(f'Value head (solvability prob): {float(v_pred[0,0]):.4f}')
        probs = p[r, c, :]
        order = np.argsort(probs)[::-1]
        tops = order[:topk]
        vals = [(int(v+1), float(probs[v])) for v in tops]
        print(f'MRV cell ({r},{c}) domain={domain} top-{topk}:', vals)

        if not iterative_fill:
            # not filling, done
            if save_filled:
                try:
                    import sudoku
                    sudoku.plot_sudoku_solution(working, filename=save_filled)
                    print('Saved grid image to', save_filled)
                except Exception:
                    print('Could not save image; sudoku.plot_sudoku_solution not available')
            break

        # iterative greedy fill: pick top-1 prediction
        top1 = int(np.argmax(probs) + 1)
        print(f'Filling MRV ({r},{c}) with top-1 -> {top1}')
        working[r, c] = top1
        steps += 1
        if steps >= max_steps:
            print('Reached max iterative fill steps')
            break

    if iterative_fill and save_filled:
        try:
            import sudoku
            sudoku.plot_sudoku_solution(working, filename=save_filled)
            print('Saved filled-grid image to', save_filled)
        except Exception:
            print('Could not import sudoku.plot_sudoku_solution; final grid:')
            print(working)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to saved TF model (folder or file)')
    parser.add_argument('--npz', help='NPZ containing csp_states to pick an example from')
    parser.add_argument('--index', type=int, default=0, help='Index within NPZ csp_states')
    parser.add_argument('--pstring', help='81-char puzzle string using . for empty cells')
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--save-filled', help='If set, save greedily-filled grid image to this filename')
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)

    if args.npz:
        data = np.load(args.npz)
        if 'csp_states' not in data:
            raise SystemExit('NPZ does not contain csp_states key')
        arr = data['csp_states']
        if args.index < 0 or args.index >= arr.shape[0]:
            raise SystemExit('Index out of range')
        grid = arr[args.index]
    elif args.pstring:
        grid = parse_pstring(args.pstring)
    else:
        raise SystemExit('Provide either --npz or --pstring')

    # show_topk_for_grid(model, grid, topk=args.topk, save_filled=args.save_filled)
    show_mrv_topk_for_grid(model, grid, topk=3, save_filled="a.png", iterative_fill=True)


if __name__ == '__main__':
    main()

