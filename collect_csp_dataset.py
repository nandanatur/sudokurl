#!/usr/bin/env python3
"""Collect CSP dataset by solving multiple puzzles with the deterministic CSP solver.

Usage:
  python3 collect_csp_dataset.py [N] [optional_csv]

If an optional CSV is provided (Kaggle-style with puzzle strings), the script will use its first N puzzles.
Otherwise it will generate N random variants of the default `start` puzzle by randomly dropping givens.
"""
import sys
import os
import numpy as np
import sudoku

def puzzles_from_csv(path, start, end):
    puzzles = []
    with open(path, 'r') as f:
        lines = f.readlines()
    if not lines:
        return puzzles
    # try to detect header, take lines after header
    if ',' in lines[0]:
        lines = lines[1:]
    for line in lines[start:end+1]:
        line = line.strip().split(',')
        s = line[1]
        s = s.strip().strip('"')
        if len(s) >= 81:
            grid = np.zeros((9,9), dtype=np.int8)
            idx = 0
            for r in range(9):
                for c in range(9):
                    ch = s[idx]
                    grid[r,c] = 0 if ch == '.' else int(ch)
                    idx += 1
            puzzles.append(grid)
    return puzzles

def merge_npz_files(file_list, out_path):
    # load arrays and concatenate along axis 0 for each key
    imports = np.load(file_list[0])
    keys = list(imports.keys())
    accum = {k: [] for k in keys}
    for fp in file_list:
        data = np.load(fp)
        for k in keys:
            accum[k].append(data[k])
    merged = {}
    for k in keys:
        try:
            merged[k] = np.concatenate(accum[k], axis=0)
        except Exception:
            # fallback: stack then reshape
            merged[k] = np.concatenate([np.atleast_1d(x) for x in accum[k]], axis=0)
    np.savez_compressed(out_path, **merged)


def main():
    try:
        start = int(sys.argv[1])
        end = int(sys.argv[2])
    except:
        raise Exception("please provide start and end training sample index")

    csv_path = "sudoku_extreme_dataset/train.csv"
    if csv_path and os.path.exists(csv_path):
        puzzles = puzzles_from_csv(csv_path, start, end)
    else:
        raise Exception("dataset is not available")

    temp_files = []
    
    os.makedirs('snapshots/strategy3', exist_ok=True)
    for puzzle in puzzles:
        print(f"Solving puzzle {start}")
        sol = sudoku.solve_with_csp(puzzle)
        if sol is None:
            print(f"  No solution found for puzzle {start}")
        else:
            valid = sudoku.is_complete(sol) and sudoku.is_valid(sol)
            print(f"  Filled cells: {np.count_nonzero(sol)}, Valid: {valid}")

        # flush the per-puzzle dataset to a temp file
        tmp = os.path.join('snapshots/strategy3', f'dataset_csp_tmp_{start}.npz')
        sudoku.flush_datasets_now(tmp)
        temp_files.append(tmp)
        start += 1

    # merge all per-puzzle npz files into one final dataset
    out_path = os.path.join('snapshots/strategy3', f'dataset_csp_merged_{start-1}.npz')
    print('Merging', len(temp_files), 'files ->', out_path)
    merge_npz_files(temp_files, out_path)
    for item in temp_files:
        os.unlink(item)
    


if __name__ == '__main__':
    main()
