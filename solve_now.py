import sys
from sudoku import solve_with_csp, start, end, is_valid, is_complete, plot_sudoku_solution
import numpy as np
import os

print('Running deterministic CSP solver...')
csv_path = 'dataset/train.csv'
if os.path.exists(csv_path):
    with open(csv_path, 'r') as f:
        data = f.readlines()
    data.pop(0)

    def get_sudoku_grid(sample):
        grid = [[0 for _ in range(9)] for _ in range(9)]
        index = 0
        for r in range(9):
            for c in range(9):
                if sample[index] == ".":
                    grid[r][c]=0
                else:
                    grid[r][c]=int(sample[index])
                index += 1
        return grid
    for item in data:
        line = item.strip().split(',')
        question = line[1]
        puzzle = np.array(get_sudoku_grid(question))
        sol = solve_with_csp(puzzle)
        if sol is None:
            print('No solution found for one sample')
        else:
            filled = np.count_nonzero(sol)
            solved_grid = is_complete(sol) and is_valid(sol)
            print('Filled cells:', filled, 'Valid solution:', solved_grid)
            np.savez_compressed('snapshots/csp_solution.npz', solution=sol, valid=bool(solved_grid))
            if solved_grid:
                plot_sudoku_solution(sol, filename='snapshots/csp_solution.png')
                print('Saved snapshots/csp_solution.png')
        # flush datasets from this process so CSP examples are persisted
        import sudoku
        sudoku.flush_datasets_now('snapshots/dataset_final_csp_flush.npz')
        import numpy as _np
        d = _np.load('snapshots/dataset_final_csp_flush.npz')
        print('dataset shapes after flush:', {k: v.shape for k, v in d.items()})
        counter += 1
else:
    # no CSV, run the default single puzzle `start`
    from sudoku import start as default_start
    sol = solve_with_csp(default_start)
    if sol is None:
        print('No solution found for default puzzle')
    else:
        filled = np.count_nonzero(sol)
        solved_grid = is_complete(sol) and is_valid(sol)
        print('Filled cells:', filled, 'Valid solution:', solved_grid)
        np.savez_compressed('snapshots/csp_solution.npz', solution=sol, valid=bool(solved_grid))
        if solved_grid:
            plot_sudoku_solution(sol, filename='snapshots/csp_solution.png')
            print('Saved snapshots/csp_solution.png')
    import sudoku
    sudoku.flush_datasets_now('snapshots/dataset_final_csp_flush.npz')
    import numpy as _np
    d = _np.load('snapshots/dataset_final_csp_flush.npz')
    print('dataset shapes after flush:', {k: v.shape for k, v in d.items()})
