import os
import numpy as np
import matplotlib.pyplot as plt
reward_cp_complete = 10
reward_goal = 500
reward_fire = -100
reward_stuck = -10
reward_step = -1

# action is about filling in a cell
# if > 0 options: get all possible options and choose one at random
# if = 0 options: set the grid back to a previous state and recurse
actions = []
Q = np.zeros((10, 10, 10))
Q += 0.01
rewards_all_episodes = []
DEBUG = True

snapshots_dir = 'snapshots'
os.makedirs(snapshots_dir, exist_ok=True)
save_every = 100   # save snapshots every N episodes
accumulate_dataset = True
dataset_accumulate_every = 500   # flush accumulated dataset to disk every N episodes

# optional accumulators for building a training set (state -> Q)
dataset_states = []
dataset_Qs = []

# New accumulators for supervised training targets derived from MRV/search
dataset_policy_states = []    # list of 9x9 grids (int8)
dataset_policy_targets = []   # list of 9x9x9 float32 (probability over actions)
dataset_value_states = []
dataset_value_targets = []

# CSP-specific dataset accumulators (state -> action chosen by CSP)
dataset_csp_states = []       # list of 9x9 grids (int8)
dataset_csp_policy = []       # list of 9x9x9 one-hot action (float32)
dataset_csp_value = []        # list of scalar values (float32), e.g., 1.0 for successful nodes


# default puzzle (classic easy example). 0 denotes empty cells.
start = np.array([
    [5,3,0,0,7,0,0,0,0],
    [6,0,0,1,9,5,0,0,0],
    [0,9,8,0,0,0,0,6,0],
    [8,0,0,0,6,0,0,0,3],
    [4,0,0,8,0,3,0,0,1],
    [7,0,0,0,2,0,0,0,6],
    [0,6,0,0,0,0,2,8,0],
    [0,0,0,4,1,9,0,0,5],
    [0,0,0,0,8,0,0,7,9]
], dtype=np.int8)

# Training hyperparameters (defaults used when running main())
alpha = 0.1
gamma = 0.99
epsilon = 0.6
epsilon_decay = 0.995
num_episodes = 1000
max_steps_per_episode = 200


def is_valid(pos):
    r, c = pos
    if r < 0 or r >= maze.shape[0]:
        return False
    if c < 0 or c >= maze.shape[1]:
        return False
    if maze[r, c] == 1:
        return False
    return True

def choose_action(actions):
    """
    Epsilon-greedy selection over a list of (r, c, v) actions.
    Returns a chosen (r, c, v) tuple.
    """
    if len(actions) == 0:
        raise ValueError("choose_action called with empty actions list")

    # exploration
    if np.random.random() < epsilon:
        return actions[np.random.randint(len(actions))]

    # exploitation: pick action with highest Q value (break ties randomly)
    values = [Q[a[0], a[1], a[2]] for a in actions]
    max_val = max(values)
    candidates = [i for i, val in enumerate(values) if val == max_val]
    chosen_idx = np.random.choice(candidates)
    return actions[chosen_idx]

def plot_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Total Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig('rewards.png')

def plot_sudoku_solution(path_or_grid, filename='sudoku_solution.png'):
    """
    Render a solved (or partially solved) Sudoku grid.
    - Accepts either a single 9x9 numpy array or a list of grids (will use last).
    - Saves an image file (filename).
    """
    if isinstance(path_or_grid, list):
        grid = np.array(path_or_grid[-1])
    else:
        grid = np.array(path_or_grid)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')

    # draw cell backgrounds and numbers
    for r in range(9):
        for c in range(9):
            val = int(grid[r, c])
            given = bool(start[r, c] != 0)  # original puzzle cells
            # background color: light gray for given cells, white otherwise
            face = '#f2f2f2' if given else 'white'
            ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor=face, edgecolor='none'))
            if val != 0:
                color = 'black' if given else '#1f77b4'
                weight = 'bold' if given else 'normal'
                ax.text(c + 0.5, r + 0.5, str(val), ha='center', va='center',
                        fontsize=18, color=color, fontweight=weight)

    # thick lines for 3x3 boxes, thinner for cell borders
    for i in range(10):
        lw = 2.5 if i % 3 == 0 else 0.8
        ax.plot([0, 9], [i, i], color='k', linewidth=lw)
        ax.plot([i, i], [0, 9], color='k', linewidth=lw)

    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Sudoku — final grid', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close(fig)


def get_optimal_path(Q, start, goal=None, actions=None, maze=None, max_steps=200):
    """
    Flexible extractor of a greedy "optimal" path from Q.
    - If `start` is a 9x9 numpy grid this will greedily apply the highest-Q
      (r,c,v) actions until the grid is complete or no valid actions remain.
      Returns a list of grid states (copies) representing the fill sequence.
    - Otherwise behaves as a simple maze-style path finder when `maze`,
      `actions` and `goal` are provided: returns list of (r, c) positions.
    """
    # Sudoku-style path (start is a 9x9 grid)
    if isinstance(start, np.ndarray) and start.shape == (9, 9):
        grid = start.copy().astype(int)
        path = [grid.copy()]
        for _ in range(max_steps):
            if is_complete(grid):
                break
            possible = get_possible_actions(grid)
            if not possible:
                break
            values = [Q[a[0], a[1], a[2]] for a in possible]
            max_val = max(values)
            candidates = [i for i, v in enumerate(values) if v == max_val]
            idx = np.random.choice(candidates)
            r, c, v = possible[idx]
            grid[r, c] = v
            path.append(grid.copy())
        return path

    # Maze-style fallback (keep original behaviour if maze/actions provided)
    if maze is None or actions is None or goal is None:
        return []

    path = [start]
    state = start
    visited = set()

    for _ in range(max_steps):
        if state == goal:
            break
        visited.add(state)

        best_action = None
        best_value = -float('inf')

        for idx, move in enumerate(actions):
            next_state = (state[0] + move[0], state[1] + move[1])

            if (0 <= next_state[0] < maze.shape[0] and
                0 <= next_state[1] < maze.shape[1] and
                maze[next_state] == 0 and
                next_state not in visited):

                if Q[state][idx] > best_value:
                    best_value = Q[state][idx]
                    best_action = idx

        if best_action is None:
            break

        move = actions[best_action]
        state = (state[0] + move[0], state[1] + move[1])
        path.append(state)

    return path

def is_complete(grid):
    return np.all(grid != 0)

def is_row_complete(grid, row_idx):
    row = grid[row_idx]
    if np.any(row == 0):
        return False
    return set(row.tolist()) == set(range(1, 10))

def is_col_complete(grid, col_idx):
    col = grid[:, col_idx]
    if np.any(col == 0):
        return False
    return set(col.tolist()) == set(range(1, 10))

def is_box_complete(grid, row_idx, col_idx):
    r0 = (row_idx // 3) * 3
    c0 = (col_idx // 3) * 3
    box = grid[r0:r0+3, c0:c0+3].flatten()
    if np.any(box == 0):
        return False
    return set(box.tolist()) == set(range(1, 10))

def is_box_complete(grid, row_idx, col_idx):
    r0 = (row_idx // 3) * 3
    c0 = (col_idx // 3) * 3
    box = grid[r0:r0+3, c0:c0+3].flatten()
    if np.any(box == 0):
        return False
    return set(box.tolist()) == set(range(1, 10))

def is_valid(grid):
    # rows
    for i in range(9):
        vals = grid[i, :][grid[i, :] != 0]
        if vals.size != np.unique(vals).size:
            return False

    # columns
    for j in range(9):
        vals = grid[:, j][grid[:, j] != 0]
        if vals.size != np.unique(vals).size:
            return False

    # 3x3 boxes
    for r0 in (0, 3, 6):
        for c0 in (0, 3, 6):
            box = grid[r0:r0+3, c0:c0+3].flatten()
            vals = box[box != 0]
            if vals.size != np.unique(vals).size:
                return False

    return True

def get_possible_actions(grid):
    arr = grid.astype(int)
    actions = []
    for r in range(9):
        for c in range(9):
            if arr[r, c] != 0:
                continue
            # values present in row, column and box
            used = set(arr[r, :]) | set(arr[:, c]) | set(arr[(r//3)*3:(r//3)*3+3, (c//3)*3:(c//3)*3+3].flatten())
            used.discard(0)
            for v in range(1, 10):
                if v not in used:
                    actions.append((r, c, v))
    # score actions by how "dense" the affected row/col/box already are
    def _density_score(action):
        r, c, v = action
        row_filled = np.count_nonzero(arr[r, :])
        col_filled = np.count_nonzero(arr[:, c])
        r0 = (r // 3) * 3
        c0 = (c // 3) * 3
        box_filled = np.count_nonzero(arr[r0:r0+3, c0:c0+3])
        return row_filled + col_filled + box_filled

    # randomize order first so equal scores get random tie-breaking, then sort by density desc
    np.random.shuffle(actions)
    actions.sort(key=_density_score, reverse=True)
    return actions

def run_episodes(start, num_episodes, max_steps_per_episode):
    global Q, epsilon, rewards_all_episodes, dataset_states, dataset_Qs 

    if not is_valid(start):
        raise ValueError("start grid is not a valid partial Sudoku (contains duplicates). Fix the puzzle before training.")

    current_max_reward = -float('inf')
    for episode in range(num_episodes):
        state = start.copy()
        total_rewards = 0
        episode_states = [state.copy()]   # record initial state
        done = False

        for step in range(max_steps_per_episode):
            # use MRV+LCV with AC-3 pruning as the primary selector
            actions = get_possible_actions_mrv(state)

            # if no legal moves -> small penalty and end episode
            if len(actions) == 0:
                total_rewards += reward_stuck
                break

            x, y, v = choose_action(actions)
            next_state = state.copy()
            next_state[x, y] = v

            # invalid placement: penalize, update Q, but DON'T end episode
            if not is_valid(next_state):
                reward = reward_fire
                old_value = Q[x, y, v]
                next_max = 0
                Q[x, y, v] = old_value + alpha * (reward + gamma * next_max - old_value)
                total_rewards += reward
                # revert change (do not apply invalid move)
                next_state = state.copy()
                # record (state unchanged)
                episode_states.append(next_state.copy())
                continue

            # valid placement: check progress / completion
            completed_row = is_row_complete(next_state, x)
            completed_col = is_col_complete(next_state, y)
            completed_box = is_box_complete(next_state, x, y)

            if is_complete(next_state):
                reward = reward_goal
                old_value = Q[x, y, v]
                next_max = 0
                Q[x, y, v] = old_value + alpha * (reward + gamma * next_max - old_value)
                total_rewards += reward
                state = next_state
                episode_states.append(state.copy())
                done = True
                break

            # valid step, give base step reward and bonus for any completed unit
            reward = reward_step
            if completed_row:
                reward += reward_cp_complete
            if completed_col:
                reward += reward_cp_complete
            if completed_box:
                reward += reward_cp_complete

            # next_max: max over possible next actions from next_state
            # compute next_actions using MRV + AC-3 pruning
            next_actions = []
            # build domains for next_state
            arr_ns = next_state.astype(int)
            domains_ns = {}
            for rr in range(9):
                for cc in range(9):
                    if arr_ns[rr, cc] != 0:
                        continue
                    used = set(arr_ns[rr, :]) | set(arr_ns[:, cc]) | set(arr_ns[(rr//3)*3:(rr//3)*3+3, (cc//3)*3:(cc//3)*3+3].flatten())
                    used.discard(0)
                    domains_ns[(rr, cc)] = [v for v in range(1, 10) if v not in used]
            if domains_ns and ac3_prune(domains_ns):
                # pick MRV cell and produce actions
                # tie-break by degree as in get_possible_actions_mrv
                min_size = min(len(d) for d in domains_ns.values())
                mrv_cells_ns = [cell for cell, d in domains_ns.items() if len(d) == min_size]
                def degree_ns(cell):
                    r, c = cell
                    neigh = set()
                    for cc in range(9):
                        if cc != c and arr_ns[r, cc] == 0:
                            neigh.add((r, cc))
                    for rr in range(9):
                        if rr != r and arr_ns[rr, c] == 0:
                            neigh.add((rr, c))
                    r0 = (r // 3) * 3
                    c0 = (c // 3) * 3
                    for rr in range(r0, r0+3):
                        for cc in range(c0, c0+3):
                            if (rr, cc) != (r, c) and arr_ns[rr, cc] == 0:
                                neigh.add((rr, cc))
                    return len(neigh)
                best_cell_ns = max(mrv_cells_ns, key=degree_ns)
                rns, cns = best_cell_ns
                # order by LCV (least constraining) using the same logic
                def lcv_local(v):
                    removed = 0
                    for cc in range(9):
                        if cc == cns or arr_ns[rns, cc] != 0:
                            continue
                        used = set(arr_ns[rns, :]) | set(arr_ns[:, cc]) | set(arr_ns[(rns//3)*3:(rns//3)*3+3, (cc//3)*3:(cc//3)*3+3].flatten())
                        if v not in used:
                            removed += 1
                    for rr in range(9):
                        if rr == rns or arr_ns[rr, cns] != 0:
                            continue
                        used = set(arr_ns[rr, :]) | set(arr_ns[:, cns]) | set(arr_ns[(rr//3)*3:(rr//3)*3+3, (cns//3)*3:(cns//3)*3+3].flatten())
                        if v not in used:
                            removed += 1
                    r0 = (rns // 3) * 3
                    c0 = (cns // 3) * 3
                    for rr in range(r0, r0+3):
                        for cc in range(c0, c0+3):
                            if (rr, cc) == (rns, cns) or arr_ns[rr, cc] != 0:
                                continue
                            used = set(arr_ns[rr, :]) | set(arr_ns[:, cc]) | set(arr_ns[(rr//3)*3:(rr//3)*3+3, (cc//3)*3:(cc//3)*3+3].flatten())
                            if v not in used:
                                removed += 1
                    return removed
                vals_ns = domains_ns[best_cell_ns]
                vals_sorted_ns = sorted(vals_ns, key=lambda vv: lcv_local(vv))
                next_actions = [(rns, cns, vv) for vv in vals_sorted_ns]
            if next_actions:
                next_max = max(Q[a[0], a[1], a[2]] for a in next_actions)
            else:
                next_max = 0

            old_value = Q[x, y, v]
            Q[x, y, v] = old_value + alpha * (reward + gamma * next_max - old_value)

            state = next_state
            total_rewards += reward
            episode_states.append(state.copy())

        # Strategy #1
        epsilon = max(0.01, epsilon * epsilon_decay)
        # Strategy #2
        # schedule epsilon based on episodes left:
        # start with low epsilon (exploit) and increase to high epsilon (explore) as episodes progress
        # eps_start = 0.6     # epsilon at episode 0 (more exploration)
        # eps_end = 0.01      # epsilon at last episode (more exploitation)
        # progress = episode / max(1, num_episodes - 1)   # 0.0 -> 1.0
        # linearly interpolate epsilon from eps_start -> eps_end as episodes progress
        # epsilon = float(np.clip(eps_start + (eps_end - eps_start) * progress,
        #                         min(eps_start, eps_end),
        #                         max(eps_start, eps_end)))
        
        rewards_all_episodes.append(total_rewards)

        # save per-episode snapshot periodically or on success
        if (episode % save_every == 0) or (total_rewards >= reward_goal) or (total_rewards > current_max_reward):
            snap_path = os.path.join(snapshots_dir, f'episode_{episode:05d}.npz')
            # save the sequence of intermediate grids and a copy of the current Q-table
            np.savez_compressed(snap_path,
                                episode=episode,
                                total_reward=total_rewards,
                                states=np.array(episode_states, dtype=np.int8),
                                Q=Q.copy())
        
        # optionally accumulate dataset for NN training:
        if accumulate_dataset:
            # store state -> target Q (use useful slice: rows 0..8, cols 0..8, values 1..9)
            dataset_states.extend(episode_states)
            dataset_Qs.extend([Q[:9, :9, 1:10].copy() for _ in episode_states])

        # periodically flush accumulated dataset to disk
        if accumulate_dataset and (episode % dataset_accumulate_every == 0 and episode > 0):
            dpath = os.path.join(snapshots_dir, f'dataset_up_to_ep_{episode:05d}.npz')
            np.savez_compressed(dpath,
                                X=np.array(dataset_states, dtype=np.int8),   # (N,9,9)
                                Y=np.array(dataset_Qs, dtype=np.float32))    # (N,9,9,9)
            # clear accumulators to free memory
            dataset_states = []
            dataset_Qs = []

        if total_rewards > current_max_reward:
            current_max_reward = total_rewards
            print(f"New max total reward: {current_max_reward} at episode: {episode} with epsilon: {epsilon:.4f}")

        if episode % 500 == 0:
            print(f"episode {episode}, total_reward {total_rewards}, epsilon {epsilon:.4f}")




def get_possible_actions_with_backtracking(grid, exclude=None):
    """
    Return possible actions like `get_possible_actions` but, if `exclude` is
    provided as a (r,c,v) tuple, rotate the returned list so the action after
    the excluded one (i.e. the "next denser" option) comes first. If the
    excluded action is not in the list, return the density-sorted list.
    """
    actions = get_possible_actions(grid)
    if exclude is None:
        return actions

    try:
        idx = next(i for i, a in enumerate(actions) if a[0] == exclude[0] and a[1] == exclude[1] and a[2] == exclude[2])
    except StopIteration:
        return actions

    # rotate so the next denser action becomes first
    if len(actions) <= 1:
        return actions
    next_idx = (idx + 1) % len(actions)
    return actions[next_idx:] + actions[:next_idx]


def get_possible_actions_mrv(grid, exclude=None):
    """
    MRV + Least-Constraining-Value selector.
    - Compute candidate domains for each empty cell.
    - Pick the cell with the minimum remaining values (MRV). Tie-break by degree (most neighbors empty).
    - Order candidate values by LCV: prefer values that eliminate the fewest choices in neighboring cells.
    - If `exclude` is provided as (r,c,v), try to rotate/skip that specific value for the chosen cell.
    Returns a list of (r,c,v) actions ordered by preference.
    """
    arr = grid.astype(int)
    # collect domains per empty cell
    domains = {}
    for r in range(9):
        for c in range(9):
            if arr[r, c] != 0:
                continue
            used = set(arr[r, :]) | set(arr[:, c]) | set(arr[(r//3)*3:(r//3)*3+3, (c//3)*3:(c//3)*3+3].flatten())
            used.discard(0)
            domain = [v for v in range(1, 10) if v not in used]
            domains[(r, c)] = domain

    if not domains:
        return []

    # MRV: find cells with smallest domain size
    min_size = min(len(d) for d in domains.values())
    mrv_cells = [cell for cell, d in domains.items() if len(d) == min_size]

    # degree tie-break: prefer cell with most empty neighbors (row/col/box)
    def degree(cell):
        r, c = cell
        neighbors = set()
        for cc in range(9):
            if cc != c and arr[r, cc] == 0:
                neighbors.add((r, cc))
        for rr in range(9):
            if rr != r and arr[rr, c] == 0:
                neighbors.add((rr, c))
        r0 = (r // 3) * 3
        c0 = (c // 3) * 3
        for rr in range(r0, r0 + 3):
            for cc in range(c0, c0 + 3):
                if (rr, cc) != (r, c) and arr[rr, cc] == 0:
                    neighbors.add((rr, cc))
        return len(neighbors)

    # pick best MRV cell
    best_cell = max(mrv_cells, key=degree)
    r, c = best_cell
    domain = domains[best_cell]

    # LCV: score each value by how many values it would remove from neighbors' domains
    def lcv_score(v):
        removed = 0
        # simulate placing v and count how many domain elements are removed in neighbors
        # row
        for cc in range(9):
            if cc == c or arr[r, cc] != 0:
                continue
            used = set(arr[r, :]) | set(arr[:, cc]) | set(arr[(r//3)*3:(r//3)*3+3, (cc//3)*3:(cc//3)*3+3].flatten())
            if v not in used:
                # placing v here would add v to used for the neighbor, removing v from its domain
                removed += 1
        # col
        for rr in range(9):
            if rr == r or arr[rr, c] != 0:
                continue
            used = set(arr[rr, :]) | set(arr[:, c]) | set(arr[(rr//3)*3:(rr//3)*3+3, (c//3)*3:(c//3)*3+3].flatten())
            if v not in used:
                removed += 1
        # box
        r0 = (r // 3) * 3
        c0 = (c // 3) * 3
        for rr in range(r0, r0 + 3):
            for cc in range(c0, c0 + 3):
                if (rr, cc) == (r, c) or arr[rr, cc] != 0:
                    continue
                used = set(arr[rr, :]) | set(arr[:, cc]) | set(arr[(rr//3)*3:(rr//3)*3+3, (cc//3)*3:(cc//3)*3+3].flatten())
                if v not in used:
                    removed += 1
        return removed

    # sort domain by increasing removed (least constraining = smallest removed)
    domain_sorted = sorted(domain, key=lambda val: lcv_score(val))

    actions = [(r, c, v) for v in domain_sorted]

    # if exclude indicates to skip a particular (r,c,v), rotate or filter
    if exclude is not None and (exclude[0], exclude[1]) == (r, c):
        # if excluded value present, rotate to try next value first
        vals = [a[2] for a in actions]
        try:
            idx = vals.index(exclude[2])
            if len(actions) > 1:
                next_idx = (idx + 1) % len(actions)
                actions = actions[next_idx:] + actions[:next_idx]
        except ValueError:
            pass

    return actions


def ac3_prune(domains):
    """AC-3 style constraint propagation over domains dict {(r,c): [vals]}
    Returns True if consistent, False if contradiction encountered. Mutates domains in place.
    """
    # neighbors function
    def neighbors(cell):
        r, c = cell
        neigh = set()
        for cc in range(9):
            if cc != c:
                neigh.add((r, cc))
        for rr in range(9):
            if rr != r:
                neigh.add((rr, c))
        r0 = (r // 3) * 3
        c0 = (c // 3) * 3
        for rr in range(r0, r0+3):
            for cc in range(c0, c0+3):
                if (rr, cc) != (r, c):
                    neigh.add((rr, cc))
        return [n for n in neigh if n in domains]

    from collections import deque
    queue = deque()
    for var in domains:
        for n in neighbors(var):
            queue.append((var, n))

    while queue:
        xi, xj = queue.popleft()
        revised = False
        dom_i = domains.get(xi, [])
        dom_j = domains.get(xj, [])
        new_dom_i = []
        for vi in dom_i:
            # keep vi if there exists some vj in dom_j != vi
            if any(vj != vi for vj in dom_j):
                new_dom_i.append(vi)
        if len(new_dom_i) < len(dom_i):
            domains[xi] = new_dom_i
            revised = True
        if not domains[xi]:
            return False
        if revised:
            for xk in neighbors(xi):
                if xk != xj:
                    queue.append((xk, xi))
    return True


def solve_with_csp(grid, max_solutions=1):
    """Deterministic recursive CSP solver using MRV+AC-3+LCV. Returns solved grid or None.
    This solver does not use RL; it's a pure CSP backtracking solver optimized with AC-3 pruning.
    """
    arr = grid.copy().astype(int)

    # build initial domains
    domains = {}
    for r in range(9):
        for c in range(9):
            if arr[r, c] != 0:
                continue
            used = set(arr[r, :]) | set(arr[:, c]) | set(arr[(r//3)*3:(r//3)*3+3, (c//3)*3:(c//3)*3+3].flatten())
            used.discard(0)
            domains[(r, c)] = [v for v in range(1, 10) if v not in used]

    if not ac3_prune(domains):
        return None

    solutions = []

    def recursive_search(dom):
        # if no domains left => solved
        if not dom:
            solutions.append(arr.copy())
            return True

        # pick MRV
        min_size = min(len(d) for d in dom.values())
        mrv_cells = [cell for cell, d in dom.items() if len(d) == min_size]
        # degree tie-break
        def degree(cell):
            r, c = cell
            neigh = set()
            for cc in range(9):
                if cc != c and arr[r, cc] == 0:
                    neigh.add((r, cc))
            for rr in range(9):
                if rr != r and arr[rr, c] == 0:
                    neigh.add((rr, c))
            r0 = (r // 3) * 3
            c0 = (c // 3) * 3
            for rr in range(r0, r0+3):
                for cc in range(c0, c0+3):
                    if (rr, cc) != (r, c) and arr[rr, cc] == 0:
                        neigh.add((rr, cc))
            return len(neigh)

        cell = max(mrv_cells, key=degree)
        r, c = cell
        # order values by LCV
        def lcv_score(v):
            removed = 0
            for cc in range(9):
                if cc == c or arr[r, cc] != 0:
                    continue
                used = set(arr[r, :]) | set(arr[:, cc]) | set(arr[(r//3)*3:(r//3)*3+3, (cc//3)*3:(cc//3)*3+3].flatten())
                if v not in used:
                    removed += 1
            for rr in range(9):
                if rr == r or arr[rr, c] != 0:
                    continue
                used = set(arr[rr, :]) | set(arr[:, c]) | set(arr[(rr//3)*3:(rr//3)*3+3, (c//3)*3:(c//3)*3+3].flatten())
                if v not in used:
                    removed += 1
            r0 = (r // 3) * 3
            c0 = (c // 3) * 3
            for rr in range(r0, r0+3):
                for cc in range(c0, c0+3):
                    if (rr, cc) == (r, c) or arr[rr, cc] != 0:
                        continue
                    used = set(arr[rr, :]) | set(arr[:, cc]) | set(arr[(rr//3)*3:(rr//3)*3+3, (cc//3)*3:(cc//3)*3+3].flatten())
                    if v not in used:
                        removed += 1
            return removed

        values = sorted(dom[cell], key=lcv_score)

        # Try every candidate and record which candidates lead to a solution.
        success_flags = []
        tried_values = []
        for v in values:
            tried_values.append(v)
            arr[r, c] = v

            # build new domains for other variables
            new_dom = {}
            contradiction = False
            for key, d in dom.items():
                if key == cell:
                    continue
                rr, cc = key
                used = set(arr[rr, :]) | set(arr[:, cc]) | set(arr[(rr//3)*3:(rr//3)*3+3, (cc//3)*3:(cc//3)*3+3].flatten())
                used.discard(0)
                nd = [val for val in d if val not in used]
                if not nd:
                    contradiction = True
                    break
                new_dom[key] = nd

            if contradiction:
                success_flags.append(False)
                arr[r, c] = 0
                continue

            # apply AC-3 pruning
            if not ac3_prune(new_dom):
                success_flags.append(False)
                arr[r, c] = 0
                continue

            # recurse
            prev_solutions = len(solutions)
            recursive_search(new_dom)
            found = len(solutions) > prev_solutions
            success_flags.append(bool(found))

            # undo assignment
            arr[r, c] = 0

        # Build policy target for this decision point
        policy = np.zeros((9, 9, 9), dtype=np.float32)
        any_success = any(success_flags)
        if any_success:
            for idx, ok in enumerate(success_flags):
                if ok:
                    v = tried_values[idx]
                    policy[r, c, v-1] = 1.0
            s = policy[r, c, :].sum()
            if s > 0:
                policy[r, c, :] /= s
            value_label = 1.0
        else:
            # no successful candidate from this state
            for v in tried_values:
                policy[r, c, v-1] = 0.0
            value_label = 0.0

        try:
            dataset_csp_states.append(arr.copy())
            dataset_csp_policy.append(policy)
            dataset_csp_value.append(float(value_label))
        except Exception:
            pass

        # Return whether we have reached the desired number of solutions
        return len(solutions) >= max_solutions

    recursive_search(domains)
    return solutions[0] if solutions else None


def _record_policy_target_from_mrv(state, actions):
    """Given a state and MRV-chosen actions for one cell, record a policy
    target: a 9x9x9 array where probabilities are non-zero only for the
    actions suggested for the MRV cell. We convert counts (here all 1) into
    a normalized distribution. Append to dataset_policy_states/targets.
    """
    # build a 9x9x9 zero array
    policy = np.zeros((9, 9, 9), dtype=np.float32)
    if not actions:
        return
    r, c, _ = actions[0]
    # each candidate action corresponds to a value for (r,c)
    for (_, _, v) in actions:
        policy[r, c, v-1] += 1.0
    # normalize over values for that cell
    s = policy[r, c, :].sum()
    if s > 0:
        policy[r, c, :] /= s
    dataset_policy_states.append(state.copy())
    dataset_policy_targets.append(policy)


def flush_datasets_now(filename=None):
    """Write accumulated dataset arrays to a compressed NPZ file.
    This is safe to call from external runners (e.g., solve_now.py).
    """
    global dataset_states, dataset_Qs, dataset_policy_states, dataset_policy_targets
    global dataset_value_states, dataset_value_targets
    global dataset_csp_states, dataset_csp_policy, dataset_csp_value

    if filename is None:
        filename = os.path.join(snapshots_dir, 'dataset_quick_flush.npz')

    np.savez_compressed(filename,
                        X=np.array(dataset_states, dtype=np.int8) if dataset_states else np.empty((0,9,9), dtype=np.int8),
                        Y=np.array(dataset_Qs, dtype=np.float32) if dataset_Qs else np.empty((0,9,9,9), dtype=np.float32),
                        policy_states=np.array(dataset_policy_states, dtype=np.int8) if dataset_policy_states else np.empty((0,9,9), dtype=np.int8),
                        policy_targets=np.array(dataset_policy_targets, dtype=np.float32) if dataset_policy_targets else np.empty((0,9,9,9), dtype=np.float32),
                        value_states=np.array(dataset_value_states, dtype=np.int8) if dataset_value_states else np.empty((0,9,9), dtype=np.int8),
                        value_targets=np.array(dataset_value_targets, dtype=np.float32) if dataset_value_targets else np.empty((0,), dtype=np.float32),
                        csp_states=np.array(dataset_csp_states, dtype=np.int8) if dataset_csp_states else np.empty((0,9,9), dtype=np.int8),
                        csp_policy=np.array(dataset_csp_policy, dtype=np.float32) if dataset_csp_policy else np.empty((0,9,9,9), dtype=np.float32),
                        csp_value=np.array(dataset_csp_value, dtype=np.float32) if dataset_csp_value else np.empty((0,), dtype=np.float32))

    # clear accumulators
    dataset_states = []
    dataset_Qs = []
    dataset_policy_states = []
    dataset_policy_targets = []
    dataset_value_states = []
    dataset_value_targets = []
    dataset_csp_states = []
    dataset_csp_policy = []
    dataset_csp_value = []


def run_episodes_with_backtracking(start, num_episodes, max_steps_per_episode, max_backtracks=50):
    """
    Run episodes similarly to `run_episodes` but when encountering no legal
    actions attempt up to `max_backtracks` steps of backtracking. Backtracking
    fully restores the grid state, epsilon, the per-state reward bookkeeping,
    and the Q-table snapshot recorded for each state.
    """
    global Q, epsilon, rewards_all_episodes, dataset_states, dataset_Qs
    global dataset_policy_states, dataset_policy_targets, dataset_value_states, dataset_value_targets

    if not is_valid(start):
        raise ValueError("start grid is not a valid partial Sudoku (contains duplicates). Fix the puzzle before training.")

    current_max_reward = -float('inf')
    for episode in range(num_episodes):
        state = start.copy()
        total_rewards = 0

        # per-step bookkeeping
        episode_states = [state.copy()]
        episode_epsilons = [epsilon]
        episode_rewards = [0]
        episode_Qs = [Q.copy()]

        done = False
        for step in range(max_steps_per_episode):
            actions = get_possible_actions(state)

            if len(actions) == 0:
                backtracked = False
                backtracks = 0

                while len(episode_states) > 1 and backtracks < max_backtracks:
                    curr = episode_states[-1]
                    prev = episode_states[-2]
                    diffs = np.where(curr != prev)
                    if diffs[0].size == 0:
                        # duplicate state; pop bookkeeping and continue
                        episode_states.pop()
                        if len(episode_epsilons) > 1:
                            episode_epsilons.pop()
                        if len(episode_rewards) > 1:
                            undone = episode_rewards.pop()
                            total_rewards -= undone
                        if len(episode_Qs) > 1:
                            episode_Qs.pop()
                            Q[:] = episode_Qs[-1]
                        backtracks += 1
                        continue

                    r = int(diffs[0][0])
                    c = int(diffs[1][0])
                    prev_move = (r, c, int(curr[r, c]))

                    # pop the last state and restore bookkeeping to previous
                    if len(episode_rewards) > 0:
                        undone = episode_rewards.pop()
                        total_rewards -= undone
                    episode_states.pop()
                    if len(episode_epsilons) > 0:
                        episode_epsilons.pop()
                    if len(episode_Qs) > 1:
                        episode_Qs.pop()
                        Q[:] = episode_Qs[-1]

                    state = episode_states[-1].copy()
                    backtracks += 1

                    # ask for alternatives using MRV+LCV (fall back to density-ordered alternatives)
                    alternatives = get_possible_actions_mrv(state, exclude=prev_move)
                    if not alternatives:
                        alternatives = get_possible_actions_with_backtracking(state, exclude=prev_move)
                    if alternatives:
                        # record a policy target derived from MRV alternatives for training
                        try:
                            _record_policy_target_from_mrv(state, alternatives)
                        except Exception:
                            # be conservative: don't break training if recording fails
                            pass
                        # restore epsilon for this restored state
                        if episode_epsilons:
                            epsilon = float(episode_epsilons[-1])
                        actions = alternatives
                        backtracked = True
                        break

                if backtracked:
                    # try picking an alternative action at the restored state
                    continue

                # nothing to backtrack to -> apply stuck penalty and end episode
                total_rewards += reward_stuck
                if len(episode_rewards) >= 1:
                    episode_rewards[-1] += reward_stuck
                break

            # choose action and perform step
            x, y, v = choose_action(actions)
            next_state = state.copy()
            next_state[x, y] = v

            if not is_valid(next_state):
                reward = reward_fire
                old_value = Q[x, y, v]
                next_max = 0
                Q[x, y, v] = old_value + alpha * (reward + gamma * next_max - old_value)
                total_rewards += reward
                next_state = state.copy()
                episode_states.append(next_state.copy())
                episode_epsilons.append(epsilon)
                episode_rewards.append(reward)
                episode_Qs.append(Q.copy())
                continue

            completed_row = is_row_complete(next_state, x)
            completed_col = is_col_complete(next_state, y)
            completed_box = is_box_complete(next_state, x, y)

            if is_complete(next_state):
                reward = reward_goal
                old_value = Q[x, y, v]
                next_max = 0
                Q[x, y, v] = old_value + alpha * (reward + gamma * next_max - old_value)
                total_rewards += reward
                state = next_state
                episode_states.append(state.copy())
                episode_epsilons.append(epsilon)
                episode_rewards.append(reward)
                episode_Qs.append(Q.copy())
                done = True
                break

            reward = reward_step
            if completed_row or completed_col or completed_box:
                reward += reward_cp_complete

            next_actions = get_possible_actions(next_state)
            if next_actions:
                next_max = max(Q[a[0], a[1], a[2]] for a in next_actions)
            else:
                next_max = 0

            old_value = Q[x, y, v]
            Q[x, y, v] = old_value + alpha * (reward + gamma * next_max - old_value)

            state = next_state
            total_rewards += reward
            episode_states.append(state.copy())
            episode_epsilons.append(epsilon)
            episode_rewards.append(reward)
            episode_Qs.append(Q.copy())

        # end episode: decay epsilon and store reward
        eps_start = 0.6
        eps_end = 0.01
        progress = episode / max(1, num_episodes - 1)
        epsilon = float(np.clip(eps_start + (eps_end - eps_start) * progress,
                                 min(eps_start, eps_end),
                                 max(eps_start, eps_end)))
        rewards_all_episodes.append(total_rewards)

        # Build value targets (return-to-go) from episode_rewards and episode_states
        try:
            # compute returns for each state in episode
            returns = []
            g = 0.0
            for rwd in reversed(episode_rewards):
                g = rwd + gamma * g
                returns.append(g)
            returns.reverse()
            # align with episode_states length (should match)
            for s_grid, val in zip(episode_states, returns):
                dataset_value_states.append(s_grid.copy())
                dataset_value_targets.append(float(val))
        except Exception:
            pass

        # save snapshot periodically
        if (episode % save_every == 0) or (total_rewards >= reward_goal):
            snap_path = os.path.join(snapshots_dir, f'episode_bt_{episode:05d}.npz')
            np.savez_compressed(snap_path,
                                episode=episode,
                                total_reward=total_rewards,
                                states=np.array(episode_states, dtype=np.int8),
                                Q=Q.copy())

        if accumulate_dataset:
            dataset_states.extend(episode_states)
            dataset_Qs.extend([Q[:9, :9, 1:10].copy() for _ in episode_states])

        if accumulate_dataset and (episode % dataset_accumulate_every == 0 and episode > 0):
            dpath = os.path.join(snapshots_dir, f'dataset_up_to_ep_bt_{episode:05d}.npz')
            # save existing state->Q data plus new policy/value datasets
            np.savez_compressed(dpath,
                                X=np.array(dataset_states, dtype=np.int8),
                                Y=np.array(dataset_Qs, dtype=np.float32),
                                policy_states=np.array(dataset_policy_states, dtype=np.int8) if dataset_policy_states else np.empty((0,9,9), dtype=np.int8),
                                policy_targets=np.array(dataset_policy_targets, dtype=np.float32) if dataset_policy_targets else np.empty((0,9,9,9), dtype=np.float32),
                                value_states=np.array(dataset_value_states, dtype=np.int8) if dataset_value_states else np.empty((0,9,9), dtype=np.int8),
                                value_targets=np.array(dataset_value_targets, dtype=np.float32) if dataset_value_targets else np.empty((0,), dtype=np.float32),
                                csp_states=np.array(dataset_csp_states, dtype=np.int8) if dataset_csp_states else np.empty((0,9,9), dtype=np.int8),
                                csp_policy=np.array(dataset_csp_policy, dtype=np.float32) if dataset_csp_policy else np.empty((0,9,9,9), dtype=np.float32),
                                csp_value=np.array(dataset_csp_value, dtype=np.float32) if dataset_csp_value else np.empty((0,), dtype=np.float32))
            dataset_states = []
            dataset_Qs = []
            dataset_policy_states = []
            dataset_policy_targets = []
            dataset_value_states = []
            dataset_value_targets = []
            dataset_csp_states = []
            dataset_csp_policy = []
            dataset_csp_value = []

        if total_rewards > current_max_reward:
            current_max_reward = total_rewards
            print(f"New max total reward: {current_max_reward} at episode: {episode} with epsilon: {epsilon:.4f}")

        if episode % 500 == 0:
            print(f"episode {episode}, total_reward {total_rewards}, epsilon {epsilon:.4f}")

def main():
    run_episodes_with_backtracking(start, num_episodes, max_steps_per_episode, max_backtracks=50)
    #run_episodes(start, num_episodes, max_steps_per_episode)
    print(Q)
    optimal_path = get_optimal_path(Q, start)
    plot_sudoku_solution(optimal_path, 'sudoku_solution.png')
    plot_rewards(rewards_all_episodes)

if __name__ == '__main__':
    main()