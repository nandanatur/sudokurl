import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

start = np.array([
    [5,0,0,0,2,7,0,0,9],
    [0,0,4,1,0,0,0,0,0],
    [0,1,0,0,5,0,3,0,0],
    [0,9,2,0,6,0,8,0,0],
    [0,5,0,0,0,0,0,0,6],
    [6,0,0,7,0,0,2,9,0],
    [8,0,0,0,7,0,0,0,2],
    [0,0,0,0,0,0,0,8,0],
    [0,0,9,0,0,3,6,0,0]
])

next_state = start.copy()

summary_file = None
sample = None
num_episodes = 10000
max_steps_per_episode = 200
alpha = 0.1
gamma = 0.9
epsilon = 0.6
epsilon_decay = 0.9995

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

snapshots_dir = None
save_every = 100   # save snapshots every N episodes
accumulate_dataset = True
dataset_accumulate_every = 500   # flush accumulated dataset to disk every N episodes

# optional accumulators for building a training set (state -> Q)
dataset_states = []
dataset_Qs = []

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
        # pick at random
        # return actions[np.random.randint(len(actions))]
        # pick first denser option
        return actions[0]

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
    plt.savefig(f'solution/strategy2/{sample}/rewards.png')

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
    - Otherwise behaves as the original maze-style path finder when `maze`,
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

def rows_complete(grid):
    counter = 0
    for row_idx in range(9):
        row = grid[row_idx]
        if np.any(row == 0):
            continue
        if set(row.tolist()) == set(range(1, 10)):
            print(row_idx)
            counter += 1
        else:
            print(f"row: {row_idx} is not complete")
    return counter

def cols_complete(grid):
    counter = 0
    for col_idx in range(9):
        col = grid[:, col_idx]
        if np.any(col == 0):
            continue
        if set(col.tolist()) == set(range(1, 10)):
            counter += 1
    return counter

def boxes_complete(grid):
    counter = 0
    for row_idx, col_idx in [(1,1), (1,4), (1,7), (4,1), (4,4), (4,7), (7,1), (7,4), (7,7)]:
        r0 = (row_idx // 3) * 3
        c0 = (col_idx // 3) * 3
        box = grid[r0:r0+3, c0:c0+3].flatten()
        if np.any(box == 0):
            continue
        if set(box.tolist()) == set(range(1, 10)):
            counter += 1
    return counter

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

    # Strategy #1: return actions as is
    # return actions
    # Strategy #2: return actions sorted on density
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
            actions = get_possible_actions(state)

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

def run():
    run_episodes(start, num_episodes, max_steps_per_episode)
    print(Q)
    optimal_path = get_optimal_path(Q, start)
    plot_sudoku_solution(optimal_path, f'solution/strategy2/{sample}/sudoku_solution.png')
    plot_rewards(rewards_all_episodes)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        try:
            sample = int(sys.argv[1])
            if sample < 0:
                raise Exception("Invalid training sample number")
            counter = -1
            with open("sudoku_extreme_dataset/train.csv") as fh:
                for line in fh:
                    if counter == sample:
                        question = line.split(",")[1]
                        start = np.array(get_sudoku_grid(question))
                        snapshots_dir = f"snapshots/strategy2/{sample}"
                        break
                    counter += 1
                else:
                    raise Exception("Invalid sample number")
            os.makedirs(snapshots_dir, exist_ok=True)
            os.makedirs(f"solution/strategy2/{sample}", exist_ok=True)
            print(f"Training sample: #{sample}")
            print(start)
            run()
        except:
            raise
    else:
        print("Please provide 1 argument to this program which is index to train.csv sample")
    
    
