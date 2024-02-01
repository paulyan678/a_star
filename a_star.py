import numpy as np
import matplotlib.pyplot as plt
import heapq as hq

class Grid:
    def __init__(self, width, height, fill_rate):
        self.grid = np.random.choice([0, 1], size=(width, height), p=[1-fill_rate, fill_rate])
        self.height = height
        self.width = width
    
    def plot_grid(self):
        plt.imshow(1 - self.grid, cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
        plt.show()
    
    def plot_path_on_grid(self, path):
        plt.imshow(1 - self.grid, cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
        plt.plot(path[:, 1], path[:, 0], color='red', linewidth=2)
        plt.show()


def reconstruct_path(dest, came_from):
    path = []
    current = dest

    while current is not None:
        path.append([current[0], current[1]])
        current = came_from.get(current)

    return path[::-1]  # Reverse the path to start from the source





def h(src : tuple, dest : tuple) -> int:
    return abs(dest[0] - src[0]) + abs(dest[1] - src[1])

def a_star(src: tuple, dest : tuple, grid) -> list:
    pq = []
    hq.heappush(pq, (h(src, dest), src))
    min_dest = np.ones((grid.height, grid.width))*np.inf
    min_dest[src[0], src[1]] = 0
    came_from = {}
    came_from[src] = None
    motion = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while len(pq) > 0:
        _, cur_explore = hq.heappop(pq)
        cur_row, cur_col = cur_explore
        if cur_explore == dest:
            return reconstruct_path(dest, came_from)
        for ver, hor in motion:
            row_ind = cur_row + ver
            col_ind = cur_col + hor
            if (row_ind >= 0) and (row_ind < grid.height) and (col_ind >= 0) and (col_ind < grid.width) and (grid.grid[row_ind, col_ind] != 1):
                new_dist = min_dest[cur_row, cur_col] + 1
                if new_dist < min_dest[row_ind, col_ind]:
                    min_dest[row_ind, col_ind] = new_dist
                    came_from[(row_ind, col_ind)] = cur_explore
                    f = h((row_ind, col_ind), dest) + new_dist
                    hq.heappush(pq, (f, (row_ind, col_ind)))
    
    return None





grid = Grid(500, 500, 0.15)
grid.plot_grid()
src = (0, 0)
dest = (499, 499)
path = a_star(src, dest, grid)
if path != None:
    grid.plot_path_on_grid(np.array(path))
else:
    print("no path found")
