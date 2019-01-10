import numpy as np
import matplotlib.pyplot as plt

grid_type = int

# Offsets:
LEFT  = np.asarray(( 0, -1))
RIGHT = np.asarray(( 0,  1))
UP    = np.asarray((-1,  0))
DOWN  = np.asarray(( 1,  0))

class BasicGrid:
    def __init__(self, rows, cols):
        self._rows = rows
        self._cols = cols

        # Add rows and columns for walls between cells
        self._grid = np.ones([rows*2+1, cols*2+1], dtype=grid_type)
        self._contents = np.empty([rows*2+1, cols*2+1], dtype=np.object)

        # Reset valid positions and walls
        self._grid[1:-1:2,1:-1] = 0
        self._grid[1:-1,1:-1:2] = 0

    def contents(self, row, col):
        return self._contents[row//2, col//2]

    def plot(self):
        plt.figure(figsize=(3*self._cols/self._rows,3))
        ax = plt.axes()
        ax.axis('off')
        plt.xlim([-0.1,self._cols+0.1]), plt.ylim([-0.1,self._rows+0.1])
        plt.xticks([]), plt.yticks([])
        plt.gca().invert_yaxis()
        # Get lists of vertical and horizontal wall locations
        v_walls = self._grid[:,::2][1::2,:]
        h_walls = self._grid[::2,:][:,1::2].transpose()
        row_range = np.linspace(0,self._rows,self._rows+1)
        col_range = np.linspace(0,self._cols,self._cols+1)
        for row in range(self._rows):
            plt.vlines(col_range[v_walls[row]==1], row, row+1)
        for col in range(self._cols):
            plt.hlines(row_range[h_walls[col]==1], col, col+1)
        return ax

    def has_wall(self, position, offset):
        row, col = position
        d_row, d_col = offset
        wall_row = 2*row+1+d_row
        wall_col = 2*col+1+d_col
        return self._grid[wall_row, wall_col]

    def save(self, filename):
        np.savetxt(filename, self._grid.astype(int), fmt='%1d')

    def load(self, filename):
        grid = np.loadtxt(filename, dtype=grid_type)
        r, c = grid.shape
        self._rows = r // 2
        self._cols = c // 2
        self._grid = grid
        self._contents = np.empty_like(self._grid, dtype=np.object)
