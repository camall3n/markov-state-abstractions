from . import gridworld

class TaxiWorld(gridworld.GridWorld):
    def __init__(self):
        super().__init__(rows=5, cols=5)

        # Add default walls for taxi problem
        self._grid[1:4,4] = 1
        self._grid[7:10,2] = 1
        self._grid[7:10,6] = 1
