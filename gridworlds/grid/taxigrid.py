from . import basicgrid

class TaxiGrid5x5(basicgrid.BasicGrid):
    def __init__(self):
        super().__init__(rows=5, cols=5)
        self._grid[1:4,4] = 1
        self._grid[7:10,2] = 1
        self._grid[7:10,6] = 1

class TaxiGrid10x10(basicgrid.BasicGrid):
    def __init__(self):
        super().__init__(rows=10, cols=10)
        self._grid[1:8,6] = 1
        self._grid[13:20,2] = 1
        self._grid[13:20,8] = 1
        self._grid[5:12,12] = 1
        self._grid[1:8,16] = 1
        self._grid[13:20,16] = 1
