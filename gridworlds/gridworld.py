from .grid import basicgrid

class GridWorld(basicgrid.BasicGrid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def reset(self):
        pass

    def step(self, action):
        pass
