from .nnutils import Reshape

class NullAbstraction(Reshape):
    def freeze(self):
        pass

    def parameters(self):
        return []
