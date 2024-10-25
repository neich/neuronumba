

class BaseLinearModel:
    def __init__(self, model):
        self.model = model

    def compute_matrix(self, sc, sigma):
        raise NotImplementedError