from src.neuronumba import HasAttr, Attr


class Integrator(HasAttr):
    dt = Attr(default=None, required=True)



