from neuronumba.basic.attr import HasAttr, Attr


class Bold(HasAttr):
    # TR of the bold signal in milliseconds
    tr = Attr(default=2000.0, required=False)

