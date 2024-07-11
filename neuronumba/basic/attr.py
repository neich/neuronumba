import inspect

class Attr(object):

    def __init__(self, default=None, required=True):
        self.default = default
        self.required = bool(required)


class HasAttr(object):
    def __init__(self, **kwargs):
        cls = type(self)
        attr_list = inspect.getmembers(cls)
        for (name, value) in attr_list:
            if isinstance(value, Attr):
                if name in kwargs:
                    setattr(self, name, kwargs.get(name))
                else:
                    setattr(self, name, value.default)

    def __set__(self, name, value):
        cls = type(self)
        assert hasattr(cls, name), f"Class {cls} has not attribute {name}"
        attr = getattr(cls, name)
        sc = type(value).__mro__
        self.__setattr__(name, value)

    def configure(self, **kwargs):
        cls = type(self)
        for name, value in kwargs.items():
            if hasattr(self, name):
                self.__set__(name, value)
