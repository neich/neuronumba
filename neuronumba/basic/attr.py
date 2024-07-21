import inspect

class Attr(object):

    def __init__(self, default=None, required=True, dependant=False):
        self.default = default
        self.required = bool(required)
        self.dependant = dependant


class HasAttr(object):
    def __init__(self, **kwargs):
        cls = type(self)
        class_attrs = dict(inspect.getmembers(cls, lambda o: isinstance(o, Attr)))
        # Initialize all attributes with its default
        for name, value in class_attrs.items():
            if not value.dependant:
                setattr(self, name, value.default)
        # Set values of defined attributes in the arguments
        for name, value in kwargs.items():
            if name not in class_attrs:
                raise AttributeError(f"Attribute <{name}> not found in <{cls.__name__}>!")
            if class_attrs[name].dependant:
                raise AttributeError(f"Attribute <{name}> of class <{cls.__name__}> is dependant and cannot be manually initialized!")
            setattr(self, name, value)


    def configure(self, **kwargs):
        cls = type(self)
        class_attrs = dict(inspect.getmembers(cls, lambda o: isinstance(o, Attr)))
        for name, value in kwargs.items():
            if name not in class_attrs:
                raise AttributeError(f"Attribute <{name}> not found in <{cls.__name__}>!")
            if class_attrs[name].dependant:
                raise AttributeError(f"Attribute <{name}> of class <{cls.__name__}> is dependant and cannot be manually initialized!")

            setattr(self, name, value)
        self._check_required()
        self._init_dependant()

    def _check_required(self):
        cls = type(self)
        class_attrs = dict(inspect.getmembers(cls, lambda o: isinstance(o, Attr)))
        for name, value in class_attrs.items():
            if value.required and getattr(self, name) is None:
                raise AttributeError(f"Attribute <{name}> of class <{cls.__name__}> has no value and is required to have one!")

    def _init_dependant(self):
        pass
