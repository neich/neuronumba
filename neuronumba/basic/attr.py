import inspect
from enum import Enum


class AttrType(Enum):
    Unknown = 1,
    Model = 2


class Attr(object):

    def __init__(self, default=None, required=True, dependant=False, attr_type=AttrType.Unknown):
        self.default = default
        self.required = bool(required)
        self.dependant = dependant
        self.attr_type = attr_type


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

    @classmethod
    def _get_attributes(cls):
        return dict(inspect.getmembers(cls, lambda o: isinstance(o, Attr)))

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
        self._init_dependant_automatic()

    def _check_required(self):
        cls = type(self)
        class_attrs = dict(inspect.getmembers(cls, lambda o: isinstance(o, Attr)))
        for name, value in class_attrs.items():
            if value.required and getattr(self, name) is None:
                raise AttributeError(f"Attribute <{name}> of class <{cls.__name__}> has no value and is required to have one!")

    def _init_dependant(self):
        pass

    def _init_dependant_automatic(self):
        pass

