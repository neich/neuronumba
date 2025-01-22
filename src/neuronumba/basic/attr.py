import inspect
from enum import Enum

class AttrEnum(set):
    def __init__(self, value=None):
        super().__init__()
        self._pre = []
        self._value = value

    def __set_name__(self, owner, name):
        self._name = owner.__name__
        base_chain = inspect.getmro(owner)
        if len(base_chain) > 2:
            self._pre.append(base_chain[1])
        self._additems(self._value)

    def _getfullname(self, name):
        return self._name + "." + name

    def __getattr__(self, name):
        if name.startswith('_'):
            return self.__dict__[name]
        full_name = self._getfullname(name)
        if full_name in self:
            return full_name
        else:
            for pre in self._pre:
                if getattr(pre, name):
                    return pre._getfullname(name)
        raise AttributeError

    def _additems(self, value):
        if isinstance(value, str):
            self.add(self._getfullname(value))
        elif isinstance(value, list):
            self.update([self._getfullname(x) for x in value])
        else:
            raise TypeError(f"Cannot update AttrEnum with type {type(value)}")


class Attr(object):

    def __init__(self, default=None, required=True, dependant=False, attributes=None):
        if attributes is None:
            self.attributes = []
        else:
            if isinstance(attributes, list):
                self.attributes = attributes
            elif isinstance(attributes, str):
                self.attributes = [attributes]
            else:
                raise TypeError(f"Cannot initialize Attr attributes with type {type(attributes)}")
        self.default = default
        self.required = bool(required)
        self.dependant = dependant


class HasAttr(object):
    Type = AttrEnum(['Unknown'])

    def __init__(self, **kwargs):
        self._defined_attrs = {}
        self._init_attributes(kwargs)

    def _attr_defined(self, attr):
        return attr in self._defined_attrs

    def _init_attributes(self, kwargs, set_default=True):
        cls = type(self)
        class_attrs = dict(inspect.getmembers(cls, lambda o: isinstance(o, Attr)))
        if set_default:
            # Initialize all attributes with its default
            for name, value in class_attrs.items():
                if not value.dependant:
                    setattr(self, name, value.default)
        # Set values of defined attributes in the arguments
        for name, value in kwargs.items():
            if name not in class_attrs:
                raise AttributeError(f"Attribute <{name}> not found in <{cls.__name__}>!")
            if class_attrs[name].dependant:
                raise AttributeError(
                    f"Attribute <{name}> of class <{cls.__name__}> is dependant and cannot be manually initialized!")
            setattr(self, name, value)
            self._defined_attrs[name] = value

    @classmethod
    def _get_attributes(cls):
        return dict(inspect.getmembers(cls, lambda o: isinstance(o, Attr)))

    def configure(self, **kwargs):
        self._init_attributes(kwargs, set_default=False)

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

