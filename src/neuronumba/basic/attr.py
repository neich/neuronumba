import inspect


class Attr(object):

    _DEFAULT_TAG = 'unknown'

    def __init__(self, default=None, required=True, dependant=False, attributes=None, doc=None):
        if attributes is None:
            self.attributes = [Attr._DEFAULT_TAG]
        elif isinstance(attributes, list):
            self.attributes = attributes
        elif isinstance(attributes, str):
            self.attributes = [attributes]
        else:
            raise TypeError(f"Cannot initialize Attr attributes with type {type(attributes)}")
        self.default = default
        self.required = bool(required)
        self.dependant = dependant
        self.doc = doc


class HasAttr(object):

    class Tag:
        """Base attribute tags. Subclasses can extend via inheritance."""
        UNKNOWN = 'unknown'

    def __init__(self, **kwargs):
        self._defined_attrs = {}
        self._configured = False
        self._set_defaults()
        self._set_values(kwargs)
        if self._has_all_required():
            self.configure()

    def _set_defaults(self):
        """Set default values for all non-dependant attributes."""
        for name, attr in type(self)._get_attributes().items():
            if not attr.dependant:
                setattr(self, name, attr.default)

    def _set_values(self, attrs):
        """Validate and set attribute values."""
        cls = type(self)
        class_attrs = cls._get_attributes()
        for name, value in attrs.items():
            if name not in class_attrs:
                raise AttributeError(f"Attribute <{name}> not found in <{cls.__name__}>!")
            if class_attrs[name].dependant:
                raise AttributeError(
                    f"Attribute <{name}> of class <{cls.__name__}> is dependant and cannot be manually initialized!")
            setattr(self, name, value)
            self._defined_attrs[name] = value

    def _has_all_required(self):
        """Check whether all required non-dependant attributes have values."""
        for name, attr in type(self)._get_attributes().items():
            if attr.required and not attr.dependant and getattr(self, name, None) is None:
                return False
        return True

    def _attr_defined(self, attr):
        return attr in self._defined_attrs

    @classmethod
    def _get_attributes(cls):
        return dict(inspect.getmembers(cls, lambda o: isinstance(o, Attr)))

    @property
    def configured(self):
        return self._configured

    def configure(self, **kwargs):
        """Update attributes and (re)compute dependant values.

        Automatically called from the constructor when all required attributes
        have values. Can also be called explicitly to provide additional
        attributes or trigger recomputation.
        """
        if kwargs:
            self._set_values(kwargs)
        self._check_required()
        self._init_dependant()
        self._init_dependant_automatic()
        self._configured = True
        return self

    def _check_required(self):
        cls = type(self)
        for name, attr in cls._get_attributes().items():
            if attr.required and not attr.dependant and getattr(self, name) is None:
                raise AttributeError(f"Attribute <{name}> of class <{cls.__name__}> has no value and is required to have one!")

    def _init_dependant(self):
        pass

    def _init_dependant_automatic(self):
        pass

    def set_attributes(self, attrs):
        """Set attribute values without recomputing dependants.

        Use configure() instead unless you need to batch-set multiple
        attributes before a single configure() call.
        """
        self._set_values(attrs)
        return self

    def get_attributes(self):
        attrs = {}
        for name, attr in type(self)._get_attributes().items():
            if not attr.dependant:
                attrs[name] = getattr(self, name)
        return attrs
