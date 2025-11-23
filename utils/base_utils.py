from argparse import Namespace


class DotDict(dict):
    def __init__(self, mapping=None, /, **kwargs):
        if mapping is None:
            mapping = {}
        elif type(mapping) is Namespace:
            mapping = vars(mapping)

        super().__init__(mapping, **kwargs)

    def __getattr__(self, key):
        try:
            value = self[key]
            if type(value) is dict:
                value = DotDict(value)
            return value
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return "<DotDict " + dict.__repr__(self) + ">"

    def todict(self):
        return {k: v for k, v in self.items()}


dotdict = DotDict