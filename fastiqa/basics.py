from fastai.vision.all import *
import json # save settings
import logging
from .loss import *

# class IqaData():
#     db = None
#
#     def __init__(self, db, filter=None, **kwargs):
#         self.label = db() if isinstance(db, type) else db
#         self.name = f'{self.db.name}_{self.__class__.__name__}'
#         self.__dict__.update(kwargs)
#
#     def __getattr__(self, k: str):
#         return getattr(self.db, k)


class IqaModel(Module):
    __name__ = None
    n_out = 1

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        super().__init__()
        if self.__name__ is None: self.__name__ = self.__class__.__name__.split('__')[0]

    def bunch(self, dls):
        # assert not isinstance(dls, (tuple, list)), "do dls.bunch() first"
        # logging.info(f'bunching ... {self.__name__}@{dls.__name__}')
        #
        # if isinstance(dls.label_col, (list, tuple)):
        #     if len(dls.label_col) != self.n_out:
        #         dls.label_col = dls.label_col[:self.n_out]
        #         print(f'Changed dls.label_col to ({dls.label_col}) to fit model.n_out ({dls.__name__})')

        return dls

def serialize(obj):
    """JSON serializer for objects not serializable by default json code"""
    # if isinstance(obj, (datetime, date)):
    #     return obj.isoformat()
    # https://stackoverflow.com/questions/19628421/how-to-check-if-str-is-implemented-by-an-object
    if type(obj).__str__ is not object.__str__: # pathlib
        return str(obj)
    if obj.__class__.__name__ == 'function':
        return obj.__name__
    if hasattr(obj, 'to_json'):
        return obj.to_json()
    d = {k:(obj.__dict__[k]) for k in obj.__dict__ if not k.startswith('_')}
    d['__class__.__name__'] = obj.__class__.__name__
    return d
    # return {k:(obj.k) for k in dir(obj) if not k.startswith('_')}
    # raise TypeError ("Type %s not serializable" % type(obj))

def to_json(self, file=None):
    if file is None:
        return json.dumps(self, default=serialize)
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(self, f, default=serialize, ensure_ascii=False, indent=4)

# https://stackoverflow.com/questions/279561/what-is-the-python-equivalent-of-static-variables-inside-a-function
def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate
