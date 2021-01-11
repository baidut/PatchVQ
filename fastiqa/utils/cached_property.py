try:
    from cached_property import cached_property
except ImportError:
    class cached_property(object):
        '''Computes attribute value and caches it in the instance.
        Python Cookbook (Denis Otkidach) https://stackoverflow.com/users/168352/denis-otkidach
        This decorator allows you to create a property which can be computed once and
        accessed many times. Sort of like memoization.
        '''

        def __init__(self, method, name=None):
            # record the unbound-method and the name
            self.method = method
            self.name = name or method.__name__
            self.__doc__ = method.__doc__

        def __get__(self, inst, cls):
            # self: <__main__.cache object at 0xb781340c>
            # inst: <__main__.Foo object at 0xb781348c>
            # cls: <class '__main__.Foo'>
            if inst is None:
                # instance attribute accessed on class, return self
                # You get here if you write `Foo.bar`
                return self
            # compute, cache and return the instance's attribute value
            result = self.method(inst)
            # setattr redefines the instance's attribute so this doesn't get called again
            setattr(inst, self.name, result)
            return result

