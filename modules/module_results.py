#
#https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute


#Class that behaves as a dict but attributes can also be found like:
#results = Results(energy=1.343, gradient=[3.0, 43., 43.0])
#results["energy"] and results.energy
class Results(dict):
    def __init__(self, *args, **kwargs):
        super(Results, self).__init__(*args, **kwargs)
        self.__dict__ = self

#Or try dataclasses https://realpython.com/python-data-classes/