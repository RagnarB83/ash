"""A Quantity trait using simtk.unit
"""
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------


from simtk import unit
from .stringunits import str_to_unit
from .IPython.traitlets import Float, TraitError, class_of
unit.nm = unit.nanometers
unit.A = unit.angstroms
unit.ps = unit.picoseconds
unit.fs = unit.femtoseconds
unit.atm = unit.atmosphere


#-----------------------------------------------------------------------------
# Dirty Hacks
#-----------------------------------------------------------------------------


# add unit aliases for nm and ps, so that on
# the command line, the user can write dt=0.002*ps and
# it will get parsed correctly
def _Quantity_str(self):
    """Change the __repr__ and __str__ on Quantity so that
    the displayed output is like '1*nanometers'. This ensures
    that the config file is valid python, since it uses repr,
    and that the displayed helptext looks right"""
    unit = self.unit.get_name()
    if not unit.startswith('/'):
        unit = '*' + unit
    return str(self._value) + unit

unit.Quantity.__str__ = _Quantity_str
unit.Quantity.__repr__ = _Quantity_str

#-----------------------------------------------------------------------------
# Classes
#-----------------------------------------------------------------------------


class Quantity(Float):
    '''A united float trait.'''

    def __init__(self, default_value, **metadata):
        if not isinstance(default_value, unit.Quantity):
            raise ValueError('default value must have units.')
        self.unit = default_value.unit
        super(Quantity, self).__init__(default_value, **metadata)

    def validate(self, obj, value):
        if isinstance(value, basestring):
            try:
                value = str_to_unit(value)
            except ValueError as e:
                raise TraitError(e.message)

        if isinstance(value, unit.Quantity):
            if value.unit.is_compatible(self.unit):
                return value
            else:
                self.error_units(obj, value)

        if isinstance(value, int) or isinstance(value, float):
            e = ("The '{name}' trait of {class_of} must have units of {unit}, "
                 "but a value without units, {value}, was specified. To "
                 "specify units, use a syntax like --{name}={default_value} "
                 "on the command line, or {name} = "
                 "{default_value} in the config file.").format(
                    name=self.name, class_of=class_of(obj), unit=self.unit,
                    value=value, default_value=self.default_value,
                    class_name=obj.__class__.__name__)

            raise TraitError(e)

        self.error(obj, value)

    def error_units(self, obj, value):
        e = ("The '%s' trait of %s must have units of %s, but a value in units"
             " of %s was specified.") % (self.name, class_of(obj), self.unit,
                                         value.unit)
        raise TraitError(e)
