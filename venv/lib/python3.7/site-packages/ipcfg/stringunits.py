"""Parse units from strings with simtk.unit

The function str_to_unit can retreive a complex unit or Quantity from a string.
"""
#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------

import ast
from simtk import unit
__all__ = ['str_to_unit']

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------


class _UnitContext(ast.NodeTransformer):

    '''Node transformer for an AST hack that turns raw strings into
    complex simt.unit.Unit expressions. See _str_to_unit for how this
    is used -- it's not really meant to stand on its own
    '''
    # we want to do some validation to ensure that the AST only
    # contains "safe" operations. These are operations that can reasonably
    # appear in unit expressions
    allowed_ops = [ast.Expression, ast.BinOp, ast.Name,
                   ast.Pow, ast.Div, ast.Mult, ast.Num]

    def visit(self, node):
        if not any(isinstance(node, a) for a in self.allowed_ops):
            raise ValueError('Invalid unit expression. Contains dissallowed '
                             'operation %s' % node.__class__.__name__)
        return super(_UnitContext, self).visit(node)

    def visit_Name(self, node):
        # we want to prefix all names to look like unit.nanometers instead
        # of just "nanometers", because I don't want to import * from
        # units into this module.
        if not hasattr(unit, node.id):
            # also, let's take this opporunity to check that the node.id
            # (which supposed to be the name of the unit, like "nanometers")
            # is actually an attribute in simtk.unit
            raise ValueError('%s is not a valid unit' % node.id)

        return ast.Attribute(value=ast.Name(id='unit', ctx=ast.Load()),
                             attr=node.id, ctx=ast.Load())
_unit_context = _UnitContext()  # global instance of the visitor


def str_to_unit(unit_string):
    '''eval() based transformer that extracts a simtk.unit object
    from a string description.

    Parameters
    ----------
    unit_string : str
        string description of a unit. this may contain expressions with
        multiplication, division, powers, etc.

    Examples
    --------
    >>> type(str_to_unit('nanometers**2/meters*gigajoules'))
    <class 'simtk.unit.unit.Unit'>
    >>> str(str_to_unit('nanometers**2/meters*gigajoules'))
    'nanometer**2*gigajoule/meter'

    '''
    # parse the string with the ast, and then run out unit context
    # visitor on it, which will basically change bare names like
    # "nanometers" into "unit.nanometers" and simulataniously check that
    # there's no nefarious stuff in the expression.

    node = _unit_context.visit(ast.parse(unit_string, mode='eval'))
    fixed_node = ast.fix_missing_locations(node)
    output = eval(compile(fixed_node, '<string>', mode='eval'))

    return output
