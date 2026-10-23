from typing import Literal

import numpy as np
import om4.utils.units as units

MU_earth = 3.986004415e14  # m**3/s**2
DU_earth_m = 6378136.3  # m
TU_earth_s = np.sqrt(DU_earth_m ** 3 / MU_earth)

units.add_unit('TU_earth', f'{TU_earth_s}*s')
units.add_unit('DU_earth', f'{DU_earth_m}*m')

#: The central bodies the orbital equations accept. Each needs DU_<body> and TU_<body> units.
CentralBody = Literal['earth', 'sun', 'moon']


def body_units(u, central_body):
    """
    Return a units string with its canonical DU/TU placeholders replaced by the central body's.

    For example, ``body_units('DU/TU**2', 'earth')`` is ``'DU_earth/TU_earth**2'``. Components
    declare their units with the placeholders and resolve them in ``model_post_init``, since the
    placeholders themselves are not valid units.
    """
    if u is None:
        return None
    return u.replace('DU', f'DU_{central_body}').replace('TU', f'TU_{central_body}')


def set_body_units(system, units_map):
    """Set the units of a component's variables from ``{name: placeholder units}``."""
    for name, u in units_map.items():
        var = system.inputs.get(name) or system.outputs[name]
        var.units = body_units(u, system.central_body)
