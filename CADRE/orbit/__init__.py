import numpy as np
import openmdao.utils.units as units

MU_earth = 3.986004415e14  # m**3/s**2
DU_earth_m = 6378136.3  # m
TU_earth_s = np.sqrt(DU_earth_m ** 3 / MU_earth)

units.add_unit('TU_earth', f'{TU_earth_s}*s')
units.add_unit('DU_earth', f'{DU_earth_m}*m')
