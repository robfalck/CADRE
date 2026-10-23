from pydantic import Field, model_validator
from pydantic_core import PydanticUndefined

from om4.api import Group
from om4.core.system import System
from om4.utils.polymorphic import Polymorphic

from CADRE.orbital_equations import CentralBody
from CADRE.orbital_equations.EOMs.perturbations.total_perturbation import TotalPerturbation
from CADRE.orbital_equations.EOMs.perturbations.spherical_harmonics_comp import SphericalHarmonicsComp
# from CADRE.EOMs.perturbations.secondary_body.secondary_body_comp import SecondaryBodyEffect
from CADRE.orbital_equations.frame_conversions.r_comp import RComp


def _perturbations_subsystems(num_nodes, central_body):
    nn = num_nodes
    cb = central_body
    return {
        'r_comp': RComp(num_nodes=nn, central_body=cb,
                        promotes_inputs=['p', 'f', 'g', 'L'], promotes_outputs=['r']),
        'spherical_harmonics': SphericalHarmonicsComp(
            num_nodes=nn, central_body=cb,
            promotes_inputs=['h', 'k', 'L', 'r'], promotes_outputs=['J2_r', 'J2_s', 'J2_w']),
        # 'atmospheric_drag': AerodynamicDrag(...), promotes_inputs=['p', 'f', 'g', 'L', 'r', 'v'],
        #                     promotes_outputs=['D_r', 'D_s']
        # 'secondary_body': SecondaryBodyEffect(...),
        #                   promotes_inputs=['time', 'x', 'y', 'z', 'vx', 'vy', 'vz'],
        #                   promotes_outputs=['sb_r', 'sb_s', 'sb_w']
        'total_perturbation': TotalPerturbation(
            num_nodes=nn, central_body=cb,
            promotes_inputs=['J2_r', 'J2_s', 'J2_w', 'T_r', 'T_s', 'T_w',
                             'D_r', 'D_s', 'sb_r', 'sb_s', 'sb_w'],
            promotes_outputs=['a_r', 'a_s', 'a_w']),
    }


class Perturbations(Group):
    """The perturbing accelerations on the orbit, in the radial/tangential/normal frame."""
    num_nodes: int = Field(default=1)
    central_body: CentralBody = Field(default='earth')

    # Derived from the fields above; not part of the constructor API.
    subsystems: dict[str, Polymorphic[System]] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode='before')
    @classmethod
    def _build_from_fields(cls, data):
        if not isinstance(data, dict):
            return data
        data = dict(data)
        data['subsystems'] = _perturbations_subsystems(data.get('num_nodes', 1),
                                                       data.get('central_body', 'earth'))
        return data
