from pydantic import Field, model_validator
from pydantic_core import PydanticUndefined

from om4.api import Group
from om4.core.system import System
from om4.utils.polymorphic import Polymorphic

from CADRE.orbital_equations import CentralBody
from CADRE.orbital_equations.EOMs.two_body_dynamics_comp import TwoBodyDynamicsComp
from CADRE.orbital_equations.EOMs.perturbations.perturbations import Perturbations
from CADRE.orbital_equations.frame_conversions.mee_to_cart import MEEToCart


def _two_body_eom_subsystems(num_nodes, central_body):
    nn = num_nodes
    cb = central_body
    return {
        'mee_to_cart': MEEToCart(num_nodes=nn, central_body=cb,
                                 promotes_inputs=['p', 'f', 'g', 'h', 'k', 'L'],
                                 promotes_outputs=['x', 'y', 'z', 'vx', 'vy', 'vz']),
        # 'propulsion': PropulsionMEE(...), promotes_inputs=['m', 'throttle', 'u_r', 'u_s', 'u_w'],
        #               promotes_outputs=['T_r', 'T_s', 'T_w', 'm_dot']
        'perturbations': Perturbations(num_nodes=nn, central_body=cb,
                                       promotes_inputs=['p', 'f', 'g', 'h', 'k', 'L',
                                                        'T_r', 'T_s', 'T_w'],
                                       promotes_outputs=['a_r', 'a_s', 'a_w']),
        'spacecraft_dynamics': TwoBodyDynamicsComp(
            num_nodes=nn, central_body=cb,
            promotes_inputs=['p', 'f', 'g', 'h', 'k', 'L', 'a_r', 'a_s', 'a_w'],
            promotes_outputs=['p_dot', 'f_dot', 'g_dot', 'h_dot', 'k_dot', 'L_dot']),
    }


class TwoBodyEOM(Group):
    """
    The perturbed two-body equations of motion in modified equinoctial elements.

    The original also took an ``sc_params`` dict, which nothing used; it is dropped here.
    """
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
        data['subsystems'] = _two_body_eom_subsystems(data.get('num_nodes', 1),
                                                      data.get('central_body', 'earth'))
        return data
