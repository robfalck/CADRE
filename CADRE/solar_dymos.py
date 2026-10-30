"""
Solar discipline for CADRE, on om4: the exposed area of each solar cell.

The exposed area is interpolated from ``exposed_area_table()``, the legacy MBI fit of the
ray-traced areas (ADL-007), with om4's ``StructuredMetaModelComp`` (OpenMDAO/om4 PR #404,
ai/OM4_NEEDS.md #3).
"""
import jax.numpy as jnp
from pydantic import Field, model_validator
from pydantic_core import PydanticUndefined

from om4.api import Group, InputVar, JaxExplicitComp, OutputVar, StructuredMetaModelComp
from om4.core.system import System
from om4.utils.polymorphic import Polymorphic
from om4.utils.promotion import PromotesSpec

from CADRE.tables import exposed_area_table


class ExposedAreaTransposeComp(JaxExplicitComp):
    """
    Move the node axis of the interpolated exposed area last (component-major).
    """
    num_nodes: int = Field(default=1)

    inputs: dict[str, InputVar] = {
        'exposed_area_nodes': InputVar(shape=('num_nodes', 7, 12), val=0.0, units='m**2',
                                       desc='Exposed area of each solar cell, node axis first'),
    }

    outputs: dict[str, OutputVar] = {
        'exposedArea': OutputVar(shape=(7, 12, 'num_nodes'), units='m**2',
                                 desc='Exposed area to sun for each solar cell over time'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        outputs['exposedArea'] = jnp.moveaxis(inputs['exposed_area_nodes'], 0, -1)


def _exposed_area_subsystems(num_nodes, method):
    nn = num_nodes
    grid, area = exposed_area_table()
    return {
        'table': StructuredMetaModelComp(
            grid={'finAngle': grid['finAngle'], 'azimuth': grid['azimuth'],
                  'elevation': grid['elevation']},
            training_outputs={'exposed_area_nodes': area}, method=method, vec_size=nn,
            # The table takes a fin angle per point; the one static fin angle is broadcast to
            # them by the promotion below.
            inputs={'finAngle': InputVar(shape=(nn,), units='rad', desc='Fin angle of solar panel',
                                         tags={'dymos.static_target'}),
                    'azimuth': InputVar(shape=(nn,), units='rad',
                                        desc='Azimuth angle of the sun in the body-fixed frame'),
                    'elevation': InputVar(shape=(nn,), units='rad',
                                          desc='Elevation angle of the sun in the body-fixed frame')},
            outputs={'exposed_area_nodes': OutputVar(shape=(nn, 7, 12), units='m**2')},
            promotes_inputs=[PromotesSpec(patterns=['finAngle'], src_indices=[0] * nn, src_shape=(1,)),
                             'azimuth', 'elevation'],
            promotes_outputs=['exposed_area_nodes']),
        'transpose': ExposedAreaTransposeComp(num_nodes=nn, promotes_inputs=['exposed_area_nodes'],
                                              promotes_outputs=['exposedArea']),
    }


class ExposedAreaGroup(Group):
    """
    Exposed area of each of the 7 cells on each of the 12 panels (``Solar_ExposedArea``).

    Inputs ``finAngle`` (static), and the sun's ``azimuth`` on [0, 2 pi] and ``elevation`` on
    [0, pi] in the body frame (``SunPositionSphericalComp``); output ``exposedArea``,
    ``(7, 12, num_nodes)``. The legacy component evaluated MBI's fit of the table; this
    interpolates that fit, sampled by ``exposed_area_table()``, to about 2e-4 of its range.

    ``method`` is the ``StructuredMetaModelComp`` method. The default, ``'lagrange3'``, is
    complex-step safe, as CADRE's ODE checks need; ``'cubic'`` is more accurate (1e-4) and
    smoother, but not complex-step safe.
    """
    num_nodes: int = Field(default=1)
    method: str = Field(default='lagrange3', description='Interpolation method of the table.')

    # Derived from the fields above; not part of the constructor API.
    subsystems: dict[str, Polymorphic[System]] = Field(default=PydanticUndefined, init=False)

    @model_validator(mode='before')
    @classmethod
    def _build_from_fields(cls, data):
        if not isinstance(data, dict):
            return data
        data = dict(data)
        data['subsystems'] = _exposed_area_subsystems(data.get('num_nodes', 1),
                                                      data.get('method', 'lagrange3'))
        return data
