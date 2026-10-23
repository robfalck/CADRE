import numpy as np
from pydantic import Field

from om4.api import ExplicitComponent, InputVar, OutputVar, PartialsSpec

from CADRE.orbital_equations import CentralBody, set_body_units


class RComp(ExplicitComponent):
    num_nodes: int = Field(default=1)
    central_body: CentralBody = Field(default='earth')

    inputs: dict[str, InputVar] = {
        'p': InputVar(shape=('num_nodes',), val=1.0, desc='semi-latus rectum'),
        'f': InputVar(shape=('num_nodes',), val=1.0),
        'g': InputVar(shape=('num_nodes',), val=1.0),
        'L': InputVar(shape=('num_nodes',), val=1.0, desc='true longitude'),
    }

    outputs: dict[str, OutputVar] = {
        'r': OutputVar(shape=('num_nodes',), val=1.0, desc='geocentric radius'),
    }

    partials: list[PartialsSpec] = [
        PartialsSpec(of='r', wrt=wrt, diagonal=True) for wrt in ('p', 'f', 'g', 'L')
    ]

    def model_post_init(self, context, /):
        super().model_post_init(context)
        set_body_units(self, {'p': 'DU', 'f': 'unitless', 'g': 'unitless', 'L': 'rad',
                              'r': 'DU'})

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        p = inputs['p']
        f = inputs['f']
        g = inputs['g']
        L = inputs['L']

        outputs['r'] = p / (1 + f * np.cos(L) + g * np.sin(L))

    def compute_partials(self, inputs, partials):
        p = inputs['p']
        f = inputs['f']
        g = inputs['g']
        L = inputs['L']

        partials['r', 'p'] = 1 / (1 + f * np.cos(L) + g * np.sin(L))
        partials['r', 'f'] = -p * np.cos(L) / ((1 + f * np.cos(L) + g * np.sin(L)) ** 2)
        partials['r', 'g'] = -p * np.sin(L) / ((1 + f * np.cos(L) + g * np.sin(L)) ** 2)
        partials['r', 'L'] = p * (f * np.sin(L) - g * np.cos(L)) / ((1 + f * np.cos(L) + g * np.sin(L)) ** 2)
