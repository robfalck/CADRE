import numpy as np
from pydantic import Field

from om4.api import ExplicitComponent, InputVar, OutputVar, PartialsSpec

from CADRE.orbital_equations import CentralBody, body_units as _body_units

_STATES = ('p', 'f', 'g', 'h', 'k', 'L')
_STATE_UNITS = {'p': 'DU', 'f': 'unitless', 'g': 'unitless', 'h': 'unitless', 'k': 'unitless',
                'L': 'rad'}
_RATE_UNITS = {'p': 'DU/TU', 'f': '1/TU', 'g': '1/TU', 'h': '1/TU', 'k': '1/TU', 'L': 'rad/TU'}

_PARTIALS = {
    'p_dot': ('p', 'f', 'g', 'L', 'a_s'),
    'f_dot': ('p', 'f', 'g', 'h', 'k', 'L', 'a_r', 'a_s', 'a_w'),
    'g_dot': ('p', 'f', 'g', 'h', 'k', 'L', 'a_r', 'a_s', 'a_w'),
    'h_dot': ('p', 'f', 'g', 'h', 'k', 'L', 'a_w'),
    'k_dot': ('p', 'f', 'g', 'h', 'k', 'L', 'a_w'),
    'L_dot': ('p', 'f', 'g', 'h', 'k', 'L', 'a_w'),
}


class TwoBodyDynamicsComp(ExplicitComponent):
    """
    The equations of motion in modified equinoctial elements, in canonical units (mu = 1).

    The elements are tagged as dymos4 state targets and their rates as state rate sources, so a
    phase discovers the six states from this component alone.
    """
    num_nodes: int = Field(default=1)
    central_body: CentralBody = Field(default='earth')

    inputs: dict[str, InputVar] = {
        # Modified equinoctial elements
        **{s: InputVar(shape=('num_nodes',), val=1.0, tags={f'dymos.state_target:{s}'})
           for s in _STATES},
        # Perturbations
        'a_r': InputVar(shape=('num_nodes',), val=0.0, desc='perturbations in radial direction'),
        'a_s': InputVar(shape=('num_nodes',), val=0.0, desc='perturbations in tangent direction'),
        'a_w': InputVar(shape=('num_nodes',), val=0.0, desc='perturbations in normal direction'),
    }

    outputs: dict[str, OutputVar] = {
        f'{s}_dot': OutputVar(shape=('num_nodes',), val=1.0,
                              tags={f'dymos.state_rate_source:{s}'})
        for s in _STATES
    }

    partials: list[PartialsSpec] = [
        PartialsSpec(of=of, wrt=wrt, diagonal=True)
        for of, wrts in _PARTIALS.items() for wrt in wrts
    ]

    def model_post_init(self, context, /):
        super().model_post_init(context)
        cb = self.central_body
        self.inputs['p'].desc = 'semi-latus rectum'
        self.inputs['L'].desc = 'true longitude'
        for s in _STATES:
            self.inputs[s].units = _body_units(_STATE_UNITS[s], cb)
            out = self.outputs[f'{s}_dot']
            out.units = _body_units(_RATE_UNITS[s], cb)
            out.tags = out.tags | {f'dymos.state_units:{self.inputs[s].units}'}
        for a in ('a_r', 'a_s', 'a_w'):
            self.inputs[a].units = _body_units('DU/TU**2', cb)

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        p = inputs['p']
        f = inputs['f']
        g = inputs['g']
        h = inputs['h']
        k = inputs['k']
        L = inputs['L']

        a_r = inputs['a_r']
        a_s = inputs['a_s']
        a_w = inputs['a_w']

        cL = np.cos(L)
        sL = np.sin(L)

        w = 1 + f * cL + g * sL
        s_sq = 1 + h ** 2 + k ** 2
        sqrt_p_mu = np.sqrt(p)

        outputs['p_dot'] = 2 * p * a_s * sqrt_p_mu / w
        outputs['f_dot'] = sqrt_p_mu * (a_r * sL + ((w + 1) * cL + f) * a_s / w - (h * sL - k * cL) * g * a_w / w)
        outputs['g_dot'] = sqrt_p_mu * (-a_r * cL + ((w + 1) * sL + g) * a_s / w + (h * sL - k * cL) * f * a_w / w)
        outputs['h_dot'] = sqrt_p_mu * s_sq * a_w * cL / (2 * w)
        outputs['k_dot'] = sqrt_p_mu * s_sq * a_w * sL / (2 * w)
        outputs['L_dot'] = sqrt_p_mu * (h * sL - k * cL) * a_w / w + np.sqrt(p) * (w / p)**2

    def compute_partials(self, inputs, partials):

        p = inputs['p']
        f = inputs['f']
        g = inputs['g']
        h = inputs['h']
        k = inputs['k']
        L = inputs['L']

        a_r = inputs['a_r']
        a_s = inputs['a_s']
        a_w = inputs['a_w']

        cL = np.cos(L)
        sL = np.sin(L)
        sqrt_p_mu = np.sqrt(p)

        w = 1 + f * cL + g * sL
        s_sq = 1 + h ** 2 + k ** 2

        partials['p_dot', 'p'] = 3 * sqrt_p_mu * a_s / w
        partials['p_dot', 'f'] = -2 * p * a_s * sqrt_p_mu / (w ** 2) * cL
        partials['p_dot', 'g'] = -2 * p * a_s * sqrt_p_mu / (w ** 2) * sL
        partials['p_dot', 'L'] = -2 * p * a_s * sqrt_p_mu / w ** 2 * (g * cL - f * sL)
        partials['p_dot', 'a_s'] = 2 * p * sqrt_p_mu / w

        partials['f_dot', 'p'] = np.sqrt(1 / p) * (a_r * sL + ((w + 1) * cL + f) * a_s / w -
                                                   (h * sL - k * cL) * g * a_w / w) / 2
        partials['f_dot', 'f'] = sqrt_p_mu * (a_s * sL * (g + sL) + (h * sL - k * cL) * g * a_w * cL) / (w ** 2)
        partials['f_dot', 'g'] = sqrt_p_mu * (-a_s * sL * (f + cL) - (h * sL - k * cL) * a_w * (1 + f * cL)) / (w ** 2)
        partials['f_dot', 'h'] = -sqrt_p_mu * sL * g * a_w / w
        partials['f_dot', 'k'] = sqrt_p_mu * cL * g * a_w / w
        partials['f_dot', 'L'] = sqrt_p_mu * (a_r * cL + a_s * (w * (cL * (g * cL - f * sL) - sL * (1 + w)) -
                                                                (g * cL - f * sL) * ((1 + w) * cL + f)) / (w ** 2)
                                              - a_w * g * (f * h + g * k + h * cL + k * sL) / (w ** 2))
        partials['f_dot', 'a_r'] = sqrt_p_mu * sL
        partials['f_dot', 'a_s'] = sqrt_p_mu * ((w + 1) * cL + f) / w
        partials['f_dot', 'a_w'] = -sqrt_p_mu * (h * sL - k * cL) * g / w

        partials['g_dot', 'p'] = np.sqrt(1 / p) * (-a_r * cL + ((w + 1) * sL + g) * a_s / w +
                                                   (h * sL - k * cL) * f * a_w / w) / 2
        partials['g_dot', 'f'] = sqrt_p_mu * ((h * sL - k * cL) * a_w * (w - f * cL) - a_s * cL * (g + sL)) / (w ** 2)
        partials['g_dot', 'g'] = sqrt_p_mu * ((cL * (f + cL)) * a_s - (h * sL - k * cL) * a_w * f * sL) / (w ** 2)
        partials['g_dot', 'h'] = sqrt_p_mu * sL * f * a_w / w
        partials['g_dot', 'k'] = -sqrt_p_mu * cL * f * a_w / w
        partials['g_dot', 'L'] = sqrt_p_mu * (a_r * sL + a_s * (w * (sL * (g * cL - f * sL) + cL * (1 + w)) -
                                                                (g * cL - f * sL) * ((1 + w) * sL + g)) / (w ** 2)
                                              + a_w * f * (f * h + g * k + h * cL + k * sL) / (w ** 2))
        partials['g_dot', 'a_r'] = -sqrt_p_mu * cL
        partials['g_dot', 'a_s'] = sqrt_p_mu * ((w + 1) * sL + g) / w
        partials['g_dot', 'a_w'] = sqrt_p_mu * (h * sL - k * cL) * f / w

        partials['h_dot', 'p'] = np.sqrt(1 / p) * s_sq * a_w * cL / (4 * w)
        partials['h_dot', 'f'] = -0.5 * sqrt_p_mu * s_sq * a_w * cL * cL / (w ** 2)
        partials['h_dot', 'g'] = -0.5 * sqrt_p_mu * s_sq * a_w * cL * sL / (w ** 2)
        partials['h_dot', 'h'] = sqrt_p_mu * h * a_w * cL / w
        partials['h_dot', 'k'] = sqrt_p_mu * k * a_w * cL / w
        partials['h_dot', 'L'] = -0.5 * sqrt_p_mu * s_sq * a_w * (g + sL) / (w ** 2)
        partials['h_dot', 'a_w'] = sqrt_p_mu * s_sq * cL / (2 * w)

        partials['k_dot', 'p'] = np.sqrt(1 / p) * s_sq * a_w * sL / (4 * w)
        partials['k_dot', 'f'] = -0.5 * sqrt_p_mu * s_sq * a_w * sL * cL / (w ** 2)
        partials['k_dot', 'g'] = -0.5 * sqrt_p_mu * s_sq * a_w * sL * sL / (w ** 2)
        partials['k_dot', 'h'] = sqrt_p_mu * h * a_w * sL / w
        partials['k_dot', 'k'] = sqrt_p_mu * k * a_w * sL / w
        partials['k_dot', 'L'] = 0.5 * sqrt_p_mu * s_sq * a_w * (f + cL) / (w ** 2)
        partials['k_dot', 'a_w'] = sqrt_p_mu * s_sq * sL / (2 * w)

        partials['L_dot', 'p'] = 0.5 * np.sqrt(1 / p) * (h * sL - k * cL) * a_w / w\
            - 1.5 * np.sqrt(1 / p**5) * w**2
        partials['L_dot', 'f'] = -sqrt_p_mu * (h * sL - k * cL) * a_w * cL / w**2 + 2 * np.sqrt(p) * w * cL / p**2
        partials['L_dot', 'g'] = -sqrt_p_mu * (h * sL - k * cL) * a_w * sL / w**2 + 2 * np.sqrt(p) * w * sL / p**2
        partials['L_dot', 'h'] = sqrt_p_mu * sL * a_w / w
        partials['L_dot', 'k'] = -sqrt_p_mu * cL * a_w / w
        partials['L_dot', 'L'] = sqrt_p_mu * ((h * cL + k * sL) - (h * sL - k * cL) * (g * cL - f * sL) / w) * a_w / w\
            + 2 * np.sqrt(p) * w * (g * cL - f * sL) / p**2
        partials['L_dot', 'a_w'] = sqrt_p_mu * (h * sL - k * cL) / w
