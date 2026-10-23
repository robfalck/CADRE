from pydantic import Field

from om4.api import ExplicitComponent, InputVar, OutputVar, PartialsSpec

from CADRE.orbital_equations import CentralBody, set_body_units

#: The perturbing accelerations summed into each direction.
_SOURCES = {
    'a_r': ('T_r', 'J2_r', 'D_r', 'sb_r'),
    'a_s': ('T_s', 'J2_s', 'D_s', 'sb_s'),
    'a_w': ('T_w', 'J2_w', 'sb_w'),
}


class TotalPerturbation(ExplicitComponent):
    num_nodes: int = Field(default=1)
    central_body: CentralBody = Field(default='earth')

    inputs: dict[str, InputVar] = {
        name: InputVar(shape=('num_nodes',), val=0.0)
        for name in ('T_r', 'T_s', 'T_w', 'J2_r', 'J2_s', 'J2_w', 'D_r', 'D_s',
                     'sb_r', 'sb_s', 'sb_w')
    }

    outputs: dict[str, OutputVar] = {
        name: OutputVar(shape=('num_nodes',), val=0.0) for name in _SOURCES
    }

    partials: list[PartialsSpec] = [
        PartialsSpec(of=of, wrt=wrt, diagonal=True, val=1.0)
        for of, wrts in _SOURCES.items() for wrt in wrts
    ]

    def model_post_init(self, context, /):
        super().model_post_init(context)
        self.inputs['J2_r'].desc = 'zonal gravity perturbation in the radial direction'
        self.inputs['J2_s'].desc = 'zonal gravity perturbation in the tangential direction'
        self.inputs['J2_w'].desc = 'zonal gravity perturbation in the normal direction'
        set_body_units(self, {name: 'DU/TU**2' for name in (*self.inputs, *self.outputs)})

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        for of, wrts in _SOURCES.items():
            outputs[of] = sum(inputs[wrt] for wrt in wrts)
