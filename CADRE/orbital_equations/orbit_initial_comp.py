import jax.numpy as jnp
from om4.api import InputVar, JaxExplicitComp, OutputVar

#: Earth's radius used for the perigee and apogee altitudes, km.
Re = 6378.137


def _static(val, units, desc):
    return InputVar(shape=(1,), val=val, units=units, desc=desc)


class OrbitInitialComp(JaxExplicitComp):
    """
    Computes the initial orbit, as modified equinoctial elements, from Keplerian elements.

    Replaces the legacy ``Orbit_Initial``, which computed the initial Cartesian position and
    velocity. The orbit here is propagated in modified equinoctial elements, so the outputs
    connect to the orbit phase's ``initial_states_*``. They have no node axis. ``p`` is in km;
    om4 converts it to the state's units on connection.

    The legacy inputs were unitless degrees; here the angles are declared in degrees and the
    altitudes in km.
    """
    inputs: dict[str, InputVar] = {
        'altPerigee': _static(500., 'km', 'Altitude of perigee'),
        'altApogee': _static(500., 'km', 'Altitude of apogee'),
        'RAAN': _static(66.279, 'deg', 'Right ascension of the ascending node'),
        'Inc': _static(82.072, 'deg', 'Inclination'),
        'argPerigee': _static(0.0, 'deg', 'Argument of perigee'),
        'trueAnomaly': _static(337.987, 'deg', 'True anomaly'),
    }

    outputs: dict[str, OutputVar] = {
        'p0': OutputVar(shape=(1,), units='km', desc='Initial semi-latus rectum'),
        'f0': OutputVar(shape=(1,), units='unitless'),
        'g0': OutputVar(shape=(1,), units='unitless'),
        'h0': OutputVar(shape=(1,), units='unitless'),
        'k0': OutputVar(shape=(1,), units='unitless'),
        'L0': OutputVar(shape=(1,), units='rad', desc='Initial true longitude'),
    }

    def compute_outputs(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None,
                        statics=None):
        d2r = jnp.pi / 180.
        r_perigee = Re + inputs['altPerigee']
        r_apogee = Re + inputs['altApogee']
        e = (r_apogee - r_perigee) / (r_apogee + r_perigee)
        a = (r_perigee + r_apogee) / 2
        Om = d2r * inputs['RAAN']
        i = d2r * inputs['Inc']
        w = d2r * inputs['argPerigee']
        nu = d2r * inputs['trueAnomaly']

        outputs['p0'] = a * (1 - e**2)
        outputs['f0'] = e * jnp.cos(w + Om)
        outputs['g0'] = e * jnp.sin(w + Om)
        outputs['h0'] = jnp.tan(i / 2) * jnp.cos(Om)
        outputs['k0'] = jnp.tan(i / 2) * jnp.sin(Om)
        outputs['L0'] = Om + w + nu
