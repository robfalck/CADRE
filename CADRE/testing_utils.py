"""Helpers shared by the om4/dymos4 component tests."""
import jax
import numpy as np

from om4.api import Group, Problem
from om4.utils.promotion import normalize_promotes_list

# Tests call the components' jax functions directly, outside om4's enable_x64 context.
jax.config.update("jax_enable_x64", True)


def wrap(system):
    """Return a Problem with ``system`` as its only subsystem, ``c``, fully promoted."""
    system.promotes_inputs = normalize_promotes_list(['*'])
    system.promotes_outputs = normalize_promotes_list(['*'])
    return Problem(model=Group(subsystems={'c': system}))


def assert_partials_ok(results, tol=1e-6):
    """Assert every checked partial matches its reference to within tol, absolute or relative."""
    assert results, 'check_partials returned no results'
    for comp, pairs in results.items():
        for key, data in pairs.items():
            abs_err = np.max(np.abs(np.asarray(data['abs_error'])))
            rel_arr = np.asarray(data['rel_error'], dtype=float)
            finite = rel_arr[np.isfinite(rel_arr)]
            rel_err = np.max(np.abs(finite)) if finite.size else np.inf
            assert min(abs_err, rel_err) <= tol, (
                f'{comp} {key}: abs={abs_err}, rel={rel_err} exceed {tol}'
            )
