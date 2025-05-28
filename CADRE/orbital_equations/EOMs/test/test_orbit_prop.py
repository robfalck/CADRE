import unittest

import numpy as np
import openmdao.api as om
import dymos as dm

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from CADRE.orbital_equations.EOMs import TwoBodyDynamicsComp


@use_tempdirs
class TestOrbitProp(unittest.TestCase):

    def test_propagation(self):
        prob = om.Problem(model=om.Group())
        prob.driver = om.pyOptSparseDriver()
        prob.driver.declare_coloring()
        prob.driver.options['print_results'] = False
        prob.driver.options['optimizer'] = 'IPOPT'
        prob.driver.opt_settings['hessian_approximation'] = 'limited-memory'
        prob.driver.opt_settings['print_level'] = 0
        prob.driver.opt_settings['tol'] = 1.0E-6
        prob.driver.opt_settings['linear_solver'] = 'mumps'
        prob.driver.declare_coloring()

        traj = prob.model.add_subsystem('traj', dm.Trajectory())
        orbit = traj.add_phase('orbit', dm.Phase(ode_class=TwoBodyDynamicsComp,
                                                 transcription=dm.Birkhoff(num_nodes=20),
                                                 ode_init_kwargs={'central_body': 'earth'}))

        orbit.set_time_options(fix_initial=True, fix_duration=True, units='TU_earth', initial_val=0.0)
        orbit.add_state('p', fix_initial=True, units='DU_earth', rate_source='p_dot', lower=0.5)
        orbit.add_state('f', fix_initial=True, units=None, rate_source='f_dot')
        orbit.add_state('g', fix_initial=True, units=None, rate_source='g_dot')
        orbit.add_state('h', fix_initial=True, units=None, rate_source='h_dot')
        orbit.add_state('k', fix_initial=True, units=None, rate_source='k_dot')
        orbit.add_state('L', fix_initial=True, units='rad', rate_source='L_dot')
        orbit.add_objective('time', loc='final')

        prob.setup()

        orbit.set_time_val(duration=8, units='TU_earth')
        orbit.set_state_val('p', (4/np.pi)**(2/3), units='DU_earth')
        orbit.set_state_val('f', 0)
        orbit.set_state_val('g', 0)
        orbit.set_state_val('h', 0)
        orbit.set_state_val('k', 0)
        orbit.set_state_val('L', 0)

        dm.run_problem(prob)

        p = prob.get_val('traj.orbit.timeseries.p')
        f = prob.get_val('traj.orbit.timeseries.f')
        g = prob.get_val('traj.orbit.timeseries.g')
        h = prob.get_val('traj.orbit.timeseries.h')
        k = prob.get_val('traj.orbit.timeseries.k')
        L = prob.get_val('traj.orbit.timeseries.L')

        assert_near_equal(p[0], p[-1], tolerance=1e-6)
        assert_near_equal(f, np.zeros((20, 1)), tolerance=1e-6)
        assert_near_equal(g, np.zeros((20, 1)), tolerance=1e-6)
        assert_near_equal(h, np.zeros((20, 1)), tolerance=1e-6)
        assert_near_equal(k, np.zeros((20, 1)), tolerance=1e-6)
        assert_near_equal(L[-1], 2*np.pi, tolerance=1e-6)

