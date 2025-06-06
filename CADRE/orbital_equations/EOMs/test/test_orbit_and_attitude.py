import unittest

import numpy as np
import openmdao.api as om
import dymos as dm

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from CADRE.orbital_equations.EOMs import TwoBodyDynamicsComp
from CADRE.orbital_equations.frame_conversions import MEEToCart
from CADRE.attitude_dymos import AngularAccelerationComp, AngularVelocityComp, AttitudeTorqueComp, \
    BodyVelComp, OBRComp, OBIComp, ORIComp, OdotBIComp
from CADRE.orbital_equations.frame_conversions import StateMuxComp



# @use_tempdirs
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

        class TwoBodyODE(om.Group):

            def initialize(self):
                self.options.declare('num_nodes', types=int)
                self.options.declare('central_body', types=str)
                self.options.declare('vectorize_outputs', types=bool, default=False)
                self.options.declare('grid_data')

            def setup(self):
                nn = self.options['num_nodes']
                cb = self.options['central_body']
                gd = self.options['grid_data']

                # Orbit propagation
                self.add_subsystem('eom', TwoBodyDynamicsComp(num_nodes=nn, central_body=cb), promotes=['*'])
                self.add_subsystem('mee_to_cart', MEEToCart(num_nodes=nn, central_body=cb), promotes=['*'])
                self.add_subsystem('state_mux_comp', StateMuxComp(num_nodes=nn), promotes=['*'])

                # Attitude
                self.add_subsystem('ori_comp', ORIComp(num_nodes=nn), promotes=['*'])
                self.add_subsystem('obr_comp', OBRComp(num_nodes=nn), promotes=['*'])
                self.add_subsystem('obi_comp', OBIComp(num_nodes=nn), promotes=['*'])
                self.add_subsystem('odotbi_comp', OdotBIComp(num_nodes=nn, grid_data=gd), promotes=['*'])
                self.add_subsystem('omega_comp', AngularVelocityComp(num_nodes=nn), promotes=['*'])
                self.add_subsystem('omega_dot_comp', AngularAccelerationComp(num_nodes=nn, grid_data=gd), promotes=['*'])
                self.add_subsystem('body_vel_comp', BodyVelComp(num_nodes=nn), promotes=['*'])
                self.add_subsystem('torque_req_comp', AttitudeTorqueComp(num_nodes=nn), promotes=['*'])

        nn = 50
        traj = prob.model.add_subsystem('traj', dm.Trajectory())
        tx = dm.PicardShooting(num_segments=1, nodes_per_seg=nn, grid_type='lgl')
        orbit = traj.add_phase('orbit', dm.Phase(ode_class=TwoBodyODE,
                                                 transcription=tx,
                                                #  transcription=dm.Birkhoff(num_nodes=nn),
                                                 ode_init_kwargs={'central_body': 'earth',
                                                                  'vectorize_outputs': False,
                                                                  'grid_data': tx.grid_data}))

        orbit.set_time_options(fix_initial=True, fix_duration=True, units='TU_earth', initial_val=0.0,
                               dt_dstau_targets=['dt_dstau'])
        orbit.add_state('p', fix_initial=True, units='DU_earth', rate_source='p_dot', lower=0.5)
        orbit.add_state('f', fix_initial=True, units='unitless', rate_source='f_dot')
        orbit.add_state('g', fix_initial=True, units='unitless', rate_source='g_dot')
        orbit.add_state('h', fix_initial=True, units='unitless', rate_source='h_dot')
        orbit.add_state('k', fix_initial=True, units='unitless', rate_source='k_dot')
        orbit.add_state('L', fix_initial=True, units='rad', rate_source='L_dot')
        orbit.add_objective('time', loc='final')

        orbit.add_timeseries_output(['x', 'y', 'z', 'vx', 'vy', 'vz', 'r_e2b_I',
                                     'v_e2b_I', 'v_e2b_B', 'O_BI', 'O_BR', 'O_RI', 'Odot_BI',
                                     'w_B', 'wdot_B', 'T_req'])

        prob.setup()

        orbit.set_time_val(duration=8, units='TU_earth')
        orbit.set_state_val('p', (4/np.pi)**(2/3), units='DU_earth')
        orbit.set_state_val('f', 0)
        orbit.set_state_val('g', 0)
        orbit.set_state_val('h', 0)
        orbit.set_state_val('k', 0)
        orbit.set_state_val('L', 0)

        dm.run_problem(prob, run_driver=False, make_plots=True)

        p = prob.get_val('traj.orbit.timeseries.p')
        f = prob.get_val('traj.orbit.timeseries.f')
        g = prob.get_val('traj.orbit.timeseries.g')
        h = prob.get_val('traj.orbit.timeseries.h')
        k = prob.get_val('traj.orbit.timeseries.k')
        L = prob.get_val('traj.orbit.timeseries.L')

        x = prob.get_val('traj.orbit.timeseries.x')
        y = prob.get_val('traj.orbit.timeseries.y')
        z = prob.get_val('traj.orbit.timeseries.z')
        vx = prob.get_val('traj.orbit.timeseries.vx')
        vy = prob.get_val('traj.orbit.timeseries.vy')
        vz = prob.get_val('traj.orbit.timeseries.vz')

        r_e2b_I = prob.get_val('traj.orbit.timeseries.r_e2b_I')
        v_e2b_I = prob.get_val('traj.orbit.timeseries.v_e2b_I')
        v_e2b_B = prob.get_val('traj.orbit.timeseries.v_e2b_B')

        assert_near_equal(p[0], p[-1], tolerance=1e-6)

        assert_near_equal(x[0], x[-1], tolerance=1e-6, tol_type='abs')
        assert_near_equal(y[0], y[-1], tolerance=1e-6, tol_type='abs')
        assert_near_equal(z[0], z[-1], tolerance=1e-6, tol_type='abs')
        assert_near_equal(vx[0], vx[-1], tolerance=1e-6, tol_type='abs')
        assert_near_equal(vy[0], vy[-1], tolerance=1e-6, tol_type='abs')
        assert_near_equal(vz[0], vz[-1], tolerance=1e-6, tol_type='abs')

        assert_near_equal(f, np.zeros((nn, 1)), tolerance=1e-6)
        assert_near_equal(g, np.zeros((nn, 1)), tolerance=1e-6)
        assert_near_equal(h, np.zeros((nn, 1)), tolerance=1e-6)
        assert_near_equal(k, np.zeros((nn, 1)), tolerance=1e-6)
        assert_near_equal(L[-1], 2*np.pi, tolerance=1e-6)

if __name__ == '__main__':
    unittest.main()
