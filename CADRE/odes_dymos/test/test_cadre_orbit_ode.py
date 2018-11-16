from __future__ import print_function

import unittest

import numpy as np

from openmdao.api import Problem, Group, pyOptSparseDriver
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from dymos import Phase

from CADRE.odes_dymos.cadre_orbit_ode import CadreOrbitODE

GM = 398600.44
rmag = 7000.0
period = 2 * np.pi * np.sqrt(rmag ** 3 / GM)
vcirc = np.sqrt(GM / rmag)
duration = period / 1

class TestCadreOrbitODE(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        p = cls.p = Problem(model=Group())

        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SNOPT'
        p.driver.options['dynamic_simul_derivs'] = True
        p.driver.opt_settings['Major iterations limit'] = 100
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        p.driver.opt_settings['iSumm'] = 6

        phase = Phase('radau-ps',
                      ode_class=CadreOrbitODE,
                      num_segments=10,
                      transcription_order=7,
                      compressed=False)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(duration, duration))

        phase.set_state_options('r_e2b_I', defect_scaler=1000, fix_initial=True, units='km')
        phase.set_state_options('v_e2b_I', defect_scaler=1000, fix_initial=True, units='km/s')
        # phase.set_state_options('SOC', defect_scaler=1, fix_initial=True, units=None)

        # phase.add_design_parameter('P_bat', opt=False, units='W')
        phase.add_design_parameter('Gamma', opt=False, units='rad')

        phase.add_objective('time', loc='final', scaler=10)

        # p.model.options['assembled_jac_type'] = 'csc'
        # p.model.linear_solver = DirectSolver(assemble_jac=True)

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = duration

        # print(phase.grid_data.num_nodes)
        #
        p['phase0.states:r_e2b_I'][:, 0] = rmag
        p['phase0.states:r_e2b_I'][:, 1] = 0.0
        p['phase0.states:r_e2b_I'][:, 2] = 0.0

        p['phase0.states:v_e2b_I'][:, 0] = 0.0
        p['phase0.states:v_e2b_I'][:, 1] = vcirc
        p['phase0.states:v_e2b_I'][:, 2] = 0.0

        # p['phase0.design_parameters:P_bat'] = 2.0
        p['phase0.design_parameters:Gamma'] = 0.0

        p.run_model()
        p.run_driver()

    def test_results(self):
        r_e2b_I = self.p.model.phase0.get_values('r_e2b_I')
        v_e2b_I = self.p.model.phase0.get_values('v_e2b_I')
        rmag_e2b = self.p.model.phase0.get_values('rmag_e2b_I')
        assert_rel_error(self, rmag_e2b, rmag * np.ones_like(rmag_e2b), tolerance=1.0E-9)
        delta_trua = 2 * np.pi * (duration / period)
        assert_rel_error(self, r_e2b_I[-1, :], rmag * np.array([np.cos(delta_trua), np.sin(delta_trua), 0]), tolerance=1.0E-9)
        assert_rel_error(self, v_e2b_I[-1, :], vcirc * np.array([-np.sin(delta_trua), np.cos(delta_trua), 0]), tolerance=1.0E-9)

        print(self.p.model.phase0.get_values('O_BI'))

    def test_partials(self):
        np.set_printoptions(linewidth=10000, edgeitems=1024)
        cpd = self.p.check_partials(compact_print=True)
        assert_check_partials(cpd, atol=1.0E-4, rtol=1.0)

    def test_simulate(self):
        phase = self.p.model.phase0
        exp_out = phase.simulate(times=500)

        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(exp_out.get_values('r_e2b_I')[:, 0], exp_out.get_values('r_e2b_I')[:, 1], 'b-')
        plt.plot(phase.get_values('r_e2b_I')[:, 0], phase.get_values('r_e2b_I')[:, 1], 'ro')

        # plt.figure()
        # plt.plot(exp_out.get_values('time'), exp_out.get_values('SOC'), 'b-')
        # plt.plot(phase.get_values('time'), phase.get_values('SOC'), 'ro')

        plt.show()
