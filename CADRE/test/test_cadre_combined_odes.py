from __future__ import print_function

import unittest

import numpy as np

from openmdao.api import Problem, IndepVarComp, Group, DirectSolver, pyOptSparseDriver
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from dymos import Phase
from dymos.utils.indexing import get_src_indices_by_row

from CADRE.cadre_orbit_ode import CadreOrbitODE
from CADRE.cadre_attitude_ode import CadreAttitudeODE

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

        NUM_SEG = 10
        TRANSCRIPTION_ORDER = 7

        orbit_phase = Phase('radau-ps',
                            ode_class=CadreOrbitODE,
                            num_segments=NUM_SEG,
                            transcription_order=TRANSCRIPTION_ORDER,
                            compressed=False)

        p.model.add_subsystem('orbit_phase', orbit_phase)

        orbit_phase.set_time_options(fix_initial=True, fix_duration=True)
        orbit_phase.set_state_options('r_e2b_I', defect_scaler=1000, fix_initial=True, units='km')
        orbit_phase.set_state_options('v_e2b_I', defect_scaler=1000, fix_initial=True, units='km/s')
        # orbit_phase.set_state_options('SOC', defect_scaler=1, fix_initial=True, units=None)
        # orbit_phase.add_design_parameter('P_bat', opt=False, units='W')
        orbit_phase.add_design_parameter('Gamma', opt=False, units='rad')
        orbit_phase.add_objective('time', loc='final', scaler=10)
        
        attitude_phase = Phase('radau-ps',
                               ode_class=CadreAttitudeODE,
                               num_segments=NUM_SEG,
                               transcription_order=TRANSCRIPTION_ORDER,
                               compressed=False)

        p.model.add_subsystem('attitude_phase', attitude_phase)

        attitude_phase.set_time_options(fix_initial=True, fix_duration=True)
        # attitude_phase.set_state_options('w_RW', defect_scaler=1000, fix_initial=True, units='1/s')
        attitude_phase.set_state_options('SOC', defect_scaler=1, fix_initial=True, units=None)
        attitude_phase.add_design_parameter('P_bat', opt=False, units='W')
        attitude_phase.add_design_parameter('LD', opt=False, units='d')
        attitude_phase.add_design_parameter('fin_angle', opt=False, units='deg')

        # Add r_e2b_I and O_BI as non-optimized controls, allowing them to be connected to external sources
        attitude_phase.add_control('r_e2b_I', opt=False, units='km')
        attitude_phase.add_control('O_BI', opt=False)

        # Connect r_e2b_I and O_BI values from all nodes in the orbit phase to the input values
        # in the attitude phase.
        control_input_nodes_idxs = orbit_phase.grid_data.subset_node_indices['control_input']
        src_idxs = get_src_indices_by_row(control_input_nodes_idxs, shape=(3,))
        p.model.connect('orbit_phase.states:r_e2b_I', 'attitude_phase.controls:r_e2b_I',
                        src_indices=src_idxs, flat_src_indices=True)

        src_idxs = get_src_indices_by_row(control_input_nodes_idxs, shape=(3,3))
        p.model.connect('orbit_phase.rhs_all.O_BI', 'attitude_phase.controls:O_BI',
                        src_indices=src_idxs, flat_src_indices=True)

        p.setup(check=True)

        # from openmdao.api import view_model
        # view_model(p.model)

        # Initialize values in the orbit phase

        p['orbit_phase.t_initial'] = 0.0
        p['orbit_phase.t_duration'] = duration

        # p['attitude_phase.states:w_RW'][:, 0] = 0.0
        # p['attitude_phase.states:w_RW'][:, 1] = 0.0
        # p['attitude_phase.states:w_RW'][:, 2] = 0.0

        p['orbit_phase.states:r_e2b_I'][:, 0] = rmag
        p['orbit_phase.states:r_e2b_I'][:, 1] = 0.0
        p['orbit_phase.states:r_e2b_I'][:, 2] = 0.0

        p['orbit_phase.states:v_e2b_I'][:, 0] = 0.0
        p['orbit_phase.states:v_e2b_I'][:, 1] = vcirc
        p['orbit_phase.states:v_e2b_I'][:, 2] = 0.0

        # Initialize values in the attitude phase

        p['attitude_phase.t_initial'] = 0.0
        p['attitude_phase.t_duration'] = duration

        # p['attitude_phase.states:w_RW'][:, 0] = 0.0
        # p['attitude_phase.states:w_RW'][:, 1] = 0.0
        # p['attitude_phase.states:w_RW'][:, 2] = 0.0

        p['attitude_phase.states:SOC'] = 1.0

        # p['attitude_phase.states:v_e2b_I'][:, 0] = 0.0
        # p['attitude_phase.states:v_e2b_I'][:, 1] = vcirc
        # p['attitude_phase.states:v_e2b_I'][:, 2] = 0.0
        #
        p['attitude_phase.design_parameters:P_bat'] = 2.0
        p['attitude_phase.design_parameters:LD'] = 5233.5
        p['attitude_phase.design_parameters:fin_angle'] = 70.0

        p.run_model()
        p.run_driver()

    def test_results(self):
        r_e2b_I = self.p.model.orbit_phase.get_values('r_e2b_I')
        v_e2b_I = self.p.model.orbit_phase.get_values('v_e2b_I')
        rmag_e2b = self.p.model.orbit_phase.get_values('rmag_e2b_I')
        assert_rel_error(self, rmag_e2b, rmag * np.ones_like(rmag_e2b), tolerance=1.0E-9)
        delta_trua = 2 * np.pi * (duration / period)
        assert_rel_error(self, r_e2b_I[-1, :], rmag * np.array([np.cos(delta_trua), np.sin(delta_trua), 0]), tolerance=1.0E-9)
        assert_rel_error(self, v_e2b_I[-1, :], vcirc * np.array([-np.sin(delta_trua), np.cos(delta_trua), 0]), tolerance=1.0E-9)

    def test_partials(self):
        np.set_printoptions(linewidth=10000, edgeitems=1024)
        cpd = self.p.check_partials(compact_print=True)
        assert_check_partials(cpd, atol=1.0E-4, rtol=1.0)

    def test_simulate(self):
        phase = self.p.model.orbit_phase
        exp_out = phase.simulate(times=500)

        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(exp_out.get_values('r_e2b_I')[:, 0], exp_out.get_values('r_e2b_I')[:, 1], 'b-')
        plt.plot(phase.get_values('r_e2b_I')[:, 0], phase.get_values('r_e2b_I')[:, 1], 'ro')

        # plt.figure()
        # plt.plot(exp_out.get_values('time'), exp_out.get_values('SOC'), 'b-')
        # plt.plot(phase.get_values('time'), phase.get_values('SOC'), 'ro')

        plt.show()
