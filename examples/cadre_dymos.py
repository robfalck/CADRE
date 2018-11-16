from __future__ import print_function

import unittest

import numpy as np

from openmdao.api import Problem, Group, pyOptSparseDriver, DirectSolver
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from dymos import Phase
from dymos.utils.indexing import get_src_indices_by_row
from dymos.phases.components import ControlInterpComp

from CADRE.odes_dymos.cadre_orbit_ode import CadreOrbitODE
from CADRE.attitude_dymos.angular_velocity_comp import AngularVelocityComp
from CADRE.odes_dymos.cadre_systems_ode import CadreSystemsODE

GM = 398600.44
rmag = 7000.0
period = 2 * np.pi * np.sqrt(rmag ** 3 / GM)
vcirc = np.sqrt(GM / rmag)
duration = period
duration = 9 * 3600.0



p = Problem(model=Group())

p.driver = pyOptSparseDriver()
p.driver.options['optimizer'] = 'SNOPT'
p.driver.options['dynamic_simul_derivs'] = True
p.driver.opt_settings['Major iterations limit'] = 1000
p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-4
p.driver.opt_settings['Major optimality tolerance'] = 1.0E-4
p.driver.opt_settings['iSumm'] = 6

NUM_SEG = 30
TRANSCRIPTION_ORDER = 7

orbit_phase = Phase('radau-ps',
                    ode_class=CadreOrbitODE,
                    num_segments=NUM_SEG,
                    transcription_order=TRANSCRIPTION_ORDER,
                    compressed=False)

p.model.add_subsystem('orbit_phase', orbit_phase)

orbit_phase.set_time_options(fix_initial=True, fix_duration=True, duration_ref=duration)
orbit_phase.set_state_options('r_e2b_I', defect_scaler=1000, fix_initial=True, units='km')
orbit_phase.set_state_options('v_e2b_I', defect_scaler=1000, fix_initial=True, units='km/s')
# orbit_phase.set_state_options('SOC', defect_scaler=1, fix_initial=True, units=None)
# orbit_phase.add_design_parameter('P_bat', opt=False, units='W')
orbit_phase.add_design_parameter('Gamma', opt=False, units='rad')
orbit_phase.add_objective('time', loc='final', scaler=10)

# Add a control interp comp to interpolate the rates of O_BI from the orbit phase.
faux_control_options = {'O_BI': {'units': None, 'shape': (3, 3)}}
p.model.add_subsystem('obi_rate_interp_comp',
                      ControlInterpComp(control_options=faux_control_options,
                                        time_units='s',
                                        grid_data=orbit_phase.grid_data),
                      promotes_outputs=[('control_rates:O_BI_rate', 'Odot_BI')])

control_input_nodes_idxs = orbit_phase.grid_data.subset_node_indices['control_input']
src_idxs = get_src_indices_by_row(control_input_nodes_idxs, shape=(3, 3))
p.model.connect('orbit_phase.rhs_all.O_BI', 'obi_rate_interp_comp.controls:O_BI',
                src_indices=src_idxs, flat_src_indices=True)
p.model.connect('orbit_phase.time.dt_dstau',
                ('obi_rate_interp_comp.dt_dstau', 'w_B_rate_interp_comp.dt_dstau'))

# Use O_BI and Odot_BI to compute the angular velocity vector
p.model.add_subsystem('angular_velocity_comp',
                      AngularVelocityComp(num_nodes=orbit_phase.grid_data.num_nodes))
p.model.connect('orbit_phase.rhs_all.O_BI', 'angular_velocity_comp.O_BI')
p.model.connect('Odot_BI', 'angular_velocity_comp.Odot_BI')

# Add another interpolation comp to compute the rate of w_B
faux_control_options = {'w_B': {'units': '1/s', 'shape': (3,)}}
p.model.add_subsystem('w_B_rate_interp_comp',
                      ControlInterpComp(control_options=faux_control_options,
                                        time_units='s',
                                        grid_data=orbit_phase.grid_data),
                      promotes_outputs=[('control_rates:w_B_rate', 'wdot_B')])

src_idxs = get_src_indices_by_row(control_input_nodes_idxs, shape=(3,))
p.model.connect('angular_velocity_comp.w_B', 'w_B_rate_interp_comp.controls:w_B',
                src_indices=src_idxs, flat_src_indices=True)

# Now add the systems phase

systems_phase = Phase('radau-ps',
                      ode_class=CadreSystemsODE,
                      num_segments=NUM_SEG,
                      transcription_order=TRANSCRIPTION_ORDER,
                      compressed=False)

p.model.add_subsystem('systems_phase', systems_phase)

systems_phase.set_time_options(fix_initial=True, fix_duration=True, duration_ref=duration)
systems_phase.set_state_options('SOC', defect_ref=10, fix_initial=True, units=None)
systems_phase.set_state_options('w_RW', defect_ref=1000, fix_initial=True, units='1/s')
systems_phase.set_state_options('data', defect_ref=10, fix_initial=True, units='Gibyte')
systems_phase.set_state_options('temperature', ref0=273, ref=373, defect_ref=100,
                                fix_initial=True, units='degK')

systems_phase.add_design_parameter('LD', opt=False, units='d')
systems_phase.add_design_parameter('fin_angle', opt=True, lower=0., upper=np.pi / 2.)
systems_phase.add_design_parameter('antAngle', opt=True, lower=-np.pi / 4, upper=np.pi / 4)
systems_phase.add_design_parameter('cellInstd', opt=True, lower=0.0, upper=1.0, ref=1.0)

# Add r_e2b_I and O_BI as non-optimized controls, allowing them to be connected to external sources
systems_phase.add_control('r_e2b_I', opt=False, units='km')
systems_phase.add_control('O_BI', opt=False)
systems_phase.add_control('w_B', opt=False)
systems_phase.add_control('wdot_B', opt=False)
systems_phase.add_control('P_comm', opt=True, lower=0.0, upper=30.0, units='W')
systems_phase.add_control('Isetpt', opt=True, lower=1.0E-4, upper=0.4, units='A')

# Connect r_e2b_I and O_BI values from all nodes in the orbit phase to the input values
# in the attitude phase.
src_idxs = get_src_indices_by_row(control_input_nodes_idxs, shape=(3,))
p.model.connect('orbit_phase.states:r_e2b_I', 'systems_phase.controls:r_e2b_I',
                src_indices=src_idxs, flat_src_indices=True)
p.model.connect('angular_velocity_comp.w_B', 'systems_phase.controls:w_B',
                src_indices=src_idxs, flat_src_indices=True)
p.model.connect('wdot_B', 'systems_phase.controls:wdot_B',
                src_indices=src_idxs, flat_src_indices=True)

src_idxs = get_src_indices_by_row(control_input_nodes_idxs, shape=(3, 3))
p.model.connect('orbit_phase.rhs_all.O_BI', 'systems_phase.controls:O_BI',
                src_indices=src_idxs, flat_src_indices=True)

p.model.options['assembled_jac_type'] = 'csc'
p.model.linear_solver = DirectSolver(assemble_jac=True)

p.setup(check=True)

# from openmdao.api import view_model
# view_model(p.model)

# Initialize values in the orbit phase

p['orbit_phase.t_initial'] = 0.0
p['orbit_phase.t_duration'] = duration

# p['systems_phase.states:w_RW'][:, 0] = 0.0
# p['systems_phase.states:w_RW'][:, 1] = 0.0
# p['systems_phase.states:w_RW'][:, 2] = 0.0

# Default starting orbit
# [ 2.89078958e+03  5.69493134e+03 -2.55340189e+03  2.56640460e-01
#               3.00387409e+00  6.99018448e+00]

p['orbit_phase.states:r_e2b_I'][:, 0] = 2.89078958e+03
p['orbit_phase.states:r_e2b_I'][:, 1] = 5.69493134e+03
p['orbit_phase.states:r_e2b_I'][:, 2] = -2.55340189e+03

p['orbit_phase.states:v_e2b_I'][:, 0] = 2.56640460e-01
p['orbit_phase.states:v_e2b_I'][:, 1] = 3.00387409e+00
p['orbit_phase.states:v_e2b_I'][:, 2] = 6.99018448e+00

# Initialize values in the systems phase

p['systems_phase.t_initial'] = 0.0
p['systems_phase.t_duration'] = duration

# p['systems_phase.states:w_RW'][:, 0] = 0.0
# p['systems_phase.states:w_RW'][:, 1] = 0.0
# p['systems_phase.states:w_RW'][:, 2] = 0.0

p['systems_phase.states:SOC'] = 1.0
p['systems_phase.states:w_RW'] = 100.0
p['systems_phase.states:data'] = 0.0
p['systems_phase.states:temperature'] = 273.0

# p['systems_phase.states:v_e2b_I'][:, 0] = 0.0
# p['systems_phase.states:v_e2b_I'][:, 1] = vcirc
# p['systems_phase.states:v_e2b_I'][:, 2] = 0.0
p['systems_phase.controls:P_comm'] = 0.0
p['systems_phase.controls:Isetpt'] = 0.01

p['systems_phase.design_parameters:LD'] = 5233.5
p['systems_phase.design_parameters:fin_angle'] = np.radians(70.0)
p['systems_phase.design_parameters:cellInstd'] = 0.0

p.run_model()

# Simulate the orbit phase to get a (exact) guess to the orbit history solution.
exp_out = orbit_phase.simulate()

# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
#
# plt.figure()
# ax = plt.axes(projection='3d')
# # plt.plot(exp_out.get_values('r_e2b_I')[:, 0], exp_out.get_values('r_e2b_I')[:, 1], 'b-')
# ax.plot3D(exp_out.get_values('r_e2b_I')[:, 0], exp_out.get_values('r_e2b_I')[:, 1], exp_out.get_values('r_e2b_I')[:, 2], 'b-')
# plt.show()

p['orbit_phase.states:r_e2b_I'] = orbit_phase.interpolate(ys=exp_out.get_values('r_e2b_I'), xs=exp_out.get_values('time'), nodes='state_input')
p['orbit_phase.states:v_e2b_I'] = orbit_phase.interpolate(ys=exp_out.get_values('v_e2b_I'), xs=exp_out.get_values('time'), nodes='state_input')

p.run_driver()

r_e2b_I = p.model.orbit_phase.get_values('r_e2b_I')
v_e2b_I = p.model.orbit_phase.get_values('v_e2b_I')
rmag_e2b = p.model.orbit_phase.get_values('rmag_e2b_I')
# assert_rel_error(self, rmag_e2b, rmag * np.ones_like(rmag_e2b), tolerance=1.0E-9)
# delta_trua = 2 * np.pi * (duration / period)
# assert_rel_error(self, r_e2b_I[-1, :],
#                  rmag * np.array([np.cos(delta_trua), np.sin(delta_trua), 0]),
#                  tolerance=1.0E-9)
# assert_rel_error(self, v_e2b_I[-1, :],
#                  vcirc * np.array([-np.sin(delta_trua), np.cos(delta_trua), 0]),
#                  tolerance=1.0E-9)

    # def test_partials(self):
    #     np.set_printoptions(linewidth=10000, edgeitems=1024)
    #     cpd = self.p.check_partials(compact_print=True, out_stream=None)
    #     assert_check_partials(cpd, atol=1.0E-4, rtol=1.0)
    #
    # def test_simulate(self):
    #     phase = self.p.model.orbit_phase
    #     exp_out = phase.simulate(times=500)
    #
    #     import matplotlib.pyplot as plt
    #
    #     plt.figure()
    #     plt.plot(exp_out.get_values('r_e2b_I')[:, 0], exp_out.get_values('r_e2b_I')[:, 1], 'b-')
    #     plt.plot(phase.get_values('r_e2b_I')[:, 0], phase.get_values('r_e2b_I')[:, 1], 'ro')
    #
    #     # plt.figure()
    #     # plt.plot(exp_out.get_values('time'), exp_out.get_values('SOC'), 'b-')
    #     # plt.plot(phase.get_values('time'), phase.get_values('SOC'), 'ro')
    #
    #     plt.show()
