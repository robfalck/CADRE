import unittest
import openmdao.api as om
import dymos as dm
import numpy as np
from CADRE.reactionwheel_dymos import ReactionWheel

class TestReactionWheelDymos(unittest.TestCase):
    def test_simple_prop(self):
        p = om.Problem()
        traj = dm.Trajectory()
        phase = traj.add_phase('phase', dm.Phase(ode_class=ReactionWheel,
                       transcription=dm.PicardShooting(num_segments=1, nodes_per_seg=20)))
        p.model.add_subsystem('traj', traj)

        phase.set_state_options('w_RW', fix_initial=True, rate_source='alpha_RW')
        phase.add_control('T_RW', opt=False)
        p.setup()

        phase.set_control_val('T_RW', [[1.0E-5, 0, 0], [1.0E-5, 0, 0]], units='N*m')

        dm.run_problem(p, run_driver=False, simulate=True, make_plots=True)
