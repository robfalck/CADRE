from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import ExplicitComponent


class ORIComp(ExplicitComponent):
    """
    Coordinate transformation from the inertial plane to the rolled
    (forward facing) plane.
    """

    dvx_dv = np.zeros((3, 3, 3))
    dvx_dv[0, :, 0] = (0., 0., 0.)
    dvx_dv[1, :, 0] = (0., 0., -1.)
    dvx_dv[2, :, 0] = (0., 1., 0.)

    dvx_dv[0, :, 1] = (0., 0., 1.)
    dvx_dv[1, :, 1] = (0., 0., 0.)
    dvx_dv[2, :, 1] = (-1., 0., 0.)

    dvx_dv[0, :, 2] = (0., -1., 0.)
    dvx_dv[1, :, 2] = (1., 0., 0.)
    dvx_dv[2, :, 2] = (0., 0., 0.)

    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('r_e2b_I', np.zeros((nn, 3)), units='km',
                       desc='Position vector from earth to satellite in '
                            'Earth-centered inertial frame over time')

        self.add_input('v_e2b_I', np.zeros((nn, 3)), units='km/s',
                       desc='Velocity vector from earth to satellite in '
                            'Earth-centered inertial frame over time')

        # Outputs
        self.add_output('O_RI', np.zeros((nn, 3, 3)), units=None,
                        desc='Rotation matrix from rolled body-fixed frame to '
                             'Earth-centered inertial frame over time')

        temp = np.ones((3*3, 3), dtype=int)
        temp[-3:, :] = 0
        template = np.kron(np.eye(nn, dtype=int), temp)
        rows, cols = np.nonzero(template)

        self.declare_partials('O_RI', 'r_e2b_I', rows=rows, cols=cols)

        template = np.kron(np.eye(nn, dtype=int), np.ones((3*3, 3), dtype=int))
        rows, cols = np.nonzero(template)

        self.declare_partials('O_RI', 'v_e2b_I', rows=rows, cols=cols, val=1)


    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        r_e2b_I = inputs['r_e2b_I']
        v_e2b_I = inputs['v_e2b_I']
        O_RI = outputs['O_RI']

        O_RI[:] = np.zeros(O_RI.shape)
        for i in range(0, self.options['num_nodes']):

            r = r_e2b_I[i, :]
            v = v_e2b_I[i, :]

            normr = np.sqrt(np.dot(r, r))
            normv = np.sqrt(np.dot(v, v))

            # Prevent overflow
            if normr < 1e-10:
                normr = 1e-10
            if normv < 1e-10:
                normv = 1e-10

            r = r / normr
            v = v / normv

            vx = np.zeros((3, 3))
            vx[0, :] = (0., -v[2], v[1])
            vx[1, :] = (v[2], 0., -v[0])
            vx[2, :] = (-v[1], v[0], 0.)

            iB = np.dot(vx, r)
            jB = -np.dot(vx, iB)

            O_RI[i, 0, :] = iB
            O_RI[i, 1, :] = jB
            O_RI[i, 2, :] = -v

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        r_e2b_I = inputs['r_e2b_I']
        v_e2b_I = inputs['v_e2b_I']

        diB_dv = np.zeros((3, 3))
        djB_dv = np.zeros((3, 3))

        for i in range(0, self.options['num_nodes']):

            r = r_e2b_I[i, :]
            v = v_e2b_I[i, :]

            normr = np.sqrt(np.dot(r, r))
            normv = np.sqrt(np.dot(v, v))

            # Prevent overflow
            if normr < 1e-10:
                normr = 1e-10
            if normv < 1e-10:
                normv = 1e-10

            r = r / normr
            v = v / normv

            dr_dr = np.zeros((3, 3))
            dv_dv = np.zeros((3, 3))

            for k in range(0, 3):
                dr_dr[k, k] += 1.0 / normr
                dv_dv[k, k] += 1.0 / normv
                dr_dr[:, k] -= r * r_e2b_I[i, k] / normr ** 2
                dv_dv[:, k] -= v * v_e2b_I[i, k] / normv ** 2

            vx = np.zeros((3, 3))
            vx[0, :] = (0., -v[2], v[1])
            vx[1, :] = (v[2], 0., -v[0])
            vx[2, :] = (-v[1], v[0], 0.)

            iB = np.dot(vx, r)

            diB_dr = vx
            diB_dv[:, 0] = np.dot(self.dvx_dv[:, :, 0], r)
            diB_dv[:, 1] = np.dot(self.dvx_dv[:, :, 1], r)
            diB_dv[:, 2] = np.dot(self.dvx_dv[:, :, 2], r)

            djB_diB = -vx
            djB_dv[:, 0] = -np.dot(self.dvx_dv[:, :, 0], iB)
            djB_dv[:, 1] = -np.dot(self.dvx_dv[:, :, 1], iB)
            djB_dv[:, 2] = -np.dot(self.dvx_dv[:, :, 2], iB)

            partials['O_RI', 'r_e2b_I'][i*18:i*18 + 9] = np.dot(diB_dr, dr_dr).flatten()
            #partials['O_RI', 'r_e2b_I'][n0:n0+9] = np.dot(diB_dr, dr_dr).flatten()
            partials['O_RI', 'v_e2b_I'][i*27:i*27+9] = np.dot(diB_dv, dv_dv).flatten()
            # print()
            # print(n0, n0+9, n0+18, n0+27)
            # print(partials['O_RI', 'r_e2b_I'].shape)
            # print(np.dot(np.dot(djB_diB, diB_dr), dr_dr).flatten().shape)

            partials['O_RI', 'r_e2b_I'][i*18+9:i*18+18] = np.dot(np.dot(djB_diB, diB_dr), dr_dr).flatten()
            # partials['O_RI', 'r_e2b_I'][n0+9:n0+18] = np.dot(np.dot(djB_diB, diB_dr), dr_dr).flatten()
            partials['O_RI', 'v_e2b_I'][i*27+9:i*27+18] = np.dot(np.dot(djB_diB, diB_dv) + djB_dv, dv_dv).flatten()

            partials['O_RI', 'v_e2b_I'][i*27+18:i*27+27] = -dv_dv.flatten()

# class ORIComp(ExplicitComponent):
#     """
#     Coordinate transformation from the inertial plane to the rolled
#     (forward facing) plane.
#     """
#     def initialize(self):
#         self.options.declare('num_nodes', types=(int,))
#
#     def setup(self):
#         nn = self.options['num_nodes']
#
#         # Inputs
#         self.add_input('runit_e2b_I', np.zeros((nn, 3)), units=None,
#                        desc='Unitized osition vector from earth to satellite in '
#                             'Earth-centered inertial frame over time')
#
#         self.add_input('vunit_e2b_I', np.zeros((nn, 3)), units=None,
#                        desc='Unitized velocity vector from earth to satellite in '
#                             'Earth-centered inertial frame over time')
#
#         self.add_input('hunit_e2b_I', np.zeros((nn, 3)), units=None,
#                        desc='Unitized orbit normal vector from earth to satellite in '
#                             'Earth-centered inertial frame over time')
#         # Outputs
#         self.add_output('O_RI', np.zeros((nn, 3, 3)), units=None,
#                         desc='Rotation matrix from rolled body-fixed frame to '
#                              'Earth-centered inertial frame over time')
#
#         cs = np.arange(nn * 3, dtype=int)
#
#         rs = np.arange(nn * 3, dtype=int) + np.repeat(0 + 6 * np.arange(nn, dtype=int), 3)
#         self.declare_partials('O_RI', 'runit_e2b_I', rows=rs, cols=cs, val=1.0)
#
#         rs = np.arange(nn * 3, dtype=int) + np.repeat(3 + 6 * np.arange(nn, dtype=int), 3)
#         self.declare_partials('O_RI', 'vunit_e2b_I', rows=rs, cols=cs, val=1.0)
#
#         rs = np.arange(nn * 3, dtype=int) + np.repeat(6 + 6 * np.arange(nn, dtype=int), 3)
#         self.declare_partials('O_RI', 'hunit_e2b_I', rows=rs, cols=cs, val=1.0)
#
#
#     def compute(self, inputs, outputs):
#         """
#         Calculate outputs.
#         """
#         rhat = inputs['runit_e2b_I']
#         vhat = inputs['vunit_e2b_I']
#         hhat = inputs['hunit_e2b_I']
#
#         outputs['O_RI'][:, 0, :] = rhat
#         outputs['O_RI'][:, 1, :] = vhat
#         outputs['O_RI'][:, 2, :] = hhat
