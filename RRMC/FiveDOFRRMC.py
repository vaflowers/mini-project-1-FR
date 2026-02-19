from math import *
import numpy as np
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.arm_models import (
    TwoDOFRobotTemplate, ScaraRobotTemplate, FiveDOFRobotTemplate
)


class FiveDOFRobot(FiveDOFRobotTemplate):
    def __init__(self):
        super().__init__()

    def calc_forward_kinematics(self, joint_values: list, radians=True):
        """
        Calculate Forward Kinematics (FK) based on the given joint angles.

        Args:
            joint_values (list): Joint angles (in radians if radians=True, otherwise in degrees).
            radians (bool): Whether the input angles are in radians (default is False).
        """
        curr_joint_values = joint_values.copy()

        if not radians: # Convert degrees to radians if the input is in degrees
            curr_joint_values = [np.deg2rad(theta) for theta in curr_joint_values]

        # Ensure that the joint angles respect the joint limits
        for i in range(4):
            theta = curr_joint_values[i]
            curr_joint_values[i] = np.clip(theta, self.joint_limits[i][0], self.joint_limits[i][1])

       
        # DH parameters for each joint

        DH = np.zeros((self.num_dof, 4))
        DH[0] = [curr_joint_values[0], self.l1, 0, -pi/2]
        DH[1] = [curr_joint_values[1] - pi, 0, self.l2, pi]
        DH[2] = [curr_joint_values[2], 0, self.l3, pi]
        DH[3] = [curr_joint_values[3] + pi/2, 0, 0, pi/2]
        DH[4] = [curr_joint_values[4], self.l4 + self.l5, 0, 0]

        # Compute the transformation matrices
        Hlist = [ut.dh_to_matrix(dh) for dh in DH]

        # Precompute cumulative transformations to avoid redundant calculations
        H_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            H_cumulative.append(H_cumulative[-1] @ Hlist[i])

        # Calculate EE position and rotation
        H_ee = H_cumulative[-1]  # Final transformation matrix for EE

        # Set the end effector (EE) position
        ee = ut.EndEffector()
        ee.x, ee.y, ee.z = (H_ee @ np.array([0, 0, 0, 1]))[:3]
       
        # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
        rpy = ut.rotm_to_euler(H_ee[:3, :3])
        ee.rotx, ee.roty, ee.rotz = rpy[0], rpy[1], rpy[2]

        return ee, Hlist


    def calc_velocity_kinematics(self, joint_values: list, vel: list, dt=0.02):
        """
        Calculates the velocity kinematics for the robot based on the given velocity input.

        Args:
            vel (list): The velocity vector for the end effector [vx, vy].
        """
        new_joint_values = list(joint_values[:self.num_dof])

        # move robot slightly out of zeros singularity
        if all(theta == 0.0 for theta in new_joint_values):
            new_joint_values = [theta + np.random.rand()*0.02 for theta in new_joint_values]
       
        # Calculate joint velocities using the inverse Jacobian
        vel = vel[:3]  # Consider only the first two components of the velocity
        joint_vel = self.inverse_jacobian(new_joint_values) @ vel
       
        joint_vel = np.clip(joint_vel,
                            [limit[0] for limit in self.joint_vel_limits],
                            [limit[1] for limit in self.joint_vel_limits]
                        )

        # Update the joint angles based on the velocity
        for i in range(self.num_dof):
            new_joint_values[i] += dt * joint_vel[i]

        # Ensure joint angles stay within limits
        new_joint_values = np.clip(new_joint_values,
                               [limit[0] for limit in self.joint_limits],
                               [limit[1] for limit in self.joint_limits]
        )
       
        return new_joint_values


   

    def jacobian(self, joint_values: list):
        """
        Calculates the 6xN Jacobian matrix for any number of joints.
       
        H_cumulative: List of (N+1) matrices [H0_0, H0_1, ..., H0_n]
        joint_types: List of strings ['R', 'R', 'P', ...] where R=Revolute, P=Prismatic.
                    If None, assumes all are Revolute.
        """
        _, h_list = self.calc_forward_kinematics(joint_values)

        H_cumulative = [np.eye(4)]
        for h in h_list:
            H_cumulative.append(H_cumulative[-1] @ h)
        num_joints = len(h_list)
           
        d_n = H_cumulative[-1][:3, 3] # end-effector position
        jacobian = np.zeros((6, num_joints))
       
        for i in range(num_joints):
            H_prev = H_cumulative[i]
            z_prev = H_prev[:3, 2] # z-axis of the previous frame
            d_prev = H_prev[:3, 3] # origin of the previous frame
           
            jv = np.cross(z_prev, (d_n - d_prev))
            jw = z_prev

            jacobian[:3,i] = jv # linear velocity
            jacobian[3:,i] = jw # angular velocity

        return jacobian[:3,:]

   

    def inverse_jacobian(self, joint_values: list):
        """
        Returns the inverse of the Jacobian matrix.

        Returns:
            np.ndarray: The inverse Jacobian matrix.
        """
        jacobian = self.jacobian(joint_values)
        lamda = 0.001
        damped_jacobian = jacobian.T @ np.linalg.pinv(jacobian @ jacobian.T + lamda**2 * np.eye(jacobian.shape[0]))
        return damped_jacobian


if __name__ == "__main__":
    from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
    model = FiveDOFRobot()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    Viz.run()
