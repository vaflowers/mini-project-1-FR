from math import *
import math
import numpy as np
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
from funrobo_kinematics.core.arm_models import (
    TwoDOFRobotTemplate, ScaraRobotTemplate, FiveDOFRobotTemplate, KinovaRobotTemplate
)


class Kinova(KinovaRobotTemplate):
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
        for i, theta in enumerate(curr_joint_values):
            curr_joint_values[i] = np.clip(theta, self.joint_limits[i][0], self.joint_limits[i][1])
        
        # DH parameters for each joint
        DH = np.zeros((self.num_dof + 1, 4))
        DH[0] = [0, 0, 0, pi]
        DH[1] = [curr_joint_values[0], -self.l1 - self.l2, 0, pi/2]
        DH[2] = [curr_joint_values[1] - pi/2, 0, self.l3, pi]
        DH[3] = [curr_joint_values[2] - pi/2, 0, 0, pi/2]
        DH[4] = [curr_joint_values[3], -self.l4 - self.l5, 0, -pi/2]
        DH[5] = [curr_joint_values[4], 0, 0, pi/2]
        DH[6] = [curr_joint_values[5], -self.l6 - self.l7, 0, pi]
        

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
    
    # def calc_inverse_kinematics(self, ee, init_joint_values, soln=0):
    #     #currently does not account for multiple solns
    #     #actually change the calcs for different thetas
    #     d5 = self.l4 + self.l5
    #     H0_6 = ut.euler_to_rotm((ee.rotx, ee.roty, ee.rotz))
    #     w_x = ee.x - d5*H0_6[0,3]
    #     w_y = ee.y - d5*H0_6[1,3]
    #     w_z = ee.z - d5*H0_6[2,3]
        
    #     theta1a = math.atan2(ee.x, ee.y)
    #     theta1b = math.atan2(ee.x, ee.y)
        
    #     r = math.sqrt(w_x**2 + w_y**2)
    #     s = w_z - self.l1
    #     theta3a = math.arccos((r**2 + s**2 - self.l3**2 - self.l4**2)/(2*self.l3*self.l4))
    #     theta3b = math.arccos((r**2 + s**2 - self.l3**2 - self.l4**2)/(2*self.l3*self.l4))
    #     theta2a = math.arcsin(((self.l3 + self.l4*math.cos(theta3))*s - self.l4*math.sin(theta3)*r)/(r**2 + s**2))
    #     theta2b= math.arccos(((self.l3 + self.l4*math.cos(theta3))*r - self.l4*math.sin(theta3)*s)/(r**2 + s**2))
        
    #     if soln == 0:
    #         theta1 = theta1a
    #         theta2 = theta2a
    #         theta3 = theta3a

    #     elif soln == 1:
    #         theta1 = theta1a
    #         theta2 = theta2b
    #         theta3 = theta3b

    #     elif soln == 2:
    #         theta1 = theta1b
    #         theta2 = theta2a
    #         theta3 = theta3a

    #     elif soln == 3:
    #         theta1 = theta1b
    #         theta2 = theta2b
    #         theta3 = theta3b
        
    #     R0_1 = self.Rz(theta1)@self.Ry(theta1)@self.Rx(theta1)
    #     R1_2 = self.Rz(theta2)@self.Ry(theta2)@self.Rx(theta2)
    #     R2_3 = self.Rz(theta3)@self.Ry(theta3)@self.Rx(theta3)
    #     R0_3 = R0_1@R1_2@R2_3
    #     R3_6 = R0_3.T @ R0_3
    
    #     theta5 = math.atan2(math.atan2(1-(math.sin(theta1)*H0_6[0])))
    
    def calc_inverse_kinematics(self, ee, joint_values: list, radians=True, soln = 0):
           
            # position and rotation of end effector
            p_ee = np.array([ee.x, ee.y, ee.z])
            r_ee = ut.euler_to_rotm([ee.rotx, ee.roty, ee.rotz])
            #wrist center position
            piv_wrist = p_ee - (self.l6 + self.l7) * (r_ee @ np.array([0, 0, 1]))
            wrist_x, wrist_y, wrist_z = piv_wrist[0], piv_wrist[1], piv_wrist[2]
            # calculate theta1
            solutions = []
            theta_1_opts = [atan2(wrist_y, wrist_x), atan2(-wrist_y, -wrist_x)]
            for theta1 in theta_1_opts:
                # triangle
                r = wrist_x * cos(theta1) + wrist_y * sin(theta1)
                s = (self.l1 + self.l2) - wrist_z
                L_sq = r**2 + s**2
               
                l2 = self.l2
                l3 = self.l3
               
                # Law of Cosines
                numerator = l2**2 + l3**2 - L_sq
                denominator = 2 * l2 * l3
               
                # ensure target is reachable
                if abs(numerator) > abs(denominator):
                    continue
                   
                cos_beta = numerator / denominator
                beta = np.arccos(cos_beta)
           
                # theta 3 solutions
                for theta_3_candidate in [np.pi - beta, -(np.pi - beta)]:
                    theta3 = theta_3_candidate + (np.pi / 2)
                   
                    # angular offset
                    alpha = np.arctan2(l3 * np.sin(theta3), l2 + l3 * np.cos(theta3))

                    # angular trajectory
                    gamma = np.arctan2(s, r)

                    # theta 2
                    theta2 = (np.pi / 2) + (gamma - alpha)
                   
                    # solve for wrist angles
                    q_list = [theta1, theta2, theta3, 0, 0]
                   
                    H_cumulative, _ = self.compute_transformation_matrices(q_list)
                    R3 = H_cumulative[3][:3, :3]
                   
                    R36 = R3.T @ r_ee
                   
                    theta6 = atan2(R36[2, 1], -R36[2, 0])
                    theta5 = atan2(sqrt(R36[2,0]**2 + R36[2,1]**2), -R36[2,2])
                    theta4 = atan2(R36[1, 2], R36[0, 2])
                   
                    candidate_q = [theta1, theta2, theta3, theta4, theta5, theta6]
                    candidate_q = [self.normalized_angle(q) for q in candidate_q]
                   
                    # set limits
                    if ut.check_joint_limits(candidate_q, self.joint_limits):
                        solutions.append(candidate_q)
            def calc_error(q):
                ee_curr, _ = self.calc_forward_kinematics(q)
                return np.linalg.norm(np.array([ee.x - ee_curr.x, ee.y - ee_curr.y, ee.z - ee_curr.z]))
           
            solutions.sort(key=calc_error)
            if not solutions:
                return np.zeros(5)
            if soln < len(solutions):
                return solutions[soln]
            return solutions[0]
        
    def compute_transformation_matrices(self, joint_values):
    
        theta = joint_values
        DH = np.array([
            [theta[0], self.l1, 0, -np.pi/2],
            [theta[1]-(np.pi/2), 0, self.l2, np.pi],
            [theta[2], 0, self.l3, np.pi],
            [theta[3]+(np.pi/2), 0, 0, np.pi/2],
            [theta[4], self.l4+self.l5, 0, 0]
        ])

        # compute transformation matrices for each joint
        Hlist = [ut.dh_to_matrix(dh) for dh in DH] 

        # compute cumulative transformations
        H_cumulative = [np.eye(4)]
        for H in Hlist:
            H_cumulative.append(H_cumulative[-1] @ H)
            
        return H_cumulative, Hlist
    
    def normalized_angle(self, angle):
        return(angle + np.pi) % (2 * np.pi) - np.pi  
        
    
        
        
    


if __name__ == "__main__":
    model = Kinova()
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()