import math
import numpy as np
import funrobo_kinematics.core.utils as ut
from funrobo_kinematics.core.visualizer import Visualizer, RobotSim
from funrobo_kinematics.core.arm_models import FiveDOFRobotTemplate



class FiveDOFRobot(FiveDOFRobotTemplate):
    def __init__(self):
        super().__init__()
    

    def calc_forward_kinematics(self, joint_values: list, radians=True):
        """
        Calculate forward kinematics based on the provided joint angles.
        
        Args:
            theta: List of joint angles (in degrees or radians).
            radians: Boolean flag to indicate if input angles are in radians.
        """
        curr_joint_values = joint_values.copy()
        
        if not radians: # Convert degrees to radians if the input is in degrees
            curr_joint_values = [np.deg2rad(theta) for theta in curr_joint_values]

        # Set the Denavit-Hartenberg parameters for each joint
        DH = np.zeros((self.num_dof, 4)) # [theta, d, a, alpha]
        DH[0] = [curr_joint_values[0], self.l1, 0, -np.pi/2]
        DH[1] = [curr_joint_values[1] - np.pi/2, 0, self.l2, np.pi]
        DH[2] = [curr_joint_values[2], 0, self.l3, np.pi]
        DH[3] = [curr_joint_values[3] + np.pi/2, 0, 0, np.pi/2]
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
        
        print(f"x: {ee.x}, y: {ee.y}, z:{ee.z}")

        return ee, Hlist
    

    def calc_velocity_kinematics(self, joint_values: list, vel: list, dt=0.02):
        """
        Calculates the velocity kinematics for the robot based on the given velocity input.

        Args:
            vel (list): The velocity vector for the end effector [vx, vy, vz].
        """
        new_joint_values = joint_values.copy()

        # move robot slightly out of zeros singularity
        if all(theta == 0.0 for theta in new_joint_values):
            new_joint_values = [theta + np.random.rand()*0.02 for theta in new_joint_values]
        
        # Calculate the joint velocity using the inverse Jacobian
        joint_vel = self.damped_inverse_jacobian(new_joint_values) @ vel

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
        Compute the Jacobian matrix for the current robot configuration.

        Args:
            joint_values (list): The joint angles for the robot.

        Returns:
            Jacobian matrix (3x5).
        """
        _, Hlist = self.calc_forward_kinematics(joint_values)

        # Precompute transformation matrices for efficiency
        H_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            H_cumulative.append(H_cumulative[-1] @ Hlist[i])

        # Define O0 for calculations
        O0 = np.array([0, 0, 0, 1])
        
        # Initialize the Jacobian matrix
        jacobian = np.zeros((3, self.num_dof))

        # Calculate the Jacobian columns
        for i in range(self.num_dof):
            H_curr = H_cumulative[i]
            H_final = H_cumulative[-1]
            
            # Calculate position vector r
            r = (H_final @ O0 - H_curr @ O0)[:3]

            # Compute the rotation axis z
            z = H_curr[:3, :3] @ np.array([0, 0, 1])

            # Compute linear velocity part of the Jacobian
            jacobian[:, i] = np.cross(z, r)

        # Replace near-zero values with zero, primarily for debugging purposes
        return ut.near_zero(jacobian)
  

    def inverse_jacobian(self, joint_values: list, pseudo=True):
        """
        Compute the inverse of the Jacobian matrix using either pseudo-inverse or regular inverse.
        
        Args:
            pseudo: Boolean flag to use pseudo-inverse (default is False).
        
        Returns:
            The inverse (or pseudo-inverse) of the Jacobian matrix.
        """

        J = self.jacobian(joint_values)

        if pseudo:
            return np.linalg.pinv(self.jacobian3x5(joint_values))
        else:
            return np.linalg.inv(self.jacobian3x5(joint_values))
        
        
    def damped_inverse_jacobian(self, joint_values: list, damping_factor=0.025):
        
        J = self.jacobian(joint_values)
        JT = np.transpose(J)
        I = np.eye(3)
        return JT @ np.linalg.inv(J @ JT + (damping_factor**2)*I)
    
    def calc_inverse_kinematics(self, ee, joint_values: list, radians=True, soln = 0):
        
        # position and rotation of end effector
        p_ee = np.array([ee.x, ee.y, ee.z])
        r_ee = ut.euler_to_rotm([ee.rotx, ee.roty, ee.rotz])
        #wrist center position
        piv_wrist = p_ee - (self.l4 + self.l5) * (r_ee @ np.array([0, 0, 1]))
        wrist_x, wrist_y, wrist_z = piv_wrist[0], piv_wrist[1], piv_wrist[2]
        # calculate theta1
        solutions = []
        theta_1_opts = [math.atan2(wrist_y, wrist_x), math.atan2(-wrist_y, -wrist_x)]
        for theta1 in theta_1_opts:
            # triangle 
            r = wrist_x * np.cos(theta1) + wrist_y * np.sin(theta1)
            s = wrist_z - self.l1 
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
                theta3 = theta_3_candidate
                
                # angular offset 
                alpha = np.arctan2(l3 * np.sin(theta3), l2 + l3 * np.cos(theta3))

                # angular trajectory 
                gamma = np.arctan2(s, r)

                # theta 2
                theta2 = (np.pi / 2) - (gamma - alpha)
                
                # solve for wrist angles
                q_list = [theta1, theta2, theta3, 0, 0]
                
                H_cumulative, _ = self.compute_transformation_matrices(q_list)
                R3 = H_cumulative[3][:3, :3] 
                
                R_35 = R3.T @ r_ee
                
                theta4 = math.atan2(R_35[1, 2], R_35[0, 2])
                theta5 = math.atan2(R_35[2, 0], R_35[2, 1])
                
                candidate_q = [theta1, theta2, theta3, theta4, theta5]
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

        
    def calc_numerical_ik(self,ee,init_joint_values,tol: float = 0.002,ilimit: int = 200):
        error = [100, 100, 100]
        joint_values = init_joint_values.copy()
        while np.linalg.norm(error) > tol:
            #print(f"joint vals: {joint_values}")
            
            for _ in range(ilimit):
                [ee_guess, _] = self.calc_forward_kinematics(joint_values)
                error = np.array([
                    abs(ee_guess.x - ee.x),
                    abs(ee_guess.y - ee.y),
                    abs(ee_guess.z - ee.z),
                ])
                
                #damped inv jacobian
                J = self.jacobian(joint_values)
                joint_values += np.linalg.pinv(J) @ error

                #if converged
                if abs(np.linalg.norm(error)) < tol:
                    print(f"error: {error}")
                    print(f"end effector pose {ee_guess.x}, {ee_guess.y}, {ee_guess.z}")
                    return joint_values
                
            joint_values = np.array([np.random.uniform(low, high) for low, high in self.joint_limits])
        
        
if __name__ == "__main__":
    
    model = FiveDOFRobot()
    
    robot = RobotSim(robot_model=model)
    viz = Visualizer(robot=robot)
    viz.run()