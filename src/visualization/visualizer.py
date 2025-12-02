import time
import numpy as np

from scipy.spatial.transform import Rotation as R

import viser
from viser.extras import ViserUrdf

from yourdfpy import URDF

from src.core.robot_loader import ManipulatorRobotURDF


class Visualizer:
    def __init__(
        self,
        robot: ManipulatorRobotURDF,
        # robot_midpoints: np.ndarray,
        robot_cov: np.ndarray,
        curve: np.ndarray,
    ):
        self.robot = robot
        self.n_links = self.robot.get_n_links()
        self.n_joints = self.robot.get_n_joints()

        # self.robot_midpoints = robot_midpoints  # (num_samples, num_links, 3)
        self.robot_cov = robot_cov  # (3, 3)

        self.curve = curve
        self.midpoints = self.get_midpoints(self.curve)
        self.positions = self.get_positions(self.curve)

        self.server = viser.ViserServer()

        urdf = URDF.load(self.robot.get_robot_path())
        self.viser_urdf = ViserUrdf(self.server, urdf_or_path=urdf)

    def visualize_trajectory(self, dt=0.1, loop=True):
        self.viser_urdf.update_cfg(np.zeros(self.n_joints))

        num_samples: int = self.curve.shape[0]
        i = 0

        while True:
            q = self.curve[i]  # (n_joints, )
            self.viser_urdf.update_cfg(q)

            time.sleep(dt)

            i += 1
            if i >= num_samples:
                if loop:
                    i = 0

                else:
                    break

    def visualize_goal(self, goal: np.ndarray, radius: float = 0.03):
        self.server.scene.add_icosphere(
            name="Goal", position=goal, radius=radius, color=[1, 0, 0]
        )

    def visualize_obstacles(
        self, means: np.ndarray, covariances: np.ndarray, radius: float = 1.0
    ):
        for mean in means:
            self.server.scene.add_icosphere(
                name="", position=mean, radius=radius, color=[1, 0, 0]
            )

    def get_positions(self, curve: np.ndarray):
        num_samples, n_joints = curve.shape
        positions = []

        for i in range(num_samples):
            q = curve[i]
            sample_positions = np.array(
                self.robot.forward_kinematics(q)
            )  # (n_links + 1,)

            positions.append(sample_positions)

        return positions

    def get_midpoints(self, curve):
        num_samples = curve.shape[0]
        midpoints = np.zeros((num_samples, self.n_links, 3))

        for i in range(num_samples):
            q = curve[i]
            joint_positions = self.robot.forward_kinematics(q)  # (n_links + 1,)

            for j in range(self.n_links):
                midpoints[i, j, :] = (joint_positions[j] + joint_positions[j + 1]) / 2.0

        return midpoints

    # @staticmethod
    # def covariance_to_ellipsoid(mean, cov, color=[1, 0, 0], scale=1.0):
    #     eigvals, eigvecs = np.linalg.eigh(cov)

    #     # Open3D sphere
    #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    #     sphere.compute_vertex_normals()
    #     sphere.paint_uniform_color(color)

    #     # Scale using eigenvalues
    #     scales = scale * np.sqrt(eigvals)
    #     sphere.scale(1.0, center=np.zeros(3))
    #     sphere.vertices = o3d.utility.Vector3dVector(
    #         np.asarray(sphere.vertices) @ np.diag(scales) @ eigvecs.T
    #     )

    #     sphere.translate(mean)
    #     return sphere
