"""
Generic visualization utilities for robot trajectories and Gaussian models.

This visualizer works with any robot implementing:

    robot.n_dof
    robot.f_task(q)
    robot.collision_points(q)
    robot.collision_covariances()

For manipulators, if the robot has a URDF path, the URDF is shown.
For drones, the trajectory and Gaussian collision model are shown directly.
"""

import sys
import select
import time
from pathlib import Path

import numpy as np
from viser import ViserServer
from viser.extras import ViserUrdf
from yourdfpy import URDF

from src.visualization.utils import EllipsoidFactory


class RobotVisualizer:
    def __init__(
        self,
        robot,
        trajectory: np.ndarray,
        follow_camera: bool = False,
        camera_offset: np.ndarray | None = None,
        camera_lookahead: float = 0.5,
    ):
        trajectory = np.asarray(trajectory, dtype=float)

        if trajectory.ndim != 2:
            raise ValueError(
                f"trajectory must have shape (num_samples, n_dof), "
                f"got {trajectory.shape}."
            )

        if trajectory.shape[1] != robot.n_dof:
            raise ValueError(
                f"trajectory has {trajectory.shape[1]} columns, "
                f"but robot.n_dof is {robot.n_dof}."
            )

        self.robot = robot

        self.trajectory = trajectory

        self.follow_camera = follow_camera
        self.camera_offset = (
            np.asarray(camera_offset, dtype=float)
            if camera_offset is not None
            else np.array([0.0, 0.0, 1.5])
        )
        self.camera_lookahead = float(camera_lookahead)


        self.server = ViserServer()
        self.robot_urdf = robot.urdf_path

        self._ellipsoid_faces = self._create_ellipsoid_faces()
        self._robot_gaussian_handles = []
        self._ellipsoid_factory = None

        self.gaussian_points = self._compute_robot_gaussian_points()
        self.gaussian_covariances = np.asarray(
            robot.collision_covariances(),
            dtype=float,
        )

        if self.gaussian_covariances.shape[0] != self.gaussian_points.shape[1]:
            raise ValueError(
                "Number of collision covariances must match number of "
                "collision points. "
                f"Got {self.gaussian_covariances.shape[0]} covariances and "
                f"{self.gaussian_points.shape[1]} points."
            )

        if self.robot_urdf is not None: 
            if not Path(self.robot_urdf).is_file():
                raise FileNotFoundError(f"URDF file not found: {self.robot_urdf}")

            self.urdf = URDF.load(self.robot_urdf)
            self.viser_urdf = ViserUrdf(self.server, urdf_or_path=self.urdf, root_node_name="/robot")

            self.robot_frame = self.server.scene.add_frame("/robot",show_axes=False)

        # if self.follow_camera:
        #     q0 = self.trajectory[0]
        #     target0 = self.robot.f_task(q0)

        #     if hasattr(target0, "full"):
        #         target0 = target0.full()

        #     target0 = np.asarray(target0, dtype=float).reshape(3)

        #     self.server.initial_camera.position = target0 + self.camera_offset
        #     self.server.initial_camera.look_at = target0
        

    def _compute_robot_gaussian_points(self) -> np.ndarray:
        """
        Evaluate robot.collision_points(q) for all trajectory samples.

        Returns
        -------
        np.ndarray
            Shape (num_samples, n_gaussians, 3).
        """
        all_points = []

        for q in self.trajectory:
            points = self.robot.collision_points(q)

            if hasattr(points, "full"):
                points = points.full()

            points = np.asarray(points, dtype=float).reshape(-1, 3)
            all_points.append(points)

        return np.stack(all_points, axis=0)

    def visualize_goal(
        self,
        goal: np.ndarray,
        radius: float = 0.05,
        color: tuple[int, int, int] = (0, 0, 255),
    ):
        self.server.scene.add_icosphere(
            name="Goal",
            position=np.asarray(goal, dtype=float),
            radius=radius,
            color=color,
        )

    def visualize_obstacles(
        self,
        means: np.ndarray,
        covariances: np.ndarray,
        n_std: float = 2.0,
        color: tuple[int, int, int] = (255, 100, 100),
        opacity: float = 0.6,
        name: str = "Obstacle",
    ):
        means = np.asarray(means, dtype=float)
        covariances = np.asarray(covariances, dtype=float)

        factory = EllipsoidFactory(n_std=float(n_std))

        for obstacle_idx, (mean, cov) in enumerate(zip(means, covariances)):
            radii, quat_wxyz = factory.cov_to_ellipsoid(cov)

            self.server.scene.add_mesh_simple(
                name=f"{name}_{obstacle_idx}",
                vertices=self._create_ellipsoid_mesh(radii),
                faces=self._ellipsoid_faces,
                position=mean,
                wxyz=quat_wxyz,
                color=color,
                opacity=opacity,
            )

    def visualize_robot_gaussians(
        self,
        n_std: float = 2.0,
        color: tuple[int, int, int] = (80, 160, 255),
        opacity: float = 0.35,
        name: str = "RobotGaussian",
    ):
        """
        Render robot collision Gaussians at the first trajectory sample.
        They are updated during animation.
        """
        self._robot_gaussian_handles = []
        self._ellipsoid_factory = EllipsoidFactory(n_std=float(n_std))

        for gaussian_idx in range(self.gaussian_points.shape[1]):
            mean = self.gaussian_points[0, gaussian_idx]
            cov = self.gaussian_covariances[gaussian_idx]

            radii, quat_wxyz = self._ellipsoid_factory.cov_to_ellipsoid(cov)

            handle = self.server.scene.add_mesh_simple(
                name=f"{name}_{gaussian_idx}",
                vertices=self._create_ellipsoid_mesh(radii),
                faces=self._ellipsoid_faces,
                position=mean,
                wxyz=quat_wxyz,
                color=color,
                opacity=opacity,
            )

            self._robot_gaussian_handles.append(handle)

    def visualize_path(
        self,
        color: tuple[int, int, int] = (50, 200, 50),
        line_width: float = 3.0,
        name: str = "Trajectory",
    ):
        """
        Visualize the task-space trajectory.

        For manipulators this is the end-effector path.
        For drones this is the center path.
        """
        task_points = []

        for q in self.trajectory:
            point = self.robot.f_task(q)

            if hasattr(point, "full"):
                point = point.full()

            task_points.append(np.asarray(point, dtype=float).reshape(3))

        task_points = np.asarray(task_points)

        if len(task_points) < 2:
            return

        segments = np.stack([task_points[:-1], task_points[1:]], axis=1)

        colors = np.tile(
            np.asarray(color, dtype=np.uint8)[None, None, :],
            (segments.shape[0], 2, 1),
        )

        self.server.scene.add_line_segments(
            name=name,
            points=segments,
            colors=colors,
            line_width=line_width,
        )

    def visualize_initializer_path(
        self,
        initial_trajectory: np.ndarray,
        color: tuple[int, int, int] = (255, 0, 0),
        line_width: float = 2.0,
        name: str = "Initializer",
    ):
        initial_trajectory = np.asarray(initial_trajectory, dtype=float)

        task_points = []

        for q in initial_trajectory:
            point = self.robot.f_task(q)

            if hasattr(point, "full"):
                point = point.full()

            task_points.append(np.asarray(point, dtype=float).reshape(3))

        task_points = np.asarray(task_points)

        if len(task_points) < 2:
            return

        segments = np.stack([task_points[:-1], task_points[1:]], axis=1)

        colors = np.tile(
            np.asarray(color, dtype=np.uint8)[None, None, :],
            (segments.shape[0], 2, 1),
        )

        self.server.scene.add_line_segments(
            name=name,
            points=segments,
            colors=colors,
            line_width=line_width,
        )

    def visualize_gaussian_splat(
        self,
        name: str,
        means: np.ndarray,
        covariances: np.ndarray,
        colors: np.ndarray,
        opacities: np.ndarray,
    ):
        self.server.add_gaussian_splats(
            name=name,
            centers=means,
            covariances=covariances,
            rgbs=colors,
            opacities=opacities,
        )

    def visualize_trajectory(
        self,
        dt: float = 0.1,
        loop: bool = True,
    ):
        """
        Animate robot and robot Gaussians.
        """
        sample_idx = 0
        num_samples = self.trajectory.shape[0]
        
        while True:
        
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.readline().strip()
                if key.lower() == "q":
                    print("Stopping visualization.")
                    break

            q = self.trajectory[sample_idx]

            self._update_robot(q)
            self._update_robot_gaussians(sample_idx)
            self._update_camera(sample_idx)

            time.sleep(dt)

            sample_idx += 1

            if sample_idx >= num_samples:
                if loop:
                    sample_idx = 0
                else:
                    break

    def _update_robot(self, q: np.ndarray):
        """
        Update URDF if available.

        For drones without URDF, the Gaussian visualization already shows the
        robot collision model.
        """
        if self.robot_urdf is None:
            return

        robot_type = self.robot.__class__.__name__
        
        if robot_type == "DroneRobot":
            x, y, z, yaw = q

            self.robot_frame.position = np.array([x, y, z])
            self.robot_frame.wxyz = np.array([
                np.cos(yaw / 2.0),
                0.0,
                0.0,
                np.sin(yaw / 2.0),
            ])

            self.viser_urdf.update_cfg(self.urdf.zero_cfg)

            return

        elif robot_type == "MobileRobot":
            x, y, yaw = q

            self.robot_frame.position = np.array([x, y, self.robot.body_center_height])
            self.robot_frame.wxyz = np.array([
                np.cos(yaw / 2.0),
                0.0,
                0.0,
                np.sin(yaw / 2.0),
            ])

            self.viser_urdf.update_cfg(self.urdf.zero_cfg)

            return

        self.viser_urdf.update_cfg(q)

    

    def _update_robot_gaussians(self, sample_idx: int):
        if not self._robot_gaussian_handles or self._ellipsoid_factory is None:
            return

        for gaussian_idx, handle in enumerate(self._robot_gaussian_handles):
            mean = self.gaussian_points[sample_idx, gaussian_idx]
            cov = self.gaussian_covariances[gaussian_idx]

            _, quat_wxyz = self._ellipsoid_factory.cov_to_ellipsoid(cov)

            handle.position = mean
            handle.wxyz = quat_wxyz

    def _update_camera(self, sample_idx: int):
        """
        Optionally keep the viewer camera above the robot/task point.

        Uses robot.f_task(q), so it is robot-agnostic:
        - for drones, this is usually the drone center;
        - for manipulators, this is usually the end-effector.
        """
        if not self.follow_camera:
            return

        q = self.trajectory[sample_idx]
        target = self.robot.f_task(q)

        if hasattr(target, "full"):
            target = target.full()

        target = np.asarray(target, dtype=float).reshape(3)

        camera_position = target + self.camera_offset

        look_at = target #+ np.array([0.1, 0.0, 0.1])
        next_idx = min(sample_idx + 1, self.trajectory.shape[0] - 1)

        if next_idx != sample_idx:
            next_target = self.robot.f_task(self.trajectory[next_idx])

            if hasattr(next_target, "full"):
                next_target = next_target.full()

            next_target = np.asarray(next_target, dtype=float).reshape(3)
            direction = next_target - target
            norm = np.linalg.norm(direction)

            if norm > 1e-9:
                look_at = camera_position + self.camera_lookahead * direction / norm

        for client in self.server.get_clients().values():
            with client.atomic():
                client.camera.position = camera_position
                client.camera.look_at = look_at

    def _create_ellipsoid_mesh(
        self,
        radii: np.ndarray,
        resolution: int = 20,
    ) -> np.ndarray:
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)

        u_grid, v_grid = np.meshgrid(u, v)

        x = radii[0] * np.cos(u_grid) * np.sin(v_grid)
        y = radii[1] * np.sin(u_grid) * np.sin(v_grid)
        z = radii[2] * np.cos(v_grid)

        return np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    def _create_ellipsoid_faces(
        self,
        resolution: int = 20,
    ) -> np.ndarray:
        faces = []

        for i in range(resolution - 1):
            for j in range(resolution - 1):
                idx = i * resolution + j
                faces.append([idx, idx + resolution, idx + 1])
                faces.append([idx + 1, idx + resolution, idx + resolution + 1])

        return np.asarray(faces, dtype=np.uint32)