import os
import numpy as np
import casadi as cas

from urdf2casadi.urdfparser import URDFparser

from typing import List


class BaseRobot:
    """
    Abstract base class for robot models.

    Parameters
    ----------
    robot_path : str
        Path to the robot URDF file.

    Attributes
    ----------
    robot_path : str
        Path to the URDF file.
    n_joints : int | None
        Number of active joints in the modeled chain.
    parser : object | None
        Backend parser used to read the robot description.
    """

    def __init__(self, robot_path: str):
        if not os.path.isfile(robot_path):
            raise FileNotFoundError(f"File '{robot_path}' not found.")

        self.robot_path = robot_path
        self.n_joints = None
        self.parser = None

    def forward_kinematics(self, q):
        """
        Compute forward kinematics for the robot.

        Parameters
        ----------
        q : cas.MX or cas.DM or np.ndarray
            Joint configuration.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError


class ManipulatorRobotURDF(BaseRobot):
    """
    Manipulator robot model built from a URDF file.

    This class parses the URDF, extracts the kinematic chain between
    `root_link` and `tip_link`, caches forward-kinematics functions for each
    link, and exposes helpers used throughout planning and visualization.

    Parameters
    ----------
    robot_path : str
        Path to the URDF file.
    root_link : str, default="base_link"
        Name of the root link of the kinematic chain.
    tip_link : str, default="end_effector"
        Name of the tip link of the kinematic chain.

    Notes
    -----
    `forward_kinematics(q)` returns a flattened CasADi vector containing the
    3D positions of all links in the chain, followed by the final end-effector
    point returned by `get_ee_endpoint(q)`.
    """

    def __init__(
        self,
        robot_path: str,
        root_link: str = "base_link",
        tip_link: str = "end_effector",
    ):
        super().__init__(robot_path)

        self.root_link = root_link
        self.tip_link = tip_link

        self.parser = URDFparser()
        self.parser.from_file(robot_path)

        chain = self.parser.robot_desc.get_chain(root_link, tip_link)
        self.links = [item for item in chain if item in self.parser.robot_desc.link_map]
        self.joint_map = self.parser.robot_desc.joint_map

        self.n_links = len(self.links)
        self.n_joints = self.parser.get_n_joints(root_link, tip_link)

        self.link_fk_funcs, self.link_joint_counts = self._build_link_fk_cache()

        self.tip_link_object = self.parser.robot_desc.link_map.get(self.tip_link)
        self.tip_link_geometry = None
        self.tip_link_origin = None

        if self.tip_link_object and self.tip_link_object.visual:
            self.tip_link_geometry = self.tip_link_object.visual.geometry
            self.tip_link_origin = self.tip_link_object.visual.origin

        self.ee_offset = self._compute_ee_visual_offset()

    def forward_kinematics(self, q):
        """
        Compute the 3D positions of all chain links and the final end-effector point.

        Parameters
        ----------
        q : cas.MX or cas.DM or np.ndarray
            Joint configuration of length `n_joints`.

        Returns
        -------
        cas.MX or cas.DM
            Flattened vector containing:
            [link_0_xyz, link_1_xyz, ..., link_N_xyz, ee_xyz].
            Its total length is 3 * (n_links + 1).
        """
        if q.shape[0] != self.n_joints:
            raise ValueError(f"Lenght of q should be {self.n_joints}, got {q.shape[0]}")

        positions = []

        for link in self.links:
            transform_fn = self.link_fk_funcs[link]
            link_joint_count = self.link_joint_counts[link]
            transform = transform_fn(q[:link_joint_count])

            positions.append(transform[0, 3])
            positions.append(transform[1, 3])
            positions.append(transform[2, 3])

        ee_pos = self.get_ee_endpoint(q)
        positions.append(ee_pos[0])
        positions.append(ee_pos[1])
        positions.append(ee_pos[2])

        return cas.vertcat(*positions)

    def get_ee_endpoint(self, q):
        """
        Compute the final end-effector position in world coordinates.

        If the tip link has visual geometry, this method adds a local offset
        derived from that geometry so the returned point better matches the
        visible end of the tool.

        Parameters
        ----------
        q : cas.MX or cas.DM or np.ndarray
            Joint configuration.

        Returns
        -------
        cas.MX or cas.DM
            3D end-effector point as a column vector of length 3.
        """
        tip_fk = self.link_fk_funcs[self.tip_link]
        transform = tip_fk(q)

        rotation = transform[:3, :3]
        position = transform[:3, 3]

        if self.tip_link_object is None or self.tip_link_object.visual is None:
            final_position = position
        else:
            world_offset = rotation @ self.ee_offset
            final_position = position + world_offset

        return cas.vertcat(
            final_position[0],
            final_position[1],
            final_position[2],
        )

    def _build_link_fk_cache(self):
        """
        Precompute FK functions and required joint counts for each link in the chain.

        Returns
        -------
        tuple[dict, dict]
            - link_fk_funcs: maps each link name to its FK CasADi function.
            - link_joint_counts: maps each link name to the number of joints
              needed to evaluate that FK.
        """
        link_fk_funcs = {}
        link_joint_counts = {}

        for link in self.links:
            fk_dict = self.parser.get_forward_kinematics(self.root_link, link)
            link_fk_funcs[link] = fk_dict["T_fk"]

            joint_info = self.parser.get_joint_info(self.root_link, link)
            link_joint_counts[link] = len(joint_info[1])

        return link_fk_funcs, link_joint_counts

    def _compute_ee_visual_offset(self):
        """
        Compute a local offset for the end-effector based on tip visual geometry.

        The goal is to place the operational end-effector point at the visible tip
        of the last link rather than at the link frame origin.

        Supported geometry heuristics:
        - mesh: uses visual origin, optionally scaled
        - cylinder: places the point at half the cylinder length along +Z
        - box: places the point at half the box height along +Z
        - sphere: places the point at +radius along Z

        Returns
        -------
        cas.DM
            Local 3D offset expressed in the tip-link frame.
        """
        if self.tip_link_object is None or self.tip_link_object.visual is None:
            return cas.DM([0.0, 0.0, 0.0])

        geometry = self.tip_link_geometry
        origin = self.tip_link_origin
        local_offset = np.zeros(3)

        if hasattr(geometry, "filename"):  # Mesh
            scale = (
                np.array(geometry.scale)
                if hasattr(geometry, "scale") and geometry.scale is not None
                else np.ones(3)
            )
            if origin is not None and origin.xyz is not None:
                local_offset = np.array(origin.xyz) * scale

        elif hasattr(geometry, "length"):  # Cylinder
            length = geometry.length if geometry.length is not None else 0.0
            local_offset = np.array([0.0, 0.0, length / 2.0])
            if origin is not None and origin.xyz is not None:
                local_offset += np.array(origin.xyz)

        elif hasattr(geometry, "size"):  # Box
            size = np.array(geometry.size)
            local_offset = np.array([0.0, 0.0, size[2] / 2.0])
            if origin is not None and origin.xyz is not None:
                local_offset += np.array(origin.xyz)

        elif hasattr(geometry, "radius"):  # Sphere
            radius = geometry.radius if geometry.radius is not None else 0.0
            local_offset = np.array([0.0, 0.0, radius])
            if origin is not None and origin.xyz is not None:
                local_offset += np.array(origin.xyz)

        else:
            if origin is not None and origin.xyz is not None:
                local_offset = np.array(origin.xyz)

        return cas.DM(local_offset)

    def get_links(self):
        """Return the ordered list of links in the active kinematic chain."""
        return self.links

    def get_joint_map(self):
        """Return the full URDF joint map provided by the parser."""
        return self.joint_map

    def get_n_links(self) -> int:
        """Return the number of links in the active chain."""
        return self.n_links

    def get_n_joints(self) -> int:
        """Return the number of joints in the active chain."""
        return self.n_joints

    def get_robot_path(self):
        """Return the URDF file path."""
        return self.robot_path

    def get_joint_limits(self):
        """
        Return joint limits for actuated joints.

        Returns
        -------
        list[tuple[float, float]]
            List of (lower, upper) limits for revolute and prismatic joints.
            If a joint has no explicit limit, `-np.inf` / `np.inf` is used.
        """
        joint_limits = []

        for _, joint in self.joint_map.items():
            if joint.type in ["revolute", "prismatic"]:
                lower = joint.limit.lower if joint.limit is not None else -np.inf
                upper = joint.limit.upper if joint.limit is not None else np.inf
                joint_limits.append((lower, upper))

        return joint_limits


class MobileManipulatorRobotURDF(ManipulatorRobotURDF):
    """
    Convenience specialization for mobile manipulators.

    Defaults to a world-fixed root frame.
    """

    def __init__(
        self,
        robot_path: str,
        root_link: str = "world",
        tip_link: str = "end_effector",
    ):
        super().__init__(robot_path, root_link, tip_link)


class DroneRobot:
    def __init__(
        self,
        arm_length: float = 0.15,
    ):
        self.n_joints = 4  # x, y, z, yaw
        self.arm_length = arm_length

    def get_collision_points(self, q):
        x, y, z, yaw = q[0], q[1], q[2], q[3]
        c, s = cas.cos(yaw), cas.sin(yaw)

        center = cas.vertcat(x, y, z)  # (3, 1)
        offset = cas.vertcat(c * self.arm_length, s * self.arm_length, 0)

        left = center - offset  # (3, 1)
        right = center + offset  # (3, 1)

        return cas.vertcat(center.T, left.T, right.T)  # (3, 3)

    def get_goal_point(self, q):
        return cas.vertcat(q[0], q[1], q[2])

    def get_n_joints(self) -> int:
        return self.n_joints
