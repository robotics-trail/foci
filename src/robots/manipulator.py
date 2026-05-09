from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import casadi as cas
import numpy as np
from urdf2casadi.urdfparser import URDFparser

from .base import BaseRobot


@dataclass(frozen=True)
class LinkGaussian:
    link: int
    t: float = 0.5
    covariance: np.ndarray = field(
        default_factory=lambda: np.eye(3) * 0.1**2
    )
    name: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.t <= 1.0:
            raise ValueError(
                f"Gaussian interpolation t must be in [0, 1], got {self.t}."
            )

        covariance = np.asarray(self.covariance, dtype=float)

        if covariance.shape != (3, 3):
            raise ValueError(
                f"Gaussian covariance must have shape (3, 3), got {covariance.shape}."
            )

        object.__setattr__(self, "covariance", covariance)


class URDFBackend:
    """
    Small backend responsible only for URDF parsing and kinematics.

    This class owns:
    - URDFparser
    - kinematic chain
    - cached FK functions
    - end-effector visual offset
    - joint limits

    It does not know anything about planning, costs, solvers, or Gaussian
    placement policy.
    """

    def __init__(
        self,
        urdf_path: str,
        root_link: str,
        tip_link: str,
    ):
        if not Path(urdf_path).is_file():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        self.urdf_path = urdf_path
        self.root_link = root_link
        self.tip_link = tip_link

        self.parser = URDFparser()
        self.parser.from_file(urdf_path)

        chain = self.parser.robot_desc.get_chain(root_link, tip_link)
        self.links = [
            item
            for item in chain
            if item in self.parser.robot_desc.link_map
        ]

        self.joint_map = self.parser.robot_desc.joint_map

        self.n_dof = self.parser.get_n_joints(root_link, tip_link)
        self.n_links = len(self.links)

        self.link_fk_funcs, self.link_joint_counts = self._build_link_fk_cache()

        self.tip_link_object = self.parser.robot_desc.link_map.get(self.tip_link)
        self.ee_offset = self._compute_ee_visual_offset()

    def _build_link_fk_cache(self) -> tuple[dict[str, Any], dict[str, int]]:
        link_fk_funcs = {}
        link_joint_counts = {}

        for link in self.links:
            fk_dict = self.parser.get_forward_kinematics(
                self.root_link,
                link,
            )
            link_fk_funcs[link] = fk_dict["T_fk"]

            joint_info = self.parser.get_joint_info(
                self.root_link,
                link,
            )
            link_joint_counts[link] = len(joint_info[1])

        return link_fk_funcs, link_joint_counts

    def link_transform(self, q: Any, link_name: str) -> Any:
        """
        Return the homogeneous transform of a link.
        """
        if link_name not in self.link_fk_funcs:
            raise KeyError(f"Unknown link '{link_name}'.")

        transform_fn = self.link_fk_funcs[link_name]
        joint_count = self.link_joint_counts[link_name]

        return transform_fn(q[:joint_count])

    def link_positions(self, q: Any) -> Any:
        """
        Return link origins plus the final visual end-effector point.

        Shape:
            (n_links + 1, 3)
        """
        positions = []

        for link in self.links:
            transform = self.link_transform(q, link)
            positions.append(transform[:3, 3].T)

        ee = self.end_effector_position(q)
        positions.append(ee.T)

        return cas.vertcat(*positions)

    def forward_kinematics(self, q: Any) -> Any:
        """
        Return flattened vector:

            [link_0_xyz, link_1_xyz, ..., link_N_xyz, ee_xyz]
        """
        positions = self.link_positions(q)
        return cas.reshape(positions, 3 * (self.n_links + 1), 1)

    def end_effector_position(self, q: Any) -> Any:
        """
        Return visual end-effector point in world coordinates.
        """
        transform = self.link_transform(q, self.tip_link)

        rotation = transform[:3, :3]
        position = transform[:3, 3]

        return position + rotation @ self.ee_offset

    def _compute_ee_visual_offset(self) -> cas.DM:
        tip = self.tip_link_object

        if tip is None or tip.visual is None:
            return cas.DM([0.0, 0.0, 0.0])

        geometry = tip.visual.geometry
        origin = tip.visual.origin

        local_offset = np.zeros(3)

        if hasattr(geometry, "filename"):
            scale = (
                np.array(geometry.scale)
                if hasattr(geometry, "scale") and geometry.scale is not None
                else np.ones(3)
            )

            if origin is not None and origin.xyz is not None:
                local_offset = np.array(origin.xyz) * scale

        elif hasattr(geometry, "length"):
            length = geometry.length or 0.0
            local_offset = np.array([0.0, 0.0, length / 2.0])

            if origin is not None and origin.xyz is not None:
                local_offset += np.array(origin.xyz)

        elif hasattr(geometry, "size"):
            size = np.array(geometry.size)
            local_offset = np.array([0.0, 0.0, size[2] / 2.0])

            if origin is not None and origin.xyz is not None:
                local_offset += np.array(origin.xyz)

        elif hasattr(geometry, "radius"):
            radius = geometry.radius or 0.0
            local_offset = np.array([0.0, 0.0, radius])

            if origin is not None and origin.xyz is not None:
                local_offset += np.array(origin.xyz)

        else:
            if origin is not None and origin.xyz is not None:
                local_offset = np.array(origin.xyz)

        return cas.DM(local_offset)

    def joint_limits(self) -> list[tuple[float, float]]:
        limits = []

        for _, joint in self.joint_map.items():
            if joint.type in ["revolute", "prismatic"]:
                lower = joint.limit.lower if joint.limit is not None else -np.inf
                upper = joint.limit.upper if joint.limit is not None else np.inf
                limits.append((lower, upper))

        return limits

    def get_links(self) -> list[str]:
        return self.links

    def get_n_links(self) -> int:
        return self.n_links


class ManipulatorRobot(BaseRobot):
    """
    High-level manipulator model used by the planner.

    Responsibilities:
    - expose the common robot API;
    - define where the robot Gaussian collision points are placed;
    - delegate URDF and FK details to URDFBackend.
    """

    def __init__(
        self,
        urdf_path: str,
        root_link: str = "base_link",
        tip_link: str = "end_effector",
        gaussian_specs: list[LinkGaussian | dict | tuple[int, float, np.ndarray]] | None = None,
    ):

        self.urdf_backend = URDFBackend(
            urdf_path=urdf_path,
            root_link=root_link,
            tip_link=tip_link,
        )

        self.gaussian_specs = self._parse_gaussian_specs(gaussian_specs)

    @property
    def n_dof(self) -> int:
        return self.urdf_backend.n_dof

    @property
    def n_links(self) -> int:
        return self.urdf_backend.n_links

    @property
    def links(self) -> list[str]:
        return self.urdf_backend.links

    @property
    def urdf_path(self) -> str:
        return self.urdf_backend.urdf_path

    @property
    def root_link(self) -> str:
        return self.urdf_backend.root_link

    @property
    def tip_link(self) -> str:
        return self.urdf_backend.tip_link

    def f_task(self, q: Any) -> Any:
        """
        Alias kept for compatibility with older naming.
        """
        return self.urdf_backend.end_effector_position(q)

    def collision_points(self, q: Any) -> Any:
        """
        Return user-defined Gaussian collision points.

        Shape:
            (n_gaussians, 3)
        """
        fk_points = self.urdf_backend.link_positions(q)

        points = []

        for spec in self.gaussian_specs:
            p0 = fk_points[spec.link, :]
            p1 = fk_points[spec.link + 1, :]
            point = (1.0 - spec.t) * p0 + spec.t * p1
            points.append(point)

        if not points:
            raise ValueError(
                "No gaussian_specs were defined. "
                "Pass at least one LinkGaussian to ManipulatorRobot."
            )

        return cas.vertcat(*points)

    def forward_kinematics(self, q: Any) -> Any:
        """
        Return flattened FK vector from the URDF backend.
        """
        self.validate_q(q)
        return self.urdf_backend.forward_kinematics(q)

    def collision_covariances(self) -> np.ndarray:
        """
        Return one covariance matrix per collision Gaussian.

        Shape:
            (n_gaussians, 3, 3)
        """
        return np.stack(
            [spec.covariance for spec in self.gaussian_specs],
            axis=0,
        )

    def joint_limits(self) -> list[tuple[float, float]]:
        return self.urdf_backend.joint_limits()

    def get_links(self) -> list[str]:
        return self.urdf_backend.get_links()

    def get_n_links(self) -> int:
        return self.urdf_backend.get_n_links()

    def _parse_gaussian_specs(
        self,
        gaussian_specs: list[LinkGaussian | dict | tuple] | None,
    ) -> list[LinkGaussian]:
        if gaussian_specs is None:
            return [
                LinkGaussian(
                    link=i,
                    t=0.5,
                    covariance=np.eye(3) * 0.1**2,
                    name=f"link_{i}_mid",
                )
                for i in range(self.n_links)
            ]

        parsed = []

        for idx, spec in enumerate(gaussian_specs):
            if isinstance(spec, LinkGaussian):
                gaussian = spec

            elif isinstance(spec, dict):
                gaussian = LinkGaussian(
                    link=int(spec["link"]),
                    t=float(spec.get("t", 0.5)),
                    covariance=np.asarray(
                        spec.get("covariance", np.eye(3) * 0.1**2),
                        dtype=float,
                    ),
                    name=spec.get("name", f"gaussian_{idx}"),
                )

            else:
                if len(spec) == 2:
                    link, t = spec
                    covariance = np.eye(3) * 0.1**2
                elif len(spec) == 3:
                    link, t, covariance = spec
                else:
                    raise ValueError(
                        "Gaussian tuple must be (link, t) or (link, t, covariance)."
                    )

                gaussian = LinkGaussian(
                    link=int(link),
                    t=float(t),
                    covariance=np.asarray(covariance, dtype=float),
                    name=f"gaussian_{idx}",
                )

            if gaussian.link < 0 or gaussian.link >= self.n_links:
                raise ValueError(
                    f"Gaussian link index {gaussian.link} is invalid. "
                    f"Expected a value in [0, {self.n_links - 1}]."
                )

            parsed.append(gaussian)

        return parsed