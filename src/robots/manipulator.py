from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import casadi as cas
import numpy as np
from urdf2casadi.urdfparser import URDFparser

from .base import BaseRobot


@dataclass(frozen=True)
class LinkGaussian:
    """
    `link` accepts either:
    - an int: raw index into the robot's link chain (fragile -- breaks
      silently if the URDF chain changes, e.g. links added/removed/reordered).
    - a str: the URDF link name (e.g. "wrist_3_link"). Resolved to an index
      against the actual robot chain at ManipulatorRobot construction time,
      so it stays correct across URDF changes as long as the link name exists.

    Prefer str whenever possible.
    """

    link: int | str
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

        self.joint_map = {
            item: self.parser.robot_desc.joint_map[item]
            for item in chain
            if item in self.parser.robot_desc.joint_map
        }

        self.n_dof = self.parser.get_n_joints(root_link, tip_link)
        self.n_links = len(self.links)

        self.link_fk_funcs, self.link_joint_counts = self._build_link_fk_cache()

        self.link_offsets: dict[str, cas.DM] = {
            link_name: self._compute_visual_offset(
                self.parser.robot_desc.link_map.get(link_name)
            )
            for link_name in self.links
        }

        self.tip_link_object = self.parser.robot_desc.link_map.get(self.tip_link)
        self.ee_offset = self.link_offsets[self.tip_link]


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
        Return the homogeneous transform of a link, with the visual
        offset already baked into the translation component.
        """
        if link_name not in self.link_fk_funcs:
            raise KeyError(f"Unknown link '{link_name}'.")

        transform_fn = self.link_fk_funcs[link_name]
        joint_count = self.link_joint_counts[link_name]

        T = transform_fn(q[:joint_count])

        offset = self.link_offsets[link_name]
        if cas.norm_2(offset) != 0:
            rotation = T[:3, :3]
            position = T[:3, 3] + rotation @ offset
            T[:3, 3] = position

        return T

    def link_positions(self, q: Any) -> Any:
        """
        Return link origins (already offset-corrected) plus the final
        visual end-effector point.
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
        Kept for backward compatibility — now link_transform() already
        applies the offset, so this is equivalent to taking the tip's
        translation column.
        """
        transform = self.link_transform(q, self.tip_link)
        return transform[:3, 3]


    @staticmethod
    def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        URDF convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
        """
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])

        return Rz @ Ry @ Rx

    def _compute_visual_offset(self, link_object) -> cas.DM:
        """
        Generalized version of the old _compute_ee_visual_offset:
        works for any link, not just the tip.

        Properly accounts for the visual <origin> rotation (rpy), not just
        its translation (xyz) -- otherwise shape-derived offsets (e.g. "half
        the cylinder length along local Z") point in the wrong direction
        whenever the URDF rotates the visual geometry relative to the link
        frame, which is extremely common for cylinder/capsule links.
        """
        if link_object is None or link_object.visual is None:
            return cas.DM([0.0, 0.0, 0.0])

        geometry = link_object.visual.geometry
        origin = link_object.visual.origin

        origin_xyz = np.zeros(3)
        origin_rpy = np.zeros(3)

        if origin is not None:
            if origin.xyz is not None:
                origin_xyz = np.array(origin.xyz)
            if origin.rpy is not None:
                origin_rpy = np.array(origin.rpy)

        rotation = self._rpy_to_matrix(*origin_rpy)

        # Shape-local offset, expressed in the *unrotated* geometry frame
        # (i.e. before applying the visual origin's own rpy).
        shape_offset = np.zeros(3)

        if hasattr(geometry, "filename"):
            # We cannot infer the true mesh centroid from the URDF alone
            # (would require loading and analyzing the mesh file itself).
            # `scale` only rescales mesh vertices, it must NOT be applied
            # to the origin translation.
            shape_offset = np.zeros(3)

        elif hasattr(geometry, "length"):
            length = geometry.length or 0.0
            shape_offset = np.array([0.0, 0.0, length / 2.0])

        elif hasattr(geometry, "size"):
            size = np.array(geometry.size)
            shape_offset = np.array([0.0, 0.0, size[2] / 2.0])

        elif hasattr(geometry, "radius"):
            radius = geometry.radius or 0.0
            shape_offset = np.array([0.0, 0.0, radius])

        local_offset = origin_xyz + rotation @ shape_offset

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

    def collision_covariances_online(self, q):

        links = self.urdf_backend.links
        covs = []

        for spec in self.gaussian_specs: 
            link_name = links[spec.link]
            T = self.urdf_backend.link_transform(q, link_name)
            R = cas.reshape(T[:3, :3], 3, 3)

            body_cov = cas.DM(spec.covariance)
            world_cov = R @ body_cov @ R.T

            covs.append(world_cov)

        return cas.vertcat(*covs)

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
                    link=spec["link"],
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
                    link=link,
                    t=float(t),
                    covariance=np.asarray(covariance, dtype=float),
                    name=f"gaussian_{idx}",
                )

            gaussian = self._resolve_gaussian_link(gaussian)

            parsed.append(gaussian)

        return parsed

    def _resolve_gaussian_link(self, gaussian: LinkGaussian) -> LinkGaussian:
        """
        Resolve LinkGaussian.link to a valid integer index into self.links,
        accepting either a raw index (int) or a URDF link name (str).
        """
        link_ref = gaussian.link

        if isinstance(link_ref, str):
            if link_ref not in self.links:
                raise ValueError(
                    f"Unknown link name '{link_ref}' for gaussian "
                    f"'{gaussian.name}'. Available links: {self.links}"
                )
            resolved_index = self.links.index(link_ref)
            gaussian = replace(gaussian, link=resolved_index)

        else:
            resolved_index = int(link_ref)
            gaussian = replace(gaussian, link=resolved_index)

        if resolved_index < 0 or resolved_index >= self.n_links:
            raise ValueError(
                f"Gaussian link index {resolved_index} is invalid. "
                f"Expected a value in [0, {self.n_links - 1}]. "
                f"Available links: {self.links}"
            )

        return gaussian