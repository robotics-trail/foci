from dataclasses import dataclass, field
from typing import Any

import casadi as cas
import numpy as np

from .base import BaseRobot


@dataclass(frozen=True)
class DroneGaussian:
    """
    Gaussian collision point attached to the drone body.

    Parameters
    ----------
    offset:
        Local body-frame offset from the drone center.

    covariance:
        Gaussian covariance matrix.

    name:
        Optional human-readable name.
    """

    offset: np.ndarray
    covariance: np.ndarray = field(
        default_factory=lambda: np.eye(3) * 0.1**2
    )
    name: str | None = None

    def __post_init__(self) -> None:
        offset = np.asarray(self.offset, dtype=float).reshape(3)
        covariance = np.asarray(self.covariance, dtype=float)

        if covariance.shape != (3, 3):
            raise ValueError(
                f"Drone Gaussian covariance must have shape (3, 3), "
                f"got {covariance.shape}."
            )

        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "covariance", covariance)


class DroneRobot(BaseRobot):
    """
    Drone model with state:

        q = [x, y, z, yaw]

    The task point is the drone center [x, y, z].

    Collision is represented by body-frame Gaussian points.  The default is a
    generic quadrotor footprint derived from `arm_length`: the body centre plus
    the four rotor hubs at (+-arm_length/sqrt(2), +-arm_length/sqrt(2), 0).

    That default is a stand-in, not a model of any particular airframe: pass
    `gaussian_specs` explicitly to match the real geometry of the URDF being
    used (urdfs/drone_example.urdf, for instance, has its rotors at
    (+-0.16, +-0.13, 0) with a 0.13 m radius).
    """

    def __init__(
        self,
        urdf_path: str,
        arm_length: float = 0.15,
        gaussian_specs: list[DroneGaussian | dict | tuple] | None = None,
        xyz_limits: list[tuple[float, float]] =  [(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)],
        yaw_limits: tuple[float, float] = (-np.pi, np.pi),
    ):
        if arm_length <= 0:
            raise ValueError("arm_length must be positive.")

        self._urdf_path = urdf_path
        self.arm_length = float(arm_length)
        self.gaussian_specs = self._parse_gaussian_specs(gaussian_specs)

        self.xyz_limits = xyz_limits
        self.yaw_limits = yaw_limits

    @property
    def n_dof(self) -> int:
        return 4
    
    @property
    def urdf_path(self) -> str:
        return self._urdf_path


    def f_task(self, q: Any) -> Any:
        """
        Drone task point: center position [x, y, z].
        """
        return cas.vertcat(q[0], q[1], q[2])

    def forward_kinematics(self, q: Any) -> Any:
        """
        For the drone, FK is simply the collision points.
        """
        return self.collision_points(q)

    def collision_points(self, q: Any) -> Any:
        """
        Return world-frame Gaussian centers.

        Shape:
            (n_gaussians, 3)
        """
        x, y, z, yaw = q[0], q[1], q[2], q[3]

        c = cas.cos(yaw)
        s = cas.sin(yaw)

        center = cas.vertcat(x, y, z)

        points = []

        for spec in self.gaussian_specs:
            ox, oy, oz = spec.offset

            # Rotation around z-axis.
            world_offset = cas.vertcat(
                c * ox - s * oy,
                s * ox + c * oy,
                oz,
            )

            point = center + world_offset
            points.append(point.T)

        return cas.vertcat(*points)

    def collision_covariances(self) -> np.ndarray:
        """
        Return one covariance matrix per drone Gaussian.

        Shape:
            (n_gaussians, 3, 3)
        """
        return np.stack(
            [spec.covariance for spec in self.gaussian_specs],
            axis=0,
        )

    def collision_covariances_online(self, q: Any) -> Any: 

        yaw = q[3]

        c, s = cas.cos(yaw), cas.sin(yaw)
        R = cas.vertcat(
            cas.horzcat(c, -s, 0), 
            cas.horzcat(s, c, 0), 
            cas.horzcat(0, 0, 1), 
        )

        covs = []

        for spec in self.gaussian_specs: 
            body_cov = cas.DM(spec.covariance)
            world_cov = R @ body_cov @ R.T

            covs.append(world_cov)

        return cas.vertcat(*covs)


    def joint_limits(self) -> list[tuple[float, float]]:
        return [
            *self.xyz_limits,
            self.yaw_limits,
        ]

    def _parse_gaussian_specs(
        self,
        gaussian_specs: list[DroneGaussian | dict | tuple] | None,
    ) -> list[DroneGaussian]:
        """
        Normalize user-defined drone Gaussian specs.
        """
        if gaussian_specs is None:
            # Generic quadrotor footprint: body centre plus four rotor hubs.
            # A single centre Gaussian would model the whole airframe as one
            # small sphere and let the rotors fly through obstacles.
            reach = self.arm_length / np.sqrt(2.0)

            specs = [
                DroneGaussian(
                    offset=np.array([0.0, 0.0, 0.0]),
                    covariance=np.eye(3) * 0.1**2,
                    name="center",
                )
            ]

            for sign_x, sign_y, name in (
                (+1.0, +1.0, "rotor_front_left"),
                (+1.0, -1.0, "rotor_front_right"),
                (-1.0, +1.0, "rotor_back_left"),
                (-1.0, -1.0, "rotor_back_right"),
            ):
                specs.append(
                    DroneGaussian(
                        offset=np.array([sign_x * reach, sign_y * reach, 0.0]),
                        covariance=np.eye(3) * 0.06**2,
                        name=name,
                    )
                )

            return specs

        parsed = []

        for idx, spec in enumerate(gaussian_specs):
            if isinstance(spec, DroneGaussian):
                gaussian = spec

            elif isinstance(spec, dict):
                gaussian = DroneGaussian(
                    offset=np.asarray(spec["offset"], dtype=float),
                    covariance=np.asarray(
                        spec.get("covariance", np.eye(3) * 0.1**2),
                        dtype=float,
                    ),
                    name=spec.get("name", f"gaussian_{idx}"),
                )

            else:
                if len(spec) == 1:
                    offset = spec[0]
                    covariance = np.eye(3) * 0.1**2
                elif len(spec) == 2:
                    offset, covariance = spec
                else:
                    raise ValueError(
                        "Drone Gaussian tuple must be "
                        "(offset,) or (offset, covariance)."
                    )

                gaussian = DroneGaussian(
                    offset=np.asarray(offset, dtype=float),
                    covariance=np.asarray(covariance, dtype=float),
                    name=f"gaussian_{idx}",
                )

            parsed.append(gaussian)

        return parsed