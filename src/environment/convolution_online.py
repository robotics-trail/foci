import math

import casadi as cas
import numpy as np
import warp as wp

wp.init()
wp.set_device("cpu")

EPS = 1e-9


@wp.func
def det33(m: wp.mat33) -> wp.float32:
    return (
        m[0, 0] * (m[1, 1] * m[2, 2] - m[2, 1] * m[1, 2])
        - m[0, 1] * (m[1, 0] * m[2, 2] - m[2, 0] * m[1, 2])
        + m[0, 2] * (m[1, 0] * m[2, 1] - m[2, 0] * m[1, 1])
    )


@wp.func
def inv33(m: wp.mat33) -> wp.mat33:
    det = det33(m)
    inv_det = 1.0 / det

    inv = wp.mat33()
    inv[0, 0] = (m[1, 1] * m[2, 2] - m[2, 1] * m[1, 2]) * inv_det
    inv[0, 1] = (m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2]) * inv_det
    inv[0, 2] = (m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]) * inv_det
    inv[1, 0] = (m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2]) * inv_det
    inv[1, 1] = (m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0]) * inv_det
    inv[1, 2] = (m[1, 0] * m[0, 2] - m[0, 0] * m[1, 2]) * inv_det
    inv[2, 0] = (m[1, 0] * m[2, 1] - m[2, 0] * m[1, 1]) * inv_det
    inv[2, 1] = (m[2, 0] * m[0, 1] - m[0, 0] * m[2, 1]) * inv_det
    inv[2, 2] = (m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1]) * inv_det
    return inv


@wp.func
def regularize33(m: wp.mat33, eps: wp.float32) -> wp.mat33:
    out = m
    out[0, 0] = out[0, 0] + eps
    out[1, 1] = out[1, 1] + eps
    out[2, 2] = out[2, 2] + eps
    return out


class ConvolutionFunctorWarpOnline(cas.Callback):
    """Gaussian convolution callback with dynamic robot covariances.

    Inputs
    ------
    0. robot_means: shape (num_points, 3)
    1. robot_covs_flat: shape (num_points, 9), row-major per covariance
    2. obstacle_means: shape (num_obstacles, 3)
    3. obstacle_covs_flat: shape (num_obstacles, 9), row-major per covariance

    Output
    ------
    scalar mean convolution cost.
    """

    def __init__(self, name: str, num_points: int, num_obstacles: int, opts=None):
        cas.Callback.__init__(self)

        if opts is None:
            opts = {}

        self.pi_cubic = float(math.sqrt((2.0 * math.pi) ** 3))
        self.num_points = int(num_points)
        self.num_obstacles = int(num_obstacles)
        self.jacobian_callback = None
        self.intermediate = None

        self.construct(name, opts)

    @wp.kernel
    def kernel(
        robot_means: wp.array(dtype=wp.vec3),
        robot_covs_flat: wp.array(dtype=wp.float32),
        obstacle_means: wp.array(dtype=wp.vec3),
        obstacle_covs_flat: wp.array(dtype=wp.float32),
        intermediate: wp.array(dtype=wp.float32),
        pi_cubic: wp.float32,
        eps: wp.float32,
    ):
        m, n = wp.tid()

        diff = robot_means[m] - obstacle_means[n]

        robot_idx = m * 9
        obstacle_idx = n * 9

        robot_cov = wp.mat33(
            robot_covs_flat[robot_idx + 0],
            robot_covs_flat[robot_idx + 1],
            robot_covs_flat[robot_idx + 2],
            robot_covs_flat[robot_idx + 3],
            robot_covs_flat[robot_idx + 4],
            robot_covs_flat[robot_idx + 5],
            robot_covs_flat[robot_idx + 6],
            robot_covs_flat[robot_idx + 7],
            robot_covs_flat[robot_idx + 8],
        )

        obstacle_cov = wp.mat33(
            obstacle_covs_flat[obstacle_idx + 0],
            obstacle_covs_flat[obstacle_idx + 1],
            obstacle_covs_flat[obstacle_idx + 2],
            obstacle_covs_flat[obstacle_idx + 3],
            obstacle_covs_flat[obstacle_idx + 4],
            obstacle_covs_flat[obstacle_idx + 5],
            obstacle_covs_flat[obstacle_idx + 6],
            obstacle_covs_flat[obstacle_idx + 7],
            obstacle_covs_flat[obstacle_idx + 8],
        )

        cov_sum = regularize33(robot_cov + obstacle_cov, eps)
        det = det33(cov_sum)
        inv = inv33(cov_sum)

        pdf = wp.exp(-0.5 * wp.dot(diff, inv @ diff)) / (wp.sqrt(det) * pi_cubic)
        wp.atomic_add(intermediate, n, pdf)

    def get_n_in(self):
        return 4

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, i):
        if i == 0:
            return cas.Sparsity.dense(self.num_points, 3)
        if i == 1:
            return cas.Sparsity.dense(self.num_points, 9)
        if i == 2:
            return cas.Sparsity.dense(self.num_obstacles, 3)
        if i == 3:
            return cas.Sparsity.dense(self.num_obstacles, 9)
        return cas.Sparsity.dense(0, 0)

    def get_sparsity_out(self, i):
        return cas.Sparsity.dense(1, 1)

    def eval(self, arg):
        robot_means = np.asarray(arg[0], dtype=np.float32).reshape(self.num_points, 3)
        robot_covs = np.asarray(arg[1], dtype=np.float32).reshape(self.num_points, 9)
        obstacle_means = np.asarray(arg[2], dtype=np.float32).reshape(self.num_obstacles, 3)
        obstacle_covs = np.asarray(arg[3], dtype=np.float32).reshape(self.num_obstacles, 9)

        if self.intermediate is None or self.intermediate.shape[0] != self.num_obstacles:
            self.intermediate = wp.zeros(self.num_obstacles, dtype=wp.float32)

        robot_means_wp = wp.from_numpy(robot_means, dtype=wp.vec3)
        robot_covs_wp = wp.from_numpy(robot_covs.reshape(-1), dtype=wp.float32)
        obstacle_means_wp = wp.from_numpy(obstacle_means, dtype=wp.vec3)
        obstacle_covs_wp = wp.from_numpy(obstacle_covs.reshape(-1), dtype=wp.float32)

        self.intermediate.zero_()

        wp.launch(
            kernel=self.kernel,
            dim=(self.num_points, self.num_obstacles),
            inputs=[
                robot_means_wp,
                robot_covs_wp,
                obstacle_means_wp,
                obstacle_covs_wp,
                self.intermediate,
                wp.float32(self.pi_cubic),
                wp.float32(EPS),
            ],
        )

        total = wp.utils.array_sum(self.intermediate)
        result = float(total) / float(self.num_points * self.num_obstacles)
        return [np.array([[result]], dtype=np.float64)]

    def has_jacobian(self):
        return True

    def get_jacobian(self, name, inames, onames, opts):
        if self.jacobian_callback is None:
            self.jacobian_callback = JacobianFunctionOnline(
                name + "_jac",
                self.num_points,
                self.num_obstacles,
                opts,
            )
        return self.jacobian_callback


class JacobianFunctionOnline(cas.Callback):
    """Jacobian of ConvolutionFunctorWarpOnline.

    The callback returns four Jacobian blocks, one for each input:

        dcost/drobot_means
        dcost/drobot_covs_flat
        dcost/dobstacle_means
        dcost/dobstacle_covs_flat

    Returning the robot covariance block is essential because robot covariances
    depend on q in problem_online.py.
    """

    def __init__(self, name: str, num_points: int, num_obstacles: int, opts=None):
        cas.Callback.__init__(self)

        if opts is None:
            opts = {}

        self.pi_cubic = float(math.sqrt((2.0 * math.pi) ** 3))
        self.num_points = int(num_points)
        self.num_obstacles = int(num_obstacles)
        self.grad_means = None
        self.grad_robot_covs = None
        self.grad_obstacle_means = None
        self.grad_obstacle_covs = None

        self.construct(name, opts)

    @wp.kernel
    def kernel_jac(
        robot_means: wp.array(dtype=wp.vec3),
        robot_covs_flat: wp.array(dtype=wp.float32),
        obstacle_means: wp.array(dtype=wp.vec3),
        obstacle_covs_flat: wp.array(dtype=wp.float32),
        grad_means: wp.array(dtype=wp.vec3),
        grad_robot_covs: wp.array(dtype=wp.float32),
        grad_obstacle_means: wp.array(dtype=wp.vec3),
        grad_obstacle_covs: wp.array(dtype=wp.float32),
        pi_cubic: wp.float32,
        eps: wp.float32,
        normalizer: wp.float32,
    ):
        m, n = wp.tid()

        diff = robot_means[m] - obstacle_means[n]

        robot_idx = m * 9
        obstacle_idx = n * 9

        robot_cov = wp.mat33(
            robot_covs_flat[robot_idx + 0],
            robot_covs_flat[robot_idx + 1],
            robot_covs_flat[robot_idx + 2],
            robot_covs_flat[robot_idx + 3],
            robot_covs_flat[robot_idx + 4],
            robot_covs_flat[robot_idx + 5],
            robot_covs_flat[robot_idx + 6],
            robot_covs_flat[robot_idx + 7],
            robot_covs_flat[robot_idx + 8],
        )

        obstacle_cov = wp.mat33(
            obstacle_covs_flat[obstacle_idx + 0],
            obstacle_covs_flat[obstacle_idx + 1],
            obstacle_covs_flat[obstacle_idx + 2],
            obstacle_covs_flat[obstacle_idx + 3],
            obstacle_covs_flat[obstacle_idx + 4],
            obstacle_covs_flat[obstacle_idx + 5],
            obstacle_covs_flat[obstacle_idx + 6],
            obstacle_covs_flat[obstacle_idx + 7],
            obstacle_covs_flat[obstacle_idx + 8],
        )

        cov_sum = regularize33(robot_cov + obstacle_cov, eps)
        det = det33(cov_sum)
        inv = inv33(cov_sum)

        inv_diff = inv @ diff
        pdf = wp.exp(-0.5 * wp.dot(diff, inv_diff)) / (wp.sqrt(det) * pi_cubic)
        pdf = pdf * normalizer

        # d pdf / d mu_robot = -pdf * inv * diff
        grad_mu = -pdf * inv_diff
        wp.atomic_add(grad_means, m, grad_mu)
        wp.atomic_add(grad_obstacle_means, n, -grad_mu)

        # d pdf / d Sigma = 0.5 * pdf * (inv*diff*diff^T*inv - inv)
        # Since inv is symmetric, inv*diff*diff^T*inv = inv_diff*inv_diff^T.
        cov_grad = wp.mat33()
        cov_grad[0, 0] = 0.5 * pdf * (inv_diff[0] * inv_diff[0] - inv[0, 0])
        cov_grad[0, 1] = 0.5 * pdf * (inv_diff[0] * inv_diff[1] - inv[0, 1])
        cov_grad[0, 2] = 0.5 * pdf * (inv_diff[0] * inv_diff[2] - inv[0, 2])
        cov_grad[1, 0] = 0.5 * pdf * (inv_diff[1] * inv_diff[0] - inv[1, 0])
        cov_grad[1, 1] = 0.5 * pdf * (inv_diff[1] * inv_diff[1] - inv[1, 1])
        cov_grad[1, 2] = 0.5 * pdf * (inv_diff[1] * inv_diff[2] - inv[1, 2])
        cov_grad[2, 0] = 0.5 * pdf * (inv_diff[2] * inv_diff[0] - inv[2, 0])
        cov_grad[2, 1] = 0.5 * pdf * (inv_diff[2] * inv_diff[1] - inv[2, 1])
        cov_grad[2, 2] = 0.5 * pdf * (inv_diff[2] * inv_diff[2] - inv[2, 2])

        wp.atomic_add(grad_robot_covs, robot_idx + 0, cov_grad[0, 0])
        wp.atomic_add(grad_robot_covs, robot_idx + 1, cov_grad[0, 1])
        wp.atomic_add(grad_robot_covs, robot_idx + 2, cov_grad[0, 2])
        wp.atomic_add(grad_robot_covs, robot_idx + 3, cov_grad[1, 0])
        wp.atomic_add(grad_robot_covs, robot_idx + 4, cov_grad[1, 1])
        wp.atomic_add(grad_robot_covs, robot_idx + 5, cov_grad[1, 2])
        wp.atomic_add(grad_robot_covs, robot_idx + 6, cov_grad[2, 0])
        wp.atomic_add(grad_robot_covs, robot_idx + 7, cov_grad[2, 1])
        wp.atomic_add(grad_robot_covs, robot_idx + 8, cov_grad[2, 2])

        wp.atomic_add(grad_obstacle_covs, obstacle_idx + 0, cov_grad[0, 0])
        wp.atomic_add(grad_obstacle_covs, obstacle_idx + 1, cov_grad[0, 1])
        wp.atomic_add(grad_obstacle_covs, obstacle_idx + 2, cov_grad[0, 2])
        wp.atomic_add(grad_obstacle_covs, obstacle_idx + 3, cov_grad[1, 0])
        wp.atomic_add(grad_obstacle_covs, obstacle_idx + 4, cov_grad[1, 1])
        wp.atomic_add(grad_obstacle_covs, obstacle_idx + 5, cov_grad[1, 2])
        wp.atomic_add(grad_obstacle_covs, obstacle_idx + 6, cov_grad[2, 0])
        wp.atomic_add(grad_obstacle_covs, obstacle_idx + 7, cov_grad[2, 1])
        wp.atomic_add(grad_obstacle_covs, obstacle_idx + 8, cov_grad[2, 2])

    def get_n_in(self):
        # Original 4 inputs + original scalar output.
        return 5

    def get_n_out(self):
        # One Jacobian block per input of the original callback.
        return 4

    def get_sparsity_in(self, i):
        if i == 0:
            return cas.Sparsity.dense(self.num_points, 3)
        if i == 1:
            return cas.Sparsity.dense(self.num_points, 9)
        if i == 2:
            return cas.Sparsity.dense(self.num_obstacles, 3)
        if i == 3:
            return cas.Sparsity.dense(self.num_obstacles, 9)
        if i == 4:
            return cas.Sparsity.dense(1, 1)
        return cas.Sparsity.dense(0, 0)

    def get_sparsity_out(self, i):
        if i == 0:
            return cas.Sparsity.dense(1, self.num_points * 3)
        if i == 1:
            return cas.Sparsity.dense(1, self.num_points * 9)
        if i == 2:
            return cas.Sparsity.dense(1, self.num_obstacles * 3)
        if i == 3:
            return cas.Sparsity.dense(1, self.num_obstacles * 9)
        return cas.Sparsity.dense(0, 0)

    def eval(self, arg):
        robot_means = np.asarray(arg[0], dtype=np.float32).reshape(self.num_points, 3)
        robot_covs = np.asarray(arg[1], dtype=np.float32).reshape(self.num_points, 9)
        obstacle_means = np.asarray(arg[2], dtype=np.float32).reshape(self.num_obstacles, 3)
        obstacle_covs = np.asarray(arg[3], dtype=np.float32).reshape(self.num_obstacles, 9)

        if self.grad_means is None or self.grad_means.shape[0] != self.num_points:
            self.grad_means = wp.zeros(self.num_points, dtype=wp.vec3)
            self.grad_robot_covs = wp.zeros(self.num_points * 9, dtype=wp.float32)

        if self.grad_obstacle_means is None or self.grad_obstacle_means.shape[0] != self.num_obstacles:
            self.grad_obstacle_means = wp.zeros(self.num_obstacles, dtype=wp.vec3)
            self.grad_obstacle_covs = wp.zeros(self.num_obstacles * 9, dtype=wp.float32)

        robot_means_wp = wp.from_numpy(robot_means, dtype=wp.vec3)
        robot_covs_wp = wp.from_numpy(robot_covs.reshape(-1), dtype=wp.float32)
        obstacle_means_wp = wp.from_numpy(obstacle_means, dtype=wp.vec3)
        obstacle_covs_wp = wp.from_numpy(obstacle_covs.reshape(-1), dtype=wp.float32)

        self.grad_means.zero_()
        self.grad_robot_covs.zero_()
        self.grad_obstacle_means.zero_()
        self.grad_obstacle_covs.zero_()

        normalizer = 1.0 / float(self.num_points * self.num_obstacles)

        wp.launch(
            kernel=self.kernel_jac,
            dim=(self.num_points, self.num_obstacles),
            inputs=[
                robot_means_wp,
                robot_covs_wp,
                obstacle_means_wp,
                obstacle_covs_wp,
                self.grad_means,
                self.grad_robot_covs,
                self.grad_obstacle_means,
                self.grad_obstacle_covs,
                wp.float32(self.pi_cubic),
                wp.float32(EPS),
                wp.float32(normalizer),
            ],
        )

        jac_robot_means = self.grad_means.numpy().reshape(1, self.num_points * 3)
        jac_robot_covs = self.grad_robot_covs.numpy().reshape(1, self.num_points * 9)
        jac_obstacle_means = self.grad_obstacle_means.numpy().reshape(1, self.num_obstacles * 3)
        jac_obstacle_covs = self.grad_obstacle_covs.numpy().reshape(1, self.num_obstacles * 9)

        return [
            jac_robot_means.astype(np.float64),
            jac_robot_covs.astype(np.float64),
            jac_obstacle_means.astype(np.float64),
            jac_obstacle_covs.astype(np.float64),
        ]
