import os
import numpy as np
import casadi as cas

from urdf2casadi.urdfparser import URDFparser


class BaseRobot:
    def __init__(self, robot_path):

        if not os.path.isfile(robot_path):
            raise FileNotFoundError(f"File {robot_path} not found")

        self.robot_path = robot_path
        self.n_joints = None
        self.parser = None
        self.fk = None

    def forward_kinematics(self, q):
        raise NotImplementedError


class ManipulatorRobotURDF(BaseRobot):
    def __init__(self, robot_path, root_link="base_link", tip_link="end_effector"):
        super().__init__(robot_path)

        self.parser = URDFparser()
        self.parser.from_file(robot_path)

        self.robot_path = robot_path
        self.root_link = root_link
        self.tip_link = tip_link

        chain = self.parser.robot_desc.get_chain(root_link, tip_link)
        self.links = [item for item in chain if item in self.parser.robot_desc.link_map]

        self.joint_map = self.parser.robot_desc.joint_map

        self.n_links = len(self.links)
        self.n_joints = self.parser.get_n_joints(root_link, tip_link)

        self.link_fk_funcs, self.link_joint_counts = (
            self.precompute_forward_kinematics()
        )

        self.tip_link_object = self.parser.robot_desc.link_map.get(self.tip_link)
        self.tip_link_geometry = None
        self.tip_link_origin = None

        if self.tip_link_object and self.tip_link_object.visual:
            self.tip_link_geometry = self.tip_link_object.visual.geometry
            self.tip_link_origin = self.tip_link_object.visual.origin

        self.ee_offset = self.precompute_ee_offset()

    def forward_kinematics(self, q):
        positions = []
        for link in self.links:
            T_link = self.link_fk_funcs[link]
            link_joint_count = self.link_joint_counts[link]

            T = T_link(q[:link_joint_count])

            positions.append(T[0, 3])
            positions.append(T[1, 3])
            positions.append(T[2, 3])

        # Obtain end effector final position
        ee_pos = self.get_ee_endpoint(q)
        positions.append(ee_pos[0])
        positions.append(ee_pos[1])
        positions.append(ee_pos[2])

        return cas.vertcat(*positions)

    def get_ee_endpoint(self, q):
        T_fk_func = self.link_fk_funcs[self.tip_link]
        T = T_fk_func(q)

        R_ee = T[:3, :3]
        p_ee = T[:3, 3]

        if self.tip_link_object.visual is None:
            p_final = p_ee

        else:
            world_offset = R_ee @ self.ee_offset
            p_final = p_ee + world_offset

        return cas.vertcat(p_final[0], p_final[1], p_final[2])

    def precompute_forward_kinematics(self):
        link_fk_funcs = {}
        link_joint_counts = {}

        for link in self.links:
            fk_dict = self.parser.get_forward_kinematics(self.root_link, link)
            link_fk_funcs[link] = fk_dict["T_fk"]

            joint_info = self.parser.get_joint_info(self.root_link, link)
            link_joint_counts[link] = len(joint_info[1])

        return link_fk_funcs, link_joint_counts

    def precompute_ee_offset(self):
        if self.tip_link_object is None or self.tip_link_object.visual is None:
            return cas.DM([0, 0, 0])

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
            local_offset = np.array([0, 0, size[2] / 2.0])
            if origin is not None and origin.xyz is not None:
                local_offset += np.array(origin.xyz)

        elif hasattr(geometry, "radius"):  # Sphere
            radius = geometry.radius
            local_offset = np.array([0, 0, radius])
            if origin is not None and origin.xyz is not None:
                local_offset += np.array(origin.xyz)

        else:
            if origin is not None and origin.xyz is not None:
                local_offset = np.array(origin.xyz)

        local_offset = cas.DM(local_offset)

        return local_offset

    def get_links(self):
        return self.links

    def get_joint_map(self):
        return self.joint_map

    def get_n_links(self):
        return self.n_links

    def get_n_joints(self):
        return self.n_joints

    def get_robot_path(self):
        return self.robot_path


class MobileManipulatorRobotURDF(ManipulatorRobotURDF):
    def __init__(self, robot_path, root_link="world", tip_link="end_effector"):
        super().__init__(robot_path, root_link, tip_link)
