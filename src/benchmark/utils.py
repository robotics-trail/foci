import numpy as np

def minimum_robot_environment_distance(robot, environment, trajectory):
    min_dist = np.inf
    argmin = None

    for t, q in enumerate(trajectory):
        robot_points = np.asarray(robot.collision_points(q)).reshape(-1, 3)

        for i, pr in enumerate(robot_points):
            for j, po in enumerate(environment.obstacle_means):
                d = np.linalg.norm(pr - po)

                if d < min_dist:
                    min_dist = d
                    argmin = {
                        "sample": t,
                        "robot_gaussian": i,
                        "obstacle": j,
                        "robot_point": pr,
                        "obstacle_point": po,
                    }

    return min_dist, argmin