import torch
from scipy.optimize import linear_sum_assignment
from theseus.geometry import SO3
import numpy as np

def get_distance_matrix(grasps1, grasps2):
    assert(len(grasps1) == len(grasps2))

    def get_translations(grasps):
        return grasps[:, :3, 3]
    def get_rotations(grasps):
        return grasps[:, :3, :3]

    t1 = get_translations(grasps1)
    t2 = get_translations(grasps2)
    R1 = get_rotations(grasps1)
    R2 = get_rotations(grasps2)

    translation_distance = torch.linalg.norm(t1.unsqueeze(1) - t2.unsqueeze(0), dim=-1)

    #inv_rotations = torch.linalg.inv(R1)
    inv_rotations = R1.transpose(-2, -1)

    relative_rotation = torch.matmul(inv_rotations.unsqueeze(1), R2.unsqueeze(0))
    so3_all = SO3(tensor = relative_rotation.reshape(-1, 3, 3))
    log_map = so3_all.log_map().reshape(R1.shape[0], R2.shape[0], 3)
    rot_dist = torch.linalg.norm(log_map, dim=-1)

    return translation_distance + rot_dist

def earth_mover_distance(grasps1, grasps2, distance_matrix=None):
    if distance_matrix is None:
        distance_matrix = get_distance_matrix(grasps1, grasps2).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    print(np.min(distance_matrix, axis=-1).mean(), np.min(distance_matrix.T, axis=-1).mean())
    return distance_matrix[row_ind, col_ind].mean()
