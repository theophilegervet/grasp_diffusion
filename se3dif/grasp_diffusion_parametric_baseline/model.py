import torch.nn as nn
from torch import Tensor


class GraspDiffusionParametricBaseline(nn.Module):

    def __init__(self):
        super().__init__()
        pass

    def forward(self, object_pcd: Tensor, noisy_grasps: Tensor, t: Tensor):
        """
        Args:
            object_pcd: object point cloud of shape (batch, num_points, 3)
            noisy_grasps: noisy grasps of shape (batch, num_grasps, 4, 4)
            t: time steps of shape (batch, num_grasps, 1)

        Returns:
            noise: grasp noise of shape (batch, num_grasps, 4, 4)
        """
        # TODO
        #  1. PointNet
        #  2. cross-interactions between grasp points and object points
        #  3. how to take into account t?
        pass
