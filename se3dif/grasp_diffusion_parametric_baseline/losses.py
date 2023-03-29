import roma
import torch
from torch.nn.functional import mse_loss


def normalize_quat(x):
    return x / x.square().sum(dim=-1).sqrt().unsqueeze(-1)


class NoisePredictionLoss:
    def __init__(self, args):
        self.rotation_parametrization = args["NetworkSpecs"]["rotation_parametrization"]
        assert self.rotation_parametrization in ["euler_angles", "quaternion"]
        self.translation_sigma = args["NetworkSpecs"]["translation_sigma"]
        self.rotation_sigma = args["NetworkSpecs"]["rotation_sigma"]
        self.diffusion_steps = args["NetworkSpecs"]["diffusion_steps"]

    def loss_fn(self, model, model_input, ground_truth, val=False, eps=1e-5):
        object_pcd = model_input["visual_context"]
        grasps = model_input["x_ene_pos"]
        batch_size, num_grasps = grasps.shape[0], grasps.shape[1]
        grasps = grasps.view(-1, 4, 4)

        t = torch.rand((batch_size * num_grasps, 1), device=grasps.device) * (1. - eps) + eps

        translations = grasps[:, :3, 3]
        # TODO Implement formula properly - let's copy the loss from somewhere if we can
        t_translations = translations + self.translation_sigma * t * torch.randn_like(translations)
        eps_translations = (self.translation_sigma / self.diffusion_steps) * torch.randn_like(translations)
        noisy_translations = t_translations + eps_translations

        if self.rotation_parametrization == "euler_angles":
            euler_angles = roma.rotmat_to_rotvec(grasps[:, :3, :3])
            t_euler_angles = euler_angles + t * self.rotation_sigma * torch.randn_like(euler_angles)
            eps_euler_angles = (self.translation_sigma / self.diffusion_steps) * torch.randn_like(euler_angles)
            eps_gt = torch.cat([eps_translations, eps_euler_angles], dim=-1)
            noisy_euler_angles = t_euler_angles + eps_euler_angles
            noisy_rotations = roma.rotvec_to_rotmat(noisy_euler_angles)

        elif self.rotation_parametrization == "quaternion":
            quaternions = roma.rotmat_to_unitquat(grasps[:, :3, :3])
            t_quaternions = normalize_quat(quaternions + t * self.rotation_sigma * torch.randn_like(quaternions))
            eps_quaternions = (self.translation_sigma / self.diffusion_steps) * torch.randn_like(quaternions)
            eps_gt = torch.cat([eps_translations, eps_quaternions], dim=-1)
            noisy_quaternions = normalize_quat(t_quaternions + eps_quaternions)
            noisy_rotations = roma.unitquat_to_rotmat(noisy_quaternions)

        noisy_grasps = torch.zeros_like(grasps)
        noisy_grasps[:, :3, :3] = noisy_rotations
        noisy_grasps[:, :3, 3] = noisy_translations

        noisy_grasps = noisy_grasps.view(batch_size, num_grasps, 4, 4)
        t = t.view(batch_size, num_grasps, 1)

        eps_predicted = model(object_pcd, noisy_grasps, t)
        raise NotImplementedError

        info = {}
        loss_dict = {"loss/noise_prediction": mse_loss(eps_predicted, eps_gt)}
        return loss_dict, info
