from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from pyglet.canvas.xlib import NoSuchDisplayException
import wandb

from se3dif.samplers import Grasp_AnnealedLD
from se3dif.utils import to_numpy
from se3dif.visualization import grasp_visualization
import tempfile
from se3dif.eval import earth_mover_distance

import torch
import numpy as np

def denoising_summary(model, model_input, ground_truth, info, writer, iter, prefix=""):
    num_grasps = 15

    observation = model_input['visual_context']
    # number of grasps to generate
    batch = 200

    ## 1. visualize generated grasps ##
    model.eval()
    model.set_latent(observation[:1, ...], batch=batch)
    generator = Grasp_AnnealedLD(model, batch=batch, T=30, T_fit=50, device=observation.device)
    H, trj_H = generator.sample(save_path=True)
    # squeezed because of batch size 1
    H = H.unsqueeze(0)
    trj_H = trj_H.unsqueeze(0)

    print(H.shape, trj_H.shape)
    H_norm_all = to_numpy(H[0, ...]).copy()
    H_norm_all[:, :3, -1] *= 1/8
    H_norm = H_norm_all[:num_grasps, ...].copy()
    trj_H = to_numpy(trj_H[0, ...])
    trj_H[..., :3, -1] *= 1/8.
    print(H_norm.shape, trj_H.shape)

    if observation.dim()==3:
        point_cloud = to_numpy(model_input['visual_context'])[0,...]/8.
    else:
        point_cloud = to_numpy(model_input['point_cloud'])[0,...]/8.

    def write_grasps(H, name, grasp_colors=None):
        try:

            if writer == "wandb":
                scene = grasp_visualization.visualize_grasps(H, colors=grasp_colors, p_cloud=point_cloud, show=False, spheres_for_points=True)
                with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as tmp_file:
                    scene.export(tmp_file.name, file_type='glb')
                    tmp_file.flush()
                    wandb.log({prefix + name : wandb.Object3D(tmp_file.name)}, step=iter)
            elif writer == "tensorboard":
                image = grasp_visualization.get_scene_grasps_image(H, colors=grasp_colors, p_cloud=point_cloud)
                figure = plt.figure()
                plt.imshow(image)
                writer.add_figure(prefix + name, figure, global_step=iter)

        except NoSuchDisplayException as e:
            print(e)
            import traceback
            traceback.print_exc()
            print("No display found. Skipping grasp visualization.")
    
    write_grasps(H_norm, "generated_grasps")
    #gt_grasps = model_input["x_ene_pos"][0, :num_grasps, :, :].cpu().numpy().copy()
    gt_grasps = model_input["x_ene_pos"][0, ...].cpu().numpy().copy()
    gt_grasps[:, :3, -1]*=1/8
    write_grasps(gt_grasps[:num_grasps, ...], "ground_truth_grasps")

    # Visualize diffusion trajectory: Only one trajectory with num_grasps grasp poses
    T = np.linspace(0, trj_H.shape[0]-1, num_grasps).astype(int)
    trj_H = trj_H[T, 0, ...]
    # Colors from red to green as the trajectory progresses
    colors = np.linspace([255, 0, 0], [0, 255, 0], num=T.shape[0]).astype(np.int8)
    write_grasps(trj_H, "diffusion_trajectory", grasp_colors=colors)

    # log earth mover distances
    # emd = 0
    # for i in range(H.shape[0]):
    #     emd += earth_mover_distance.earth_mover_distance(H[i], model_input["x_ene_pos"][i])
    # emd /= H.shape[0]
    # torch tensor from numpy array H_norm_all
    emd = earth_mover_distance.earth_mover_distance(torch.from_numpy(H_norm_all), torch.from_numpy(gt_grasps))
    print("emd", emd)
    if writer == "wandb":
        wandb.log({prefix+"emd" : emd}, step=iter)
    else:
        writer.add_scalar(prefix + "emd", emd, iter)