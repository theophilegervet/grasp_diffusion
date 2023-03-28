from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from pyglet.canvas.xlib import NoSuchDisplayException
import wandb

from se3dif.samplers import Grasp_AnnealedLD
from se3dif.utils import to_numpy
from se3dif.visualization import grasp_visualization
import tempfile

import numpy as np

def denoising_summary(model, model_input, ground_truth, info, writer, iter, prefix=""):
    num_grasps = 15

    observation = model_input['visual_context']
    batch = num_grasps

    ## 1. visualize generated grasps ##
    model.eval()
    model.set_latent(observation[:1,...], batch=batch)
    generator = Grasp_AnnealedLD(model, batch=batch, T=30, T_fit=50, device=observation.device)
    H, trj_H = generator.sample(save_path=True)

    H = to_numpy(H)
    H[:, :3, -1]*=1/8
    trj_H = to_numpy(trj_H)
    trj_H[..., :3, -1] *=1/8.

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
                    wandb.log({prefix + name :wandb.Object3D(tmp_file.name)})
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
    
    write_grasps(H, "generated_grasps")
    gt_grasps = model_input["x_ene_pos"][0, :num_grasps, :, :].cpu().numpy().copy()
    gt_grasps[:, :3, -1]*=1/8
    write_grasps(gt_grasps, "ground_truth_grasps")

    # Visualize diffusion trajectory: Only one trajectory with num_grasps grasp poses
    T = np.linspace(0, trj_H.shape[0]-1, num_grasps).astype(int)
    trj_H = trj_H[T, 0, ...]
    # Colors from red to green as the trajectory progresses
    colors = np.linspace([255, 0, 0], [0, 255, 0], num=T.shape[0]).astype(np.int8)
    write_grasps(trj_H, "diffusion_trajectory", grasp_colors=colors)
