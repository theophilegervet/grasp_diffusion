from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from pyglet.canvas.xlib import NoSuchDisplayException
import wandb

from se3dif.samplers import Grasp_AnnealedLD
from se3dif.utils import to_numpy
from se3dif.visualization import grasp_visualization
import tempfile


def denoising_summary(model, model_input, ground_truth, info, writer, iter, prefix=""):
    observation = model_input['visual_context']
    batch = 4

    ## 1. visualize generated grasps ##
    model.eval()
    model.set_latent(observation[:1,...], batch=batch)
    generator = Grasp_AnnealedLD(model, batch=batch, T=30, T_fit=50, device=observation.device)
    H = generator.sample()

    H = to_numpy(H)
    H[:, :3, -1]*=1/8
    if observation.dim()==3:
        point_cloud = to_numpy(model_input['visual_context'])[0,...]/8.
    else:
        point_cloud = to_numpy(model_input['point_cloud'])[0,...]/8.

    def write_grasps(H, name):
        try:

            if writer == "wandb":
                scene = grasp_visualization.visualize_grasps(H, p_cloud=point_cloud, show=False, spheres_for_points=True)
                with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as tmp_file:
                    scene.export(tmp_file.name, file_type='glb')
                    tmp_file.flush()
                    wandb.log({prefix + name :wandb.Object3D(tmp_file.name)})
            elif writer == "tensorboard":
                image = grasp_visualization.get_scene_grasps_image(H, p_cloud=point_cloud)
                figure = plt.figure()
                plt.imshow(image)
                writer.add_figure(prefix + name, figure, global_step=iter)

        except NoSuchDisplayException as e:
            print(e)
            import traceback
            traceback.print_exc()
            print("No display found. Skipping grasp visualization.")
    
    write_grasps(H, "generated_grasps")
    gt_grasps = model_input["x_ene_pos"][0, :10, :, :].cpu().numpy().copy()
    gt_grasps[:, :3, -1]*=1/8
    write_grasps(gt_grasps, "ground_truth_grasps")
