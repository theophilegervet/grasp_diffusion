import numpy as np
import torch
import trimesh
import trimesh.transformations as tra
import open3d
import io
from PIL import Image
from pyglet import gl


def create_gripper_marker_trimesh(color=[0, 0, 255], tube_radius=0.001, sections=6, scale = 1.):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.

    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    cfl = trimesh.creation.cylinder(
        radius=0.002*scale,
        sections=sections,
        segment=[
            [4.10000000e-02*scale, -7.27595772e-12*scale, 6.59999996e-02*scale],
            [4.10000000e-02*scale, -7.27595772e-12*scale, 1.12169998e-01*scale],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=0.002*scale,
        sections=sections,
        segment=[
            [-4.100000e-02*scale, -7.27595772e-12*scale, 6.59999996e-02*scale],
            [-4.100000e-02*scale, -7.27595772e-12*scale, 1.12169998e-01*scale],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=0.002*scale, sections=sections, segment=[[0, 0, 0], [0, 0, 6.59999996e-02*scale]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=0.002*scale,
        sections=sections,
        segment=[[-4.100000e-02*scale, 0, 6.59999996e-02*scale], [4.100000e-02*scale, 0, 6.59999996e-02*scale]],
    )

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color

    return tmp


def get_gripper_control_points_open3d(grasp, show_sweep_volume=False, color=(0.2, 0.8, 0.)):
    """Open3D Visualization of parallel-jaw grasp.

    From https://github.com/adithyamurali/TaskGrasp/blob/master/visualize.py

    Arguments:
        grasp: [4, 4] np array
    """

    meshes = []
    align = tra.euler_matrix(np.pi / 2, -np.pi / 2, 0)

    # Cylinder 3,5,6
    cylinder_1 = open3d.geometry.TriangleMesh.create_cylinder(
        radius=0.005, height=0.139)
    transform = np.eye(4)
    transform[0, 3] = -0.03
    transform = np.matmul(align, transform)
    transform = np.matmul(grasp, transform)
    cylinder_1.paint_uniform_color(color)
    cylinder_1.transform(transform)

    # Cylinder 1 and 2
    cylinder_2 = open3d.geometry.TriangleMesh.create_cylinder(
        radius=0.005, height=0.07)
    transform = tra.euler_matrix(0, np.pi / 2, 0)
    transform[0, 3] = -0.065
    transform = np.matmul(align, transform)
    transform = np.matmul(grasp, transform)
    cylinder_2.paint_uniform_color(color)
    cylinder_2.transform(transform)

    # Cylinder 5,4
    cylinder_3 = open3d.geometry.TriangleMesh.create_cylinder(
        radius=0.005, height=0.06)
    transform = tra.euler_matrix(0, np.pi / 2, 0)
    transform[2, 3] = 0.065
    transform = np.matmul(align, transform)
    transform = np.matmul(grasp, transform)
    cylinder_3.paint_uniform_color(color)
    cylinder_3.transform(transform)

    # Cylinder 6, 7
    cylinder_4 = open3d.geometry.TriangleMesh.create_cylinder(
        radius=0.005, height=0.06)
    transform = tra.euler_matrix(0, np.pi / 2, 0)
    transform[2, 3] = -0.065
    transform = np.matmul(align, transform)
    transform = np.matmul(grasp, transform)
    cylinder_4.paint_uniform_color(color)
    cylinder_4.transform(transform)

    cylinder_1.compute_vertex_normals()
    cylinder_2.compute_vertex_normals()
    cylinder_3.compute_vertex_normals()
    cylinder_4.compute_vertex_normals()

    meshes.append(cylinder_1)
    meshes.append(cylinder_2)
    meshes.append(cylinder_3)
    meshes.append(cylinder_4)

    # Just for visualizing - sweep volume
    if show_sweep_volume:
        finger_sweep_volume = open3d.geometry.TriangleMesh.create_box(
            width=0.06, height=0.02, depth=0.14)
        transform = np.eye(4)
        transform[0, 3] = -0.06 / 2
        transform[1, 3] = -0.02 / 2
        transform[2, 3] = -0.14 / 2

        transform = np.matmul(align, transform)
        transform = np.matmul(grasp, transform)
        finger_sweep_volume.paint_uniform_color(color)
        finger_sweep_volume.transform(transform)
        finger_sweep_volume.compute_vertex_normals()

        meshes.append(finger_sweep_volume)

    return meshes


def visualize_points_trimesh(model, input):
    ## Compute the SDF values for arbitrary 3D points ##
    c_thrs = 0.1

    model.eval()
    with torch.no_grad():
        sdf = model.sdf_net(input)

    x_sdf = input['x_sdf'][0,...].cpu().numpy()
    sdf = sdf['sdf'][0,...].cpu().numpy()
    sdf_points = trimesh.points.PointCloud(x_sdf)
    colors = np.zeros((x_sdf.shape[0],3))
    colors[:,0] = 255.*(c_thrs - np.clip(sdf, 0, c_thrs))/c_thrs
    sdf_points.colors = colors
    trimesh.Scene([sdf_points]).show()

    ## VISUALIZE GRASP POINTS ##
    with torch.no_grad():
        x, sdf = model.get_points_and_features(input)

    ## Visualization points ##
    point_clouds = input['point_cloud'].cpu().numpy()
    x = x[0,0,...].cpu().numpy()
    sdf = sdf[0,0,...].cpu().numpy()

    scale = input['scale'].cpu().numpy()
    H = input['x_ene'].cpu().numpy()

    ## Trimesh
    ## pointcloud model
    p_cloud_tri = trimesh.points.PointCloud(point_clouds[0,...])

    ## grasp points
    c1_thrs = 0.2
    c2_trhs = 0.4
    c3_trhs = 0.6
    pc2 = trimesh.points.PointCloud(x)
    colors = np.zeros((x.shape[0],3))
    colors[:, 0] = 255. * (c1_thrs - np.clip(sdf, 0, c1_thrs)) / c1_thrs
    delta_c12 = c2_trhs - c1_thrs
    colors[:, 1] = 255. * (delta_c12 - np.clip(sdf, c1_thrs, c2_trhs)) / delta_c12
    delta_c23 = c3_trhs - c2_trhs
    colors[:, 2] = 255. * (delta_c23 - np.clip(sdf, c2_trhs, c3_trhs)) / delta_c23

    pc2.colors = colors

    print('max occ: ', sdf[...].max())
    print('min occ: ', sdf[...].min())

    grip = create_gripper_marker_trimesh(scale=scale[0,0]).apply_transform(H[0,0,...])
    trimesh.Scene([p_cloud_tri, pc2, grip]).show()


def visualize_grasps_trimesh(Hs, scale=1., p_cloud=None, energies=None, colors=None, mesh=None, show=True):
    ## Set color list
    if colors is None:
        if energies is None:
            color = np.zeros(Hs.shape[0])
        else:
            min_energy = energies.min()
            energies -=min_energy
            color = energies/(np.max(energies)+1e-6)

    ## Grips
    grips = []
    for k in range(Hs.shape[0]):
        H = Hs[k,...]

        if colors is None:
            c = color[k]
            c_vis = [0, 0, int(c*254)]
        else:
            c_vis = list(colors[k,...])

        grips.append(
            create_gripper_marker_trimesh(color=c_vis, scale=scale).apply_transform(H)
        )

    ## Visualize grips and the object
    if mesh is not None:
        scene = trimesh.Scene([mesh]+ grips)
    elif p_cloud is not None:
        p_cloud_tri = trimesh.points.PointCloud(p_cloud)
        scene = trimesh.Scene([p_cloud_tri]+ grips)
    else:
        scene = trimesh.Scene(grips)

    if show:
        scene.show()
    else:
        return scene


def generate_mesh_grid(xmin=[-1.,-1.,-1.], xmax = [1., 1.,1.], n_points=20):
    x = torch.linspace(xmin[0], xmax[0], n_points)
    y = torch.linspace(xmin[1], xmax[1], n_points)
    z = torch.linspace(xmin[2], xmax[2], n_points)

    xx, yy, zz  = torch.meshgrid((x,y,z))

    xyz = torch.cat((xx.reshape(-1,1), yy.reshape(-1,1), zz.reshape(-1,1),),1)
    return xyz


def get_scene_grasps_image(Hs, scale=1., p_cloud=None, energies=None, colors=None, mesh=None, library="trimesh"):
    ## Set color list
    if colors is None:
        if energies is None:
            color = np.zeros(Hs.shape[0])
        else:
            min_energy = energies.min()
            energies -=min_energy
            color = energies/(np.max(energies)+1e-6)

    ## Grips
    grips = []
    for k in range(Hs.shape[0]):
        H = Hs[k,...]

        if colors is None:
            c = color[k]
            c_vis = np.random.rand(3)*255
        else:
            c_vis = list(colors[k,...])

        if library == "trimesh":
            grips.append(
                create_gripper_marker_trimesh(color=c_vis, scale=scale).apply_transform(H))
        elif library == "open3d":
            grips.extend(
                get_gripper_control_points_open3d(H, color=c_vis / 255.0))

    ## Visualize grips and the object
    if mesh is not None:
        scene = trimesh.Scene([mesh]+ grips)

    elif p_cloud is not None:
        c = np.ones((p_cloud.shape[0],4))
        c[:,0] = np.zeros(p_cloud.shape[0])

        if library == "trimesh":
            p_cloud_tri = trimesh.points.PointCloud(p_cloud, colors=c)
            scene = trimesh.Scene([p_cloud_tri] + grips)

        elif library == "open3d":
            opcd = open3d.geometry.PointCloud()
            opcd.points = open3d.utility.Vector3dVector(p_cloud)
            opcd.colors = open3d.utility.Vector3dVector(c[:, :3])
            geometries = [open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)]
            geometries.append(opcd)
            geometries += grips

    else:
        scene = trimesh.Scene(grips)

    if library == "trimesh":
        window_conf = gl.Config(double_buffer=True, depth_size=24)
        data = scene.save_image(resolution=[1080, 1080], window_conf=window_conf, visible=False)
        image = np.array(Image.open(io.BytesIO(data)))
    elif library == "open3d":
        vis = open3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1080, height=1080)
        for geom in geometries:
            vis.add_geometry(geom)
            vis.update_geometry(geom)
        vis.poll_events()
        vis.update_renderer()
        image = np.asarray(vis.capture_screen_float_buffer())
        vis.clear_geometries()
        vis.destroy_window()

    return image
