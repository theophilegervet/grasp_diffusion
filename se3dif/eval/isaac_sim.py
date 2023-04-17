from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
from se3dif.datasets.acronym_dataset import PointcloudAcronymAndSDFDataset, AcronymGrasps
import h5py
import torch
from theseus.geometry import SO3
from isaacgym import gymtorch
from se3dif.visualization import grasp_visualization
import matplotlib.pyplot as plt
import time

global_scale = 1
# global_offset = np.array([10.0, 0, 1])
global_offset = np.array([0.0, 0, 0])

test_grasp = torch.eye(4)
# manually defined eye(4)
test_grasp = torch.Tensor(
    [[1.0, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]
    ]
).numpy()
# test_grasp = torch.Tensor(
#     [[0.0, 0, 1, 0],
#      [0, 1, 0, 0],
#      [-1, 0, 0, 0],
#      [0, 0, 0, 1]
#     ]
# ).numpy()
glob_transform = np.array(
    [[1.0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
    ]
)

def test():
    from scipy.spatial.transform import Rotation

    # R = SO3(tensor=torch.inverse(torch.from_numpy(object_data['grasp'][:3, :3])).unsqueeze(0)).to_quaternion()[0]
    grasp = test_grasp
    grasp = get_sample_object_grasp()['grasp']
    print("\n\n\n================================")
    test_point = np.array([1,2,3,1])
    print("data", grasp, test_point)

    print("mult")
    print(grasp @ test_point)

    R = SO3(tensor=torch.from_numpy(grasp[:3, :3]).unsqueeze(0)).to_quaternion()[0]
    from scipy.spatial.transform import Rotation
    Rx2 = Rotation.from_matrix(grasp[:3, :3])
    R2 = Rx2.as_quat()
    print("rot repr", Rx2.as_quat(), Rx2.as_euler("zyx"))
    print("quat", R, R2)

    print("scipy apply", Rx2.apply(test_point[:3]))

    # gym_quat = gymapi.Quat.from_euler_zyx(Rx2.as_euler("zyx")[2], Rx2.as_euler("zyx")[1], Rx2.as_euler("zyx")[0])
    # gym_quat = gymapi.Quat.from_euler_zyx(Rx2.as_euler("zyx")[0], Rx2.as_euler("zyx")[1], Rx2.as_euler("zyx")[2]) # -> get correct gym angles back
    gym_quat = gymapi.Quat(*R2)
    print("gym rot", gym_quat.rotate((gymapi.Vec3(test_point[0], test_point[1], test_point[2]))))
    print("gym rot", gym_quat.rotate((gymapi.Vec3(test_point[2], test_point[1], test_point[0]))))
    import itertools
    for (a, b, c) in list(itertools.permutations(list(test_point[:3]))):
        print("gym rot", gym_quat.rotate((gymapi.Vec3(a, b, c))))
    print("gym angles", gym_quat.to_euler_zyx())

    T = grasp[:3, 3] * global_scale + global_offset
    print("translation", T)

    panda_pose = gymapi.Transform()
    panda_pose.p = gymapi.Vec3(T[0], T[1], T[2])
    # panda_pose.r = gymapi.Quat(R[0], R[1], R[2], R[3])
    # panda_pose.r = gymapi.Quat(R2[0], R2[1], R2[2], R2[3])
    # panda_pose.r = gymapi.Quat.from_euler_zyx(Rx2.as_euler("zyx")[0], Rx2.as_euler("zyx")[1], Rx2.as_euler("zyx")[2])
    panda_pose.r = gym_quat

    print("res point", panda_pose.transform_vector(gymapi.Vec3(1, 2, 3)))
    print("res point", panda_pose.transform_point(gymapi.Vec3(1, 2, 3)))


def plot_grasp(grasp_obj, grasp_ind):
    mesh = grasp_obj.load_mesh()
    pcl = mesh.sample(200)

    # com = mesh.center_mass
    # mesh.vertices -= com

    # grasp = test_grasp
    grasp = torch.from_numpy(grasp_obj.good_grasps[grasp_ind,...])
    # grasp[:3, 3] = 0
    # grasp = glob_transform @ grasp
    scene = grasp_visualization.visualize_grasps(grasp[None, ...], mesh=mesh, show=True, spheres_for_points=False)

    print(grasp)
    print(torch.Tensor([1., 2, 3, 1]).T @ grasp.float())
    print(grasp.float() @ torch.Tensor([1., 2, 3, 1]))

    # image = grasp_visualization.get_scene_grasps_image(grasp_obj.good_grasps[grasp_ind:grasp_ind+1], p_cloud=pcl)
    # figure = plt.figure()
    # plt.imshow(image)
    # plt.show()
    # scene = grasp_visualization.visualize_grasps(grasp_obj.good_grasps[grasp_ind:grasp_ind+1], p_cloud=pcl, show=True, spheres_for_points=False)

    # scene = grasp_visualization.visualize_grasps(grasp_obj.good_grasps[grasp_ind:grasp_ind+1], mesh=mesh, show=True, spheres_for_points=False)


def get_sample_object_grasp():
    grasp_ind = 7
    grasp_file = "/home/sirdome/katefgroup/tschindl/datasets/analogical_grasping/grasps/Cup/Cup_106f8b5d2f2c777535c291801eaf5463_6.809242501596408e-05.h5"

    grasp_obj = AcronymGrasps(grasp_file)
    plot_grasp(grasp_obj, grasp_ind)
    with h5py.File(grasp_file, 'r') as data:
        mesh_fname = data["object/file"][()].decode('utf-8')
        mesh_type = mesh_fname.split('/')[1]
        mesh_id = mesh_fname.split('/')[-1].split('.')[0]
        mesh_scale = data["object/scale"][()]

    #     grasps = np.array(data["grasps/transforms"])
    #     success = np.array(data["grasps/qualities/flex/object_in_gripper"])
    #     good_idxs = np.argwhere(success==1)[:,0]
    #     good_grasps = grasps[good_idxs,...]
    
    # print("res", mesh_fname, mesh_type, mesh_id, mesh_scale, good_grasps.shape)

    grasp = np.array(
        [[1, 0, 0, -0.1],
         [0, 1, 0, 0.25],
         [0, 0, 1, 0.19],
         [0, 0, 0, 1]],
         dtype=np.float32
    )
    grasp = grasp_obj.good_grasps[grasp_ind,...]

    # grasp[:3, 3] = 0

    transform = np.array(
        [[0.0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1]
        ]
    )

    # grasp = test_grasp
    # grasp = transform @ grasp
    grasp = glob_transform @ grasp

    # new_grasp = grasp.copy()
    # new_grasp[1, 3] = grasp[2, 3]
    # new_grasp[2, 3] = grasp[1, 3]
    # new_grasp[1, :] = grasp[2, :]
    # new_grasp[2, :] = grasp[1, :]
    # grasp = new_grasp.copy()
    # new_grasp[:, 1] = grasp[:, 2]
    # new_grasp[:, 2] = grasp[:, 1]

    # grasp = new_grasp

    res = {
        'object_root': '/home/sirdome/katefgroup/tschindl/grasp_diffusion/assets/',
        'object_file': 'test_mesh_2.urdf',
        'object_scale': mesh_scale,

        'grasp': grasp
        # 'grasp': good_grasps[3,...],
    }
    # print(res)

    return res

def setup_sim(gym):

    # get default set of parameters
    sim_params = gymapi.SimParams()

    # set common parameters
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.dt = 1 / 3000
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    # sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)

    # set PhysX-specific parameters
    # sim_params.physx.use_gpu = True
    # sim_params.physx.solver_type = 1
    # sim_params.physx.num_position_iterations = 6
    # sim_params.physx.num_velocity_iterations = 1
    # sim_params.physx.contact_offset = 0.01
    # sim_params.physx.rest_offset = 0.0

    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.use_gpu = True

    # set Flex-specific parameters
    # sim_params.flex.solver_type = 5
    # sim_params.flex.num_outer_iterations = 4
    # sim_params.flex.num_inner_iterations = 20
    # sim_params.flex.relaxation = 0.8
    # sim_params.flex.warm_start = 0.5

    # create sim with these parameters
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    return sim

def add_gravity(gym, sim):
    sim_params = gym.get_sim_params(sim)
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    gym.set_sim_params(sim, sim_params)

def load_ground_plane(gym, sim):
    # configure the ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
    plane_params.distance = 0
    plane_params.static_friction = 1
    plane_params.dynamic_friction = 1
    plane_params.restitution = 0

    # create the ground plane
    gym.add_ground(sim, plane_params)

def create_viewer(gym):
    cam_props = gymapi.CameraProperties()
    print("cam: ", cam_props, cam_props.far_plane, cam_props.near_plane)
    viewer = gym.create_viewer(sim, cam_props)
    return viewer

def load_panda(gym: gymapi.Gym, sim):
    asset_root = "/home/sirdome/katefgroup/tschindl/vcpd/assets"
    panda_asset_file = "panda.urdf"
    # asset_root = "/home/sirdome/katefgroup/tschindl/isaacgym/assets"
    # panda_asset_file = "urdf/franka_description/robots/franka_panda.urdf"

    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.01
    asset_options.fix_base_link = True
    asset_options.override_com = True
    asset_options.override_inertia = True
    asset_options.disable_gravity = True
    asset_options.flip_visual_attachments = True
    panda_asset = gym.load_asset(sim, asset_root, panda_asset_file, asset_options)

    obj_prop = gymapi.RigidShapeProperties()
    obj_prop.friction = 3.0
    obj_prop.restitution = 0.9
    gym.set_asset_rigid_shape_properties(panda_asset, [obj_prop])
    # configure panda dofs
    panda_dof_props = gym.get_asset_dof_properties(panda_asset)
    panda_lower_limits = panda_dof_props["lower"] * global_scale
    panda_upper_limits = panda_dof_props["upper"] * global_scale
    panda_ranges = panda_upper_limits - panda_lower_limits

    # grippers
    panda_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    panda_dof_props["stiffness"].fill(800.0)
    panda_dof_props["damping"].fill(40.0)

    # default dof states and position targets
    panda_num_dofs = gym.get_asset_dof_count(panda_asset)
    default_dof_pos = np.zeros(panda_num_dofs, dtype=np.float32)
    # grippers open
    default_dof_pos = panda_upper_limits
    print("dof pos", default_dof_pos)

    default_dof_state = np.zeros(panda_num_dofs, gymapi.DofState.dtype)
    default_dof_state["pos"] = default_dof_pos

    return default_dof_pos, default_dof_state, panda_asset, panda_dof_props

def load_obj(gym, sim, object_root, object_file):
    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.01
    asset_options.override_com = True
    asset_options.override_inertia = True
    asset_options.vhacd_enabled = True

    # asset_options.disable_gravity = True

    # current_asset = gym.load_asset(sim, "/home/sirdome/katefgroup/tschindl/grasp_diffusion/assets/", "test_mesh.urdf", asset_options)
    current_asset = gym.load_asset(sim, object_root, object_file, asset_options)
    obj_prop = gymapi.RigidShapeProperties()
    obj_prop.friction = 5.0
    # obj_prop.restitution = 0.9
    obj_prop.rolling_friction = 3.0
    gym.set_asset_rigid_shape_properties(current_asset, [obj_prop])

    return current_asset


def create_env(gym, sim):
    spacing = 20.0
    #lower = gymapi.Vec3(-spacing, 0.0, -spacing)
    lower = gymapi.Vec3(0.0, 0.0, 0.0)
    upper = gymapi.Vec3(spacing, spacing, spacing)

    env = gym.create_env(sim, lower, upper, 1)

    # Load object and grasp position
    object_data = get_sample_object_grasp()

    # Panda gripper

    default_dof_pos, default_dof_state, panda_asset, panda_dof_props = load_panda(gym, sim)

    panda_pose = gymapi.Transform()
    # pose.p = gymapi.Vec3(1.0, 1.0, 1.0)
    # pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    # adjust gripper as it is rotated around z-axis by 90 degrees
    adjust_rot_matrix = np.array(
        [[0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]]
    )

    from scipy.spatial.transform import Rotation

    grasp_matrix = object_data['grasp'] @ adjust_rot_matrix

    # R = SO3(tensor=torch.inverse(torch.from_numpy(object_data['grasp'][:3, :3])).unsqueeze(0)).to_quaternion()[0]
    print(grasp_matrix)
    # R = SO3(tensor=torch.from_numpy(object_data['grasp'][:3, :3]).unsqueeze(0)).to_quaternion()[0]
    Rx2 = Rotation.from_matrix(grasp_matrix[:3, :3])
    R2 = Rx2.as_quat()
    print("rot repr", Rx2.as_quat(), Rx2.as_euler("zyx"))
    # print("quat", R, R2)
    T = grasp_matrix[:3, 3] * global_scale + global_offset
    panda_pose.p = gymapi.Vec3(T[0], T[1], T[2])
    # panda_pose.r = gymapi.Quat(R[0], R[1], R[2], R[3])
    panda_pose.r = gymapi.Quat(R2[0], R2[1], R2[2], R2[3])
    # panda_pose.r = gymapi.Quat.from_euler_zyx(Rx2.as_euler("zyx")[0], Rx2.as_euler("zyx")[1], Rx2.as_euler("zyx")[2])

    print(panda_pose.r, panda_pose.p)
    print("res point", panda_pose.transform_point(gymapi.Vec3(1, 2, 3)))

    panda_handle = gym.create_actor(env, panda_asset, panda_pose, "Gripper", 0)
    # actor_handle = gym.create_actor(env, panda_asset, panda_dof_props, "Gripper")

    gym.set_actor_scale(env, panda_handle, global_scale)

    gym.set_actor_dof_properties(env, panda_handle, panda_dof_props)
    gym.set_actor_dof_states(env, panda_handle, default_dof_state, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, panda_handle, default_dof_pos)

    
    # Object to grasp
    obj_asset = load_obj(gym, sim, object_data['object_root'], object_data['object_file'])

    obj_pose = gymapi.Transform()
    # pose.p = gymapi.Vec3(1.0, 1.0, 5.0)
    # pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
    obj_pose.p = gymapi.Vec3(global_offset[0], global_offset[1], global_offset[2])
    obj_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, 0.0)
    # pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)
    obj_handle = gym.create_actor(env, obj_asset, obj_pose, "Object", 0)
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, obj_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
    
    print(object_data['object_scale'])
    gym.set_actor_scale(env, obj_handle, object_data['object_scale'] * global_scale)

    return env, panda_handle

def run_sim(gym, sim, env, panda_handle, viewer):
    step = 0
    while not gym.query_viewer_has_closed(viewer):
        # if step == 300:
        #     # gym.set_dof_position_target_tensor(env, panda_handle, np.zeros(2, gymapi.DofState.dtype))
        #     gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(torch.zeros(2)))
        # if step == 3000:
        #     add_gravity(gym, sim)
        step += 1
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

def cleanup(gym, sim, viewer):
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

# obj_grasp = get_sample_object_grasp()

# test()

gym = gymapi.acquire_gym()
sim = setup_sim(gym)
load_ground_plane(gym, sim)

env, panda_handle = create_env(gym, sim)

viewer = create_viewer(gym)
run_sim(gym, sim, env, panda_handle, viewer)
cleanup(gym, sim, viewer)