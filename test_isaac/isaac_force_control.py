# place the .urdf and .obj files in the same directory as this script and run it from this directory
# change the get_data to run the simulation on the battery

from isaacgym import gymapi
import numpy as np
import h5py
import torch
from isaacgym import gymtorch
import time
from scipy.spatial.transform import Rotation

global_offset = np.array([10.0, 0, 1])

def get_data():
    battery = {
        'object_root': './',
        'object_file': 'battery.urdf',
        'object_scale': 0.1015580546,
        'grasp': np.array([[-0.12490466,  0.84040001,  0.52737714, -0.05648107],
                    [-0.0936886 ,  0.51917433, -0.84951778,  0.091113  ],
                    [-0.98773543, -0.15551796,  0.0138886 ,  0.09488805],
                    [ 0.        ,  0.        ,  0.        ,  1.        ]])
    }
    cup = {
        'object_root': './',
        'object_file': 'cup.urdf',
        'object_scale': 6.80924e-05,
        'grasp': np.array(
                [[-0.35832797, -0.93208311, -0.05312394, -0.04456832],
                [-0.93354378,  0.35832797,  0.00985247,  0.26671346],
                [ 0.00985247,  0.05312394, -0.99853932,  0.19418912],
                [ 0.        ,  0.        ,  0.        ,  1.        ]])
    }

    # return battery
    return cup

def setup_sim(gym):
    sim_params = gymapi.SimParams()

    sim_params.dt = 1 / 300
    sim_params.substeps = 10
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 80
    sim_params.physx.num_velocity_iterations = 10
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.use_gpu = True
    sim_params.physx.contact_collection = gymapi.CC_ALL_SUBSTEPS
    sim_params.physx.contact_offset = 0.001

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    return sim

def load_ground_plane(gym, sim):
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
    plane_params.distance = 0
    plane_params.static_friction = 0
    plane_params.dynamic_friction = 0
    plane_params.restitution = 0

    gym.add_ground(sim, plane_params)

def create_viewer(gym):
    cam_props = gymapi.CameraProperties()
    viewer = gym.create_viewer(sim, cam_props)
    return viewer

def load_panda(gym: gymapi.Gym, sim):
    asset_root = "."
    panda_asset_file = "panda.urdf"

    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.01
    asset_options.fix_base_link = True
    asset_options.override_com = True
    asset_options.override_inertia = True
    asset_options.disable_gravity = True
    asset_options.flip_visual_attachments = True
    # asset_options.vhacd_enabled = True
    panda_asset = gym.load_asset(sim, asset_root, panda_asset_file, asset_options)

    obj_prop = gymapi.RigidShapeProperties()
    obj_prop.friction = 3.0
    obj_prop.restitution = 0.9
    obj_prop.thickness = 0.01
    gym.set_asset_rigid_shape_properties(panda_asset, [obj_prop])

    # configure panda dofs
    panda_dof_props = gym.get_asset_dof_properties(panda_asset)
    panda_lower_limits = panda_dof_props["lower"]
    panda_upper_limits = panda_dof_props["upper"]
    panda_ranges = panda_upper_limits - panda_lower_limits

    # grippers
    # panda_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    panda_dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
    # panda_dof_props["stiffness"].fill(800.0)
    panda_dof_props["stiffness"].fill(0.0)
    # panda_dof_props["damping"].fill(40.0)
    panda_dof_props["damping"].fill(1.0)
    panda_dof_props["friction"].fill(0.0)

    print("props:", panda_dof_props)

    # default dof states and position targets
    panda_num_dofs = gym.get_asset_dof_count(panda_asset)
    default_dof_pos = np.zeros(panda_num_dofs, dtype=np.float32)

    # grippers open
    default_dof_pos = panda_upper_limits
    default_dof_state = np.zeros(panda_num_dofs, gymapi.DofState.dtype)
    print('default pos', default_dof_state)
    default_dof_state["pos"] = default_dof_pos
    print('default pos', default_dof_state)

    return default_dof_pos, default_dof_state, panda_asset, panda_dof_props

def load_obj(gym, sim, object_root, object_file):
    asset_options = gymapi.AssetOptions()
    asset_options.disable_gravity = True

    asset_options.armature = 0.01
    asset_options.override_com = True
    asset_options.override_inertia = True
    asset_options.vhacd_enabled = True

    current_asset = gym.load_asset(sim, object_root, object_file, asset_options)
    obj_prop = gymapi.RigidShapeProperties()
    obj_prop.friction = 5.0
    obj_prop.restitution = 0.9
    obj_prop.rolling_friction = 3.0
    obj_prop.thickness = 0.01
    gym.set_asset_rigid_shape_properties(current_asset, [obj_prop])

    return current_asset

def create_env(gym, sim):
    spacing = 20.0
    lower = gymapi.Vec3(0.0, 0.0, 0.0)
    upper = gymapi.Vec3(spacing, spacing, spacing)
    env = gym.create_env(sim, lower, upper, 1)

    # Panda gripper
    default_dof_pos, default_dof_state, panda_asset, panda_dof_props = load_panda(gym, sim)

    # Load object and grasp position
    # object_data = get_sample_object_grasp()
    object_data = get_data()
    print(object_data)
    grasp_matrix = object_data['grasp']

    panda_pose = gymapi.Transform()
    R = Rotation.from_matrix(grasp_matrix[:3, :3]).as_quat()
    T = grasp_matrix[:3, 3] + global_offset
    panda_pose.p = gymapi.Vec3(T[0], T[1], T[2])
    panda_pose.r = gymapi.Quat(R[0], R[1], R[2], R[3])

    panda_handle = gym.create_actor(env, panda_asset, panda_pose, "Gripper", group=1, filter=0)

    gym.set_actor_scale(env, panda_handle, 1)

    gym.set_actor_dof_properties(env, panda_handle, panda_dof_props)
    print("setting probs", panda_dof_props)
    # gym.set_actor_dof_states(env, panda_handle, default_dof_state, gymapi.STATE_ALL)
    gym.set_actor_dof_states(env, panda_handle, default_dof_state, gymapi.STATE_POS)

    # gym.set_actor_dof_position_targets(env, panda_handle, default_dof_pos)
    # effort = np.ones(2, dtype=np.float32) * 3
    # print("effort", gym.apply_actor_dof_efforts(env, panda_handle, effort))

    print("cur states:", gym.get_actor_dof_states(env, panda_handle, gymapi.STATE_VEL))
    print("cur states:", gym.get_actor_dof_states(env, panda_handle, gymapi.STATE_POS))
    print("cur states:", gym.get_actor_dof_states(env, panda_handle, gymapi.STATE_ALL))

 
    # Object to grasp
    obj_asset = load_obj(gym, sim, object_data['object_root'], object_data['object_file'])
    obj_pose = gymapi.Transform()
    obj_pose.p = gymapi.Vec3(global_offset[0], global_offset[1], global_offset[2])
    obj_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, 0.0)

    obj_handle = gym.create_actor(env, obj_asset, obj_pose, "Object", group=1, filter=0)
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, obj_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
    
    gym.set_actor_scale(env, obj_handle, object_data['object_scale'])

    return env, panda_handle

def run_sim(gym, sim, env, panda_handle, viewer):
    _root_tensor = gym.acquire_actor_root_state_tensor(sim)
    root_tensor = gymtorch.wrap_tensor(_root_tensor)

    start_time = time.time()
    next_step = 0
    while not gym.query_viewer_has_closed(viewer):
        print("cur states:", gym.get_actor_dof_states(env, panda_handle, gymapi.STATE_VEL))
        print("cur states:", gym.get_actor_dof_states(env, panda_handle, gymapi.STATE_POS))
        print("cur states:", gym.get_actor_dof_states(env, panda_handle, gymapi.STATE_ALL))
        print("cur states:", gym.get_actor_dof_velocity_targets(env, panda_handle))
        print("cur states:", gym.get_actor_dof_position_targets(env, panda_handle))
        print("cur states:", gym.get_actor_dof_forces(env, panda_handle))
        print()
        if next_step == 0 and time.time()-start_time > 3:
            next_step += 1
            print("========== start closing ==========")
            # set position target as closed
            # gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(torch.zeros(2)))
            effort = np.ones(2, dtype=np.float32) * 70
            print("effort", gym.apply_actor_dof_efforts(env, panda_handle, -effort))
        if next_step == 1 and time.time()-start_time > 10:
            # don't increment step to keep moving up

            print("========== move up ==========")
            rigid_obj = torch.clone(root_tensor)
            rigid_obj[0, 2] += 0.0005
            actor_indices = torch.tensor([panda_handle], dtype=torch.int32, device=root_tensor.device)
            gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(rigid_obj),
                                                    gymtorch.unwrap_tensor(actor_indices),
                                                    1)
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_dof_force_tensor(sim)
        gym.refresh_mass_matrix_tensors(sim)
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_net_contact_force_tensor(sim)

        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

def cleanup(gym, sim, viewer):
    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

gym = gymapi.acquire_gym()
sim = setup_sim(gym)
load_ground_plane(gym, sim)

env, panda_handle = create_env(gym, sim)

viewer = create_viewer(gym)
run_sim(gym, sim, env, panda_handle, viewer)
cleanup(gym, sim, viewer)
