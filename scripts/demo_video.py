import os
import sys
import cv2
import numpy as np
import torch
import time
import tyro

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from data_config import SMPLX_PATH
from prompt_hmr.smpl_family import SMPLX as SMPLX_Layer
from prompt_hmr.utils.rotation_conversions import axis_angle_to_matrix
from prompt_hmr.vis.viser import viser_vis_human, viser_vis_world4d
from prompt_hmr.vis.traj import get_floor_mesh
from pipeline import Pipeline


def main(input_video='data/examples/boxing_short.mp4', 
         static_camera=False,
         run_viser=True,
         viser_total=1500, 
         viser_subsample=1):
    smplx = SMPLX_Layer(SMPLX_PATH).cuda()

    output_folder = 'results/' + os.path.basename(input_video).split('.')[0]
    # If results already exist, ask the user how to proceed instead of exiting silently.
    results_pkl = os.path.join(output_folder, "results.pkl")
    if os.path.exists(results_pkl):
        orig_output = output_folder
        while True:
            print(f"Output folder '{output_folder}' already contains results (results.pkl).")
            print("Choose action: [d]elete folder and continue, [a]ppend numeric suffix, [c]ancel (exit)")
            choice = input("Enter d/a/c: ").strip().lower()
            if choice in ('d', 'delete'):
                import shutil
                try:
                    shutil.rmtree(output_folder)
                    print(f"Deleted folder '{output_folder}'. Continuing.")
                    break
                except Exception as e:
                    print(f"Failed to delete '{output_folder}': {e}")
                    print("Please choose another action.")
                    continue
            elif choice in ('a', 'append'):
                # Find next available folder name by appending _1, _2, ...
                i = 1
                while True:
                    candidate = f"{orig_output}_{i}"
                    # if candidate folder doesn't exist, use it
                    if not os.path.exists(candidate):
                        output_folder = candidate
                        print(f"Using new output folder '{output_folder}'.")
                        break
                    # if candidate exists but has no results.pkl, it's safe to use
                    if not os.path.exists(os.path.join(candidate, 'results.pkl')):
                        output_folder = candidate
                        print(f"Using existing folder '{output_folder}' (no results.pkl present).")
                        break
                    i += 1
                break
            elif choice in ('c', 'cancel'):
                print('Cancelled by user. Exiting.')
                return
            else:
                print('Invalid choice, please enter d, a, or c.')
    
    pipeline = Pipeline(static_cam=static_camera)
    results = pipeline.__call__(input_video, 
                                output_folder, 
                                save_only_essential=True,
                                text_flag=True)
    # Viser
    if run_viser:
        # Downsample for viser visualization
        images = pipeline.images[:viser_total][::viser_subsample]
        world4d = pipeline.create_world4d(step=viser_subsample, total=viser_total)
        world4d = {i:world4d[k] for i,k in enumerate(world4d)}

        # Get vertices
        all_verts = []
        # We'll collect per-timestep arrays and then stack into ndarrays
        global_orient = []  # will become (T, 3, 3)
        body_pose = []      # will become (T, 21, 3, 3)
        betas = []          # will become (T, n_betas)
        transl = []         # will become (T, 3)
        for k in world4d:

            world3d = world4d[k]
            if len(world3d['track_id']) == 0: # no people
                continue
            rotmat = axis_angle_to_matrix(world3d['pose'].reshape(-1, 55, 3))

            # print(world3d)

            # 只保留第一个人的动作 — convert each entry to a numpy array
            # rotmat[0,0] -> first joint (global_orient) as (3,3)
            # rotmat[0,1:22] -> body pose (21,3,3)
            global_orient.append(rotmat[0,0].numpy())
            body_pose.append(rotmat[0,1:22].numpy())

            if len(betas) == 0:  # only need to save betas/transl once
                betas.append(world3d['shape'][0].numpy())

            transl.append(world3d['trans'][0].numpy())

            verts = smplx(global_orient = rotmat[:,:1].cuda(),
                        body_pose = rotmat[:,1:22].cuda(),
                        betas = world3d['shape'].cuda(),
                        transl = world3d['trans'].cuda()).vertices.cpu().numpy()
            
            world3d['vertices'] = verts
            all_verts.append(torch.tensor(verts, dtype=torch.bfloat16))

        # Stack lists into ndarrays with timestep as first dim
        if len(global_orient) == 0:
            print(f'No person frames found in {output_folder}, skipping save.')
        else:
            global_orient = np.stack(global_orient, axis=0)
            body_pose = np.stack(body_pose, axis=0)
            betas_array = betas[0]
            transl = np.stack(transl, axis=0)

            numpy_dict = {
                'root_orient': global_orient,
                'pose_body': body_pose,
                'betas':betas_array,
                'trans': transl,
                'num_betas': betas_array.shape[0]}

            np.savez(f'{output_folder}/smplx_traj.npz', **numpy_dict)

        all_verts = torch.cat(all_verts)
        [gv, gf, gc] = get_floor_mesh(all_verts, scale=2)

    
        server, gui = viser_vis_world4d(images, 
                                        world4d, 
                                        smplx.faces, 
                                        floor=[gv, gf],
                                        init_fps=30/viser_subsample)
        
        url = f'https://localhost:{server.get_port()}'
        print(f'Please use this url to view the results: {url}')
        print('For longer video, it will take a few seconds for the webpage to load.')

        gui_playing, gui_timestep, gui_framerate, num_frames = gui
        while True:
            # Update the timestep if we're playing.
            if gui_playing.value:
                gui_timestep.value = (gui_timestep.value + 1) % num_frames

            time.sleep(1.0 / gui_framerate.value)
        


if __name__ == '__main__':
    tyro.cli(main)