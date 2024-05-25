#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_from_batch
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
import concurrent.futures
from torch.utils.data import DataLoader

def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)


def render_set(model_path, name, iteration, scene, gaussians, pipeline,audio_dir, batch_size):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    inf_audio_dir = audio_dir
    
    makedirs(render_path, exist_ok=True)
    if name != 'custom':
        makedirs(gts_path, exist_ok=True)
    
    viewpoint_stack = scene
    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=False,num_workers=32,collate_fn=list)
    
    loader = iter(viewpoint_stack_loader)
    
    if name == "train" :
        process_until = 1000
        print(" -------------------------------------------------")
        print("        train set rendering  :   {} frames   ".format(process_until))
        print(" -------------------------------------------------")
    else:
        process_until = len(viewpoint_stack.dataset) 
        print(" -------------------------------------------------")
        print("        test set rendering  :   {}  frames  ".format(process_until))
        print(" -------------------------------------------------") 
    print("point nums:",gaussians._xyz.shape[0])
    image = []
    gt = []
    audio_attention = []
    eye_attention = []
    null_attention = []
    cam_attention = []
    
    iterations = process_until // batch_size
    if process_until % batch_size != 0:
        iterations += 1
    total_time = 0
    #render image
    for idx in tqdm(range(iterations), desc="Rendering progress",total = iterations):
        try:
            viewpoint_cams = next(loader)
            output = render_from_batch(viewpoint_cams, gaussians, pipeline, 
                                    random_color= False, stage='fine',
                                    batch_size=batch_size, visualize_attention=False, only_infer=True)
        except:
            break
        total_time += output["inference_time"]
        image.append(output["rendered_image_tensor"].cpu())
        gt.append(output["gt_tensor"].cpu())
        
    image_tensor = torch.cat(image,dim=0)[:process_until]
    gt_image_tensor = torch.cat(gt,dim=0)[:process_until]
    
    print("total frame:",(image_tensor.shape[0]))
    print("FPS:",(torch.cat(image,dim=0).shape[0])/(total_time))
    
    
    #render attention
    loader = iter(viewpoint_stack_loader)
    for idx in range(iterations):
        try:
            viewpoint_cams = next(loader)
            output = render_from_batch(viewpoint_cams, gaussians, pipeline, 
                                    random_color= False, stage='fine',
                                    batch_size=batch_size, visualize_attention=True, only_infer=True) 
        except:
            break    
        total_time += output["inference_time"]
        audio_attention.append(output["audio_attention"].cpu())
        eye_attention.append(output["eye_attention"].cpu())
        cam_attention.append(output["cam_attention"].cpu())
        null_attention.append(output["null_attention"].cpu())
        
        
    audio_tensor = torch.cat(audio_attention,0)[:process_until]
    eye_tensor = torch.cat(eye_attention,0)[:process_until]
    cam_tensor = torch.cat(cam_attention,0)[:process_until]
    null_tensor = torch.cat(null_attention,0)[:process_until]
    
    if name != 'custom':
        write_frames_to_video(tensor_to_image(gt_image_tensor),gts_path+f'/gt', use_imageio = True)
    write_frames_to_video(tensor_to_image(image_tensor),render_path+'/renders', use_imageio = True)
    write_frames_to_video(tensor_to_image(audio_tensor),render_path+'/audio', use_imageio = False)
    write_frames_to_video(tensor_to_image(eye_tensor),render_path+'/eye', use_imageio = False)
    write_frames_to_video(tensor_to_image(null_tensor),render_path+'/null', use_imageio = False)
    write_frames_to_video(tensor_to_image(cam_tensor),render_path+'/cam', use_imageio = False)

    if name != 'custom':
        cmd = f'ffmpeg -loglevel quiet -y -i {render_path}/renders.mp4 -i {inf_audio_dir} -c:v copy -c:a aac {render_path}/{model_path.split("/")[-2]}_{name}_{iteration}iter_renders.mov'
        os.system(cmd)
    cmd = f'ffmpeg -loglevel quiet -y -i {gts_path}/gt.mp4 -i {inf_audio_dir} -c:v copy -c:a aac {gts_path}/{model_path.split("/")[-2]}_{name}_{iteration}iter_gt.mov'
    os.system(cmd)
    cmd = f'ffmpeg -loglevel quiet -y -i {render_path}/audio.mp4 -i {inf_audio_dir} -c:v copy -c:a aac {render_path}/{model_path.split("/")[-2]}_{name}_{iteration}iter_audio.mov'
    os.system(cmd)
    cmd = f'ffmpeg -loglevel quiet -y -i {render_path}/eye.mp4 -i {inf_audio_dir} -c:v copy -c:a aac {render_path}/{model_path.split("/")[-2]}_{name}_{iteration}iter_eye.mov'
    os.system(cmd)
    cmd = f'ffmpeg -loglevel quiet -y -i {render_path}/null.mp4 -i {inf_audio_dir} -c:v copy -c:a aac {render_path}/{model_path.split("/")[-2]}_{name}_{iteration}iter_null.mov'
    os.system(cmd)
    cmd = f'ffmpeg -loglevel quiet -y -i {render_path}/cam.mp4 -i {inf_audio_dir} -c:v copy -c:a aac {render_path}/{model_path.split("/")[-2]}_{name}_{iteration}iter_cam.mov'
    os.system(cmd)
    
    if name != 'custom':
        os.remove(f"{gts_path}/gt.mp4")
    os.remove(f"{render_path}/renders.mp4")

    os.remove(f"{render_path}/audio.mp4")
    os.remove(f"{render_path}/eye.mp4")
    os.remove(f"{render_path}/null.mp4")
    os.remove(f"{render_path}/cam.mp4")
    
def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, args):
    skip_train, skip_test, skip_video, batch_size= args.skip_train, args.skip_test, args.skip_video, args.batch
    
    with torch.no_grad():
        data_dir = dataset.source_path
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, custom_aud=args.custom_aud)
        
        gaussians.eval()
        
        if args.custom_aud != '':
            audio_dir = os.path.join(data_dir, args.custom_wav)
            render_set(dataset.model_path, "custom", scene.loaded_iter, scene.getCustomCameras(), gaussians, pipeline, audio_dir, batch_size)
        
        if not skip_train:
            audio_dir = os.path.join(data_dir, "aud_train.wav")
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, audio_dir, batch_size)

        if not skip_test:
            audio_dir = os.path.join(data_dir, "aud_novel.wav")
            render_set(dataset.model_path, "test",iteration, scene.getTestCameras(), gaussians, pipeline, audio_dir, batch_size)

def write_frames_to_video(frames, path, codec='mp4v', fps=25, use_imageio=False):
    if use_imageio:
        imageio.mimwrite(f'{path}.mp4', frames, fps=fps, quality=8, output_params=['-vf', f'fps={fps}'], macro_block_size=None)
    else:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video = cv2.VideoWriter(f'{path}.mp4', fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))

        for frame in frames:
            video.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        video.release()

def tensor_to_image(tensor, normalize=True):
    if torch.is_tensor(tensor):
        image = tensor.detach().cpu().numpy().squeeze()
    else:
        image = tensor
        
    if normalize:
        image = 255 * image
        image = image.clip(0, 255).astype(np.uint8)

    if len(image.shape) == 3:
        image = image.transpose(1, 2, 0)
    elif len(image.shape) == 4:
        image = image.transpose(0, 2, 3, 1)
    return image        

            
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--custom_aud", type=str, default='')
    parser.add_argument("--custom_wav", type=str, default='')
    # parser.add_argument("--audio_dir", type=str)
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    args.only_infer = True
    print(args)
    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args)