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
import numpy as np
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from time import time 
    
    
def render_from_batch(viewpoint_cameras, pc : GaussianModel, pipe, random_color= False, scaling_modifier = 1.0, stage="fine", batch_size=1, visualize_attention=False, only_infer = False, canonical_tri_plane_factor_list = None, iteration=None):
    if only_infer:
        time1 = time()
        batch_size = len(viewpoint_cameras)
    means3D = pc.get_xyz.unsqueeze(0).repeat(batch_size, 1, 1) # [B, N, 3]
    opacity = pc._opacity.unsqueeze(0).repeat(batch_size, 1, 1) # [B, N, 1]
    shs = pc.get_features.unsqueeze(0).repeat(batch_size, 1, 1, 1) # [B, N, 16, 3]
    scales = pc._scaling.unsqueeze(0).repeat(batch_size, 1, 1) # [B, N, 3]
    rotations = pc._rotation.unsqueeze(0).repeat(batch_size, 1, 1) # [B, N, 4] 
    attention = None
    colors_precomp = None
    cov3D_precomp = None
     
    aud_features = []
    eye_features = []
    rasterizers = []
    gt_imgs = []
    viewspace_point_tensor_list = []
    means2Ds = []
    lips_list = []
    bg_w_torso_list = []
    gt_masks = []
    gt_w_bg = []
    cam_features = []
    
    for viewpoint_camera in viewpoint_cameras:
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        means2Ds.append(screenspace_points)
        viewspace_point_tensor_list.append(screenspace_points)
        
        if random_color:
            background = torch.rand((3,), dtype=torch.float32, device="cuda")
            bg_image = background[:, None, None] * torch.ones((1, viewpoint_camera.image_height, viewpoint_camera.image_width), device=background.device)
        elif only_infer:
            bg_image = viewpoint_camera.bg_w_torso.to('cuda')
        else: 
            white_or_black = torch.randint(2, (1,)).item()
            background = torch.full((3,), white_or_black, dtype=torch.float32, device="cuda")
            bg_image = background[:, None, None] * torch.ones((1, viewpoint_camera.image_height, viewpoint_camera.image_width), device=background.device)
        
        aud_features.append(viewpoint_camera.aud_f.unsqueeze(0).to(means3D.device))
        eye_features.append(torch.from_numpy(np.array([viewpoint_camera.eye_f])).unsqueeze(0).to(means3D.device))
        cam_features.append(torch.from_numpy(np.concatenate((viewpoint_camera.R.reshape(-1), viewpoint_camera.T.reshape(-1))).reshape(1,-1)).to(means3D.device))
        bg_w_torso_list.append(viewpoint_camera.bg_w_torso.cpu())
        
        bg_image = viewpoint_camera.bg_w_torso.to('cuda')
        
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
            
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx= tanfovx,
            tanfovy= tanfovy,
            bg=bg_image,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        rasterizers.append(GaussianRasterizer(raster_settings=raster_settings))

        bg_mask = viewpoint_camera.head_mask 
        bg_mask = torch.from_numpy(bg_mask).to("cuda")
        gt_image = viewpoint_camera.original_image.cuda()
        gt_w_bg.append(gt_image.unsqueeze(0))
        gt_image = gt_image * bg_mask + bg_image * (~ bg_mask)
        gt_imgs.append(gt_image.unsqueeze(0).cuda())
        lips_list.append(viewpoint_camera.lips_rect)
        bg_mask = bg_mask.to(torch.float).unsqueeze(0).unsqueeze(0)
        gt_masks.append(bg_mask)
    
    if stage == "coarse":
        aud_features, eye_features, cam_features = None, None, None 
        means3D_final, scales_temp, rotations_temp, opacity_temp, shs_temp = pc._deformation(means3D, scales, rotations, opacity, shs, aud_features, eye_features, cam_features)
        if "scales" in canonical_tri_plane_factor_list:
            scales_temp = scales_temp-2
            scales_final = scales_temp
        else: 
            scales_final = scales
            scales_temp = None
        if "rotations" in canonical_tri_plane_factor_list:
            rotations_final = rotations_temp
        else: 
            rotations_final = rotations
            rotations_temp = None
        if "opacity" in canonical_tri_plane_factor_list:
            opacity_final = opacity_temp
        else:
            opacity_final = opacity
            opacity_temp = None
        if "shs" in canonical_tri_plane_factor_list:
            shs_final = shs_temp
        else:
            shs_final = shs
            shs_temp = None
            
        pc.replace_gaussian(scales_temp, rotations_temp, opacity_temp, shs_temp)

        scales_final = pc.scaling_activation(scales_final)
        rotations_final = torch.nn.functional.normalize(rotations_final,dim=2) 
        opacity_final = pc.opacity_activation(opacity_final)

    elif stage == "fine":
        aud_features = torch.cat(aud_features,dim=0)
        eye_features = torch.cat(eye_features,dim=0)
        cam_features = torch.cat(cam_features,dim=0)
        means3D_final, scales_final, rotations_final, opacity_final, shs_final, attention = pc._deformation(means3D, scales, rotations, opacity, shs, aud_features, eye_features,cam_features)
                                                                                                    
        scales_final = pc.scaling_activation(scales_final)
        rotations_final = torch.nn.functional.normalize(rotations_final,dim=2) 
        opacity_final = pc.opacity_activation(opacity_final)
        
    
    rendered_image_list = []
    radii_list = []
    depth_list = []
    visibility_filter_list = []
    audio_image_list = []
    eye_image_list = []
    cam_image_list = []
    null_image_list = []
    rendered_lips = []
    gt_lips = []
    
    for idx, rasterizer in enumerate(rasterizers):
        colors_precomp = None
        rendered_image, radii, depth = rasterizer(
            means3D = means3D_final[idx],
            means2D = means2Ds[idx],
            shs = shs_final[idx],
            colors_precomp = colors_precomp,
            opacities = opacity_final[idx],
            scales = scales_final[idx],
            rotations = rotations_final[idx],
            cov3D_precomp = cov3D_precomp,)

        rendered_image_list.append(rendered_image.unsqueeze(0))
        radii_list.append(radii.unsqueeze(0))
        depth_list.append(depth.unsqueeze(0))
        visibility_filter_list.append((radii > 0).unsqueeze(0))
        
        if not only_infer:
            y1,y2,x1,x2 = lips_list[idx]
            lip_crop = rendered_image[:,y1:y2,x1:x2]
            gt_lip_crop = gt_imgs[idx][:,:,y1:y2,x1:x2]
            rendered_lips.append(lip_crop.flatten())
            gt_lips.append(gt_lip_crop.flatten())
            
        audio_image, eye_image ,cam_image, null_image = None, None, None, None
        if visualize_attention:
            colors_precomp = attention.mean(dim=1)[idx,:,0].unsqueeze(1).repeat(1, 3)
            min_val = colors_precomp.min()
            max_val = colors_precomp.max()
            colors_precomp = (colors_precomp - min_val) / (max_val - min_val)

            audio_image, radii, depth = rasterizer(
                means3D = means3D_final[idx],
                means2D = means2Ds[idx],
                shs = None,
                colors_precomp = colors_precomp,
                opacities = opacity_final[idx], 
                scales = scales_final[idx],
                rotations = rotations_final[idx],
                cov3D_precomp = cov3D_precomp,)
                    
            colors_precomp = attention.mean(dim=1)[idx,:,1].unsqueeze(1).repeat(1, 3)
            min_val = colors_precomp.min()
            max_val = colors_precomp.max()
            colors_precomp = (colors_precomp - min_val) / (max_val - min_val)
            
            eye_image, radii, depth = rasterizer(
                means3D = means3D_final[idx],
                means2D = means2Ds[idx],
                shs = None,
                colors_precomp = colors_precomp,
                opacities = opacity_final[idx], 
                scales = scales_final[idx],
                rotations = rotations_final[idx],
                cov3D_precomp = cov3D_precomp,)
            
            colors_precomp = attention.mean(dim=1)[idx,:,2].unsqueeze(1).repeat(1, 3)
            min_val = colors_precomp.min()
            max_val = colors_precomp.max()
            colors_precomp = (colors_precomp - min_val) / (max_val - min_val)
            
            cam_image, radii, depth = rasterizer(
                means3D = means3D_final[idx],
                means2D = means2Ds[idx],
                shs = None,
                colors_precomp = colors_precomp,
                opacities = opacity_final[idx], 
                scales = scales_final[idx],
                rotations = rotations_final[idx],
                cov3D_precomp = cov3D_precomp,)
            
            colors_precomp = attention.mean(dim=1)[idx,:,3].unsqueeze(1).repeat(1, 3)
            min_val = colors_precomp.min()
            max_val = colors_precomp.max()
            colors_precomp = (colors_precomp - min_val) / (max_val - min_val)
            
            null_image, radii, depth = rasterizer(
                means3D = means3D_final[idx],
                means2D = means2Ds[idx],
                shs = None,
                colors_precomp = colors_precomp,
                opacities = opacity_final[idx], 
                scales = scales_final[idx],
                rotations = rotations_final[idx],
                cov3D_precomp = cov3D_precomp,)

            audio_image_list.append(audio_image.unsqueeze(dim=0))
            eye_image_list.append(eye_image.unsqueeze(dim=0))
            cam_image_list.append(cam_image.unsqueeze(dim=0))
            null_image_list.append(null_image.unsqueeze(dim=0))
        
    radii = torch.cat(radii_list,0).max(dim=0).values
    visibility_filter_tensor = torch.cat(visibility_filter_list).any(dim=0)
    rendered_image_tensor = torch.cat(rendered_image_list,0)
    gt_tensor = torch.cat(gt_imgs,0)
    depth_tensor = torch.cat(depth_list,dim=0)
    gt_masks_tensor = torch.cat(gt_masks,dim=0)
    gt_w_bg_tensor = torch.cat(gt_w_bg,dim=0)
    
    audio_image_tensor, eye_image_tensor, null_image_tensor, cam_image_tensor = None, None, None, None
    if visualize_attention:
        audio_image_tensor = torch.cat(audio_image_list,0)
        eye_image_tensor = torch.cat(eye_image_list,0)
        cam_image_tensor = torch.cat(cam_image_list, 0)
        null_image_tensor = torch.cat(null_image_list,0)
    
    rendered_lips_tensor ,gt_lips_tensor, rendered_w_bg_tensor = None, None, None
    inference_time = None
    
    if not only_infer:
        rendered_lips_tensor = torch.cat(rendered_lips,0)
        gt_lips_tensor = torch.cat(gt_lips,0)
    if only_infer:
        inference_time = time()-time1
        
        
    return {"rendered_image_tensor": rendered_image_tensor,
        "gt_tensor":gt_tensor,
        "viewspace_points": screenspace_points,
        "visibility_filter_tensor" : visibility_filter_tensor,
        "viewspace_point_tensor_list" : viewspace_point_tensor_list,
        "radii": radii,
        "depth_tensor": depth_tensor,
        "audio_attention": audio_image_tensor,
        "eye_attention": eye_image_tensor,
        "cam_attention" : cam_image_tensor,
        "null_attention": null_image_tensor,
        "rendered_lips_tensor":rendered_lips_tensor,
        "gt_lips_tensor":gt_lips_tensor,
        "rendered_w_bg_tensor":rendered_w_bg_tensor,
        "inference_time":inference_time,
        "gt_masks_tensor":gt_masks_tensor,
        "gt_w_bg_tensor":gt_w_bg_tensor,
        }
