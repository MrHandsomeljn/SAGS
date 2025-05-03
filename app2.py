import os
import sys
import cv2
import torch

import numpy as np
import gradio as gr
import torch.nn.functional as F
import torchvision.transforms.functional as func

from tqdm import tqdm
from plyfile import PlyData, PlyElement
from argparse import ArgumentParser, Namespace

from gaussiansplatting.scene import Scene
from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene.gaussian_model import GaussianModel
from gaussiansplatting.arguments import ModelParams, PipelineParams

from seg_utils import grounding_dino_prompt
from seg_utils import conv2d_matrix, compute_ratios, update
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor, sam_model_registry)
from gradio_litmodel3d import LitModel3D

import datetime

# 获取当前时间字符串的lambda函数
current_time = lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log():
    with open(".log", "a+") as f:
        f.write(f"{[current_time()]}")
        f.write(torch.cuda.memory_summary())
        f.write("\n\n##########\n\n")

pipeline, background = None, None
empty_3D = torch.zeros((0,3),device='cuda')


# region 加载Gaussian

def get_combined_args(parser : ArgumentParser, model_path):
    # cmdlne_string = sys.argv[1:]
    # cfgfile_string = "Namespace()"
    cmdlne_string = ['--model_path', model_path]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)



# endregion




# region 加载Sam & feature

def load_sam(
        SAM_ARCH = 'vit_h',
        SAM_CKPT_PATH = './gaussiansplatting/dependencies/sam_ckpt/sam_vit_h_4b8939.pth'
    ):
    model_type = SAM_ARCH
    sam = sam_model_registry[model_type](checkpoint=SAM_CKPT_PATH).to('cuda')
    print(f"Loading SAM: {SAM_ARCH} from {SAM_CKPT_PATH}")
    predictor = SamPredictor(sam)
    return predictor

# extract and use sam features
def extract_sam_features(predictor, cameras, gaussians):
    sam_features = {}
    render_images = {}
    print("Prepocessing: extracting SAM features...")
    for view in tqdm(cameras):
        image_name = view.image_name
        render_pkg = render(view, gaussians, pipeline, background)

        render_image = render_pkg["render"].permute(1, 2, 0).detach().cpu().numpy()
        render_image = (255 * np.clip(render_image, 0, 1)).astype(np.uint8)

        predictor.set_image(render_image)
        sam_features[image_name] = predictor.features
    return sam_features

def extract_sam_features_pt(scene_path, predictor, cameras, gaussians):
    # 检查路径是否存在
    if scene_path and os.path.exists(scene_path):
        sam_path = os.path.join(scene_path,"sam_pt")
        try:
            print(f"Loading sam feature from {sam_path} ...")
            sam_features = torch.load(sam_path)
            return sam_features
        except Exception as e:
            print(f" Failed: {e}")
    
    print("Extract sam feature ...")
    sam_features = extract_sam_features(predictor, cameras, gaussians)
    
    if scene_path:
        sam_path = os.path.join(scene_path,"sam_pt")
        os.makedirs(os.path.dirname(sam_path), exist_ok=True)
        print(f"保存SAM特征到{sam_path}")
        torch.save(sam_features, sam_path)
    
    return sam_features



# endregion




# region 3D投影操作

## Project 3D points to 2D plane
def project_to_2d_batch(viewpoint_cameras, points3D) -> torch.Tensor: # C,N,2
    full_matrices = torch.stack([camera.full_proj_transform for camera in viewpoint_cameras])  # [C, 4, 4]
    if points3D.shape[-1] != 4: points3D = F.pad(input=points3D, pad=(0, 1), mode='constant', value=1)  # [N, 4]
    p_hom = torch.bmm(points3D.unsqueeze(0).expand(len(viewpoint_cameras), -1, -1), full_matrices)  # [C, N, 4]
    p_w = 1.0 / (p_hom[..., 3:] + 0.0000001)  # [C, N, 1]
    p_proj = p_hom[..., :3] * p_w  # [C, N, 3]
    heights = torch.tensor([camera.image_height for camera in viewpoint_cameras], device=points3D.device)  # [C]
    widths  = torch.tensor([camera.image_width  for camera in viewpoint_cameras], device=points3D.device)  # [C]
    sizes = torch.stack([widths, heights], dim=1).unsqueeze(1)  # [C, 1, 2]
    point_image = 0.5 * ((p_proj[..., :2] + 1) * sizes - 1)  # [C, N, 2]
    point_image = torch.round(point_image)
    return point_image  # 返回[C, N, 2]张量，C为相机数量，N为点数量

def project_to_2d(viewpoint_camera, points3D): # N,2
    full_matrix = viewpoint_camera.full_proj_transform  # torch.cuda, w2c @ K 
    if points3D.shape[-1] != 4:
        points3D = F.pad(input=points3D, pad=(0, 1), mode='constant', value=1)
    p_hom = (points3D @ full_matrix).transpose(0, 1)  # N, 4 -> 4, N   -1 ~ 1
    p_w = 1.0 / (p_hom[-1, :] + 0.0000001)
    p_proj = p_hom[:3, :] * p_w

    h = viewpoint_camera.image_height
    w = viewpoint_camera.image_width

    point_image = 0.5 * ((p_proj[:2] + 1) * torch.tensor([w, h]).unsqueeze(-1).to(p_proj.device) - 1) # image plane
    point_image = point_image.detach().clone()
    point_image = torch.round(point_image.transpose(0, 1))

    return point_image

# 给定单视角2Dprompt，创建3Dprompt
def get_3d_prompts(prompts_2d, point_image, xyz, depth=None):
    r = 4
    x_range = torch.arange(prompts_2d[0] - r, prompts_2d[0] + r)
    y_range = torch.arange(prompts_2d[1] - r, prompts_2d[1] + r)
    x_grid, y_grid = torch.meshgrid(x_range, y_range)
    neighbors = torch.stack([x_grid, y_grid], dim=2).reshape(-1, 2).to("cuda")
    prompts_index = [torch.where((point_image == p).all(dim=1))[0] for p in neighbors]
    indexs = []
    for index in prompts_index:
        if index.nelement() > 0:
            indexs.append(index)
    indexs = torch.unique(torch.cat(indexs, dim=0))
    indexs_depth = depth[indexs]
    valid_depth = indexs_depth[indexs_depth > 0]
    _, sorted_indices = torch.sort(valid_depth)
    valid_indexs = indexs[depth[indexs] > 0][sorted_indices[0]]
    
    return xyz[valid_indexs][:3].unsqueeze(0)

## Given 1st view point prompts, find corresponding 3D Gaussian point prompts
# 将3D点投影到2D，使得2D标注能映射到特定3D点上。
def generate_3d_prompts(xyz, viewpoint_camera, prompts_2d) -> torch.Tensor:
    w2c_matrix = viewpoint_camera.world_view_transform
    full_matrix = viewpoint_camera.full_proj_transform
    # project to image plane
    xyz = F.pad(input=xyz, pad=(0, 1), mode='constant', value=1)
    p_hom = (xyz @ full_matrix).transpose(0, 1)  # N, 4 -> 4, N
    p_w = 1.0 / (p_hom[-1, :] + 0.0000001)
    p_proj = p_hom[:3, :] * p_w
    # project to camera space
    p_view = (xyz @ w2c_matrix[:, :3]).transpose(0, 1)  # N, 3 -> 3, N
    depth = p_view[-1, :]
    valid_depth = depth >= 0

    h = viewpoint_camera.image_height
    w = viewpoint_camera.image_width

    point_image = 0.5 * ((p_proj[:2] + 1) * torch.tensor([w, h]).unsqueeze(-1).to(p_proj.device) - 1)
    point_image = torch.round(point_image.transpose(0, 1)).long()

    prompts_2d = torch.tensor(prompts_2d).to("cuda")
    prompts_3d = torch.stack([
        get_3d_prompts(prompts_2d[i], point_image, xyz, depth) 
        for i in range(prompts_2d.shape[0])
    ])
    prompts_3D = prompts_3d.reshape(-1, prompts_3d.shape[-1])
    return prompts_3D

def mask_inverse(xyz, viewpoint_camera, sam_mask):
    w2c_matrix = viewpoint_camera.world_view_transform
    # project to camera space
    xyz = F.pad(input=xyz, pad=(0, 1), mode='constant', value=1)
    p_view = (xyz @ w2c_matrix[:, :3]).transpose(0, 1)  # N, 3 -> 3, N
    depth = p_view[-1, :].detach().clone()
    valid_depth = depth >= 0

    h = viewpoint_camera.image_height # int
    w = viewpoint_camera.image_width  # int
    

    if sam_mask.shape[0] != h or sam_mask.shape[1] != w: # false
        sam_mask = func.resize(sam_mask.unsqueeze(0), (h, w), antialias=True).squeeze(0).long()
    else:
        sam_mask = sam_mask.long()

    point_image = project_to_2d(viewpoint_camera, xyz) # torch([N,2])
    point_image = point_image.long()

    valid_x = (point_image[:, 0] >= 0) & (point_image[:, 0] < w)
    valid_y = (point_image[:, 1] >= 0) & (point_image[:, 1] < h)
    valid_mask = valid_x & valid_y & valid_depth # [N]=bool
    point_mask = torch.full((point_image.shape[0],), -1, device="cuda") # [N]=-1
    point_mask[valid_mask] = sam_mask[point_image[valid_mask, 1], point_image[valid_mask, 0]] # [N]=sam_mask[x,y]
    indices_mask = torch.where(point_mask == 1)[0]

    return point_mask, indices_mask



# endregion




# region 分割

## Gaussian Decomposition
# 高斯边缘切割，原位操作，不增加点云数量
# 做完之后，处于物体边缘的点会被修改
# 可以作为分割后的后处理进行
def gaussian_decomp(gaussians, viewpoint_camera, input_mask, indices_mask):
    xyz = gaussians.get_xyz
    point_image = project_to_2d(viewpoint_camera, xyz)

    conv2d = conv2d_matrix(gaussians, viewpoint_camera, indices_mask, device="cuda")
    height = viewpoint_camera.image_height
    width = viewpoint_camera.image_width
    index_in_all, ratios, dir_vector = compute_ratios(conv2d, point_image, indices_mask, input_mask, height, width)

    decomp_gaussians = update(gaussians, viewpoint_camera, index_in_all, ratios, dir_vector)

    return decomp_gaussians

def save_gs(pc, indices_mask, save_path):
    xyz = pc._xyz.detach().cpu()[indices_mask].numpy()
    normals = np.zeros_like(xyz)
    f_dc = pc._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu()[indices_mask].numpy()
    f_rest = pc._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu()[indices_mask].numpy()
    opacities = pc._opacity.detach().cpu()[indices_mask].numpy()
    scale = pc._scaling.detach().cpu()[indices_mask].numpy()
    rotation = pc._rotation.detach().cpu()[indices_mask].numpy()
    dtype_full = [(attribute, 'f4') for attribute in pc.construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(save_path)

# endregion






class GradioAnnotationTool:
    def __init__(self, model_path, predictor):
        self.predictor = predictor
        if model_path is not None: self.load_gaussian_scene(model_path)
        else:
            self.scene         = None 
            self.args          = None 
            self.dataset       = None 
            self.images        = {} 
            self.sam_features  = {}
            self.current_image =  None

        self.has_ckpt = False
        self.saved_curr_gs_path = None
        self.saved_last_gs_path = None
        self.saved_last_ckpt_gs_path = None
        self.saved_curr_ckpt_gs_path = None
        self.result_gaussian = None

        self.maskid = 0
        self.segmode = 0 # Single, Multi
        self.clear_all_mark()

    def load_gaussian_scene(self, model_path, progress:gr.Progress=None):
        global pipeline
        global background
        
        if self.scene is not None: del self.scene
        if self.seg2d_mark is not None: del self.seg2d_mark

        print("Loading Gaussian Scene")
        if progress is not None: progress(0, desc="Get Args")

        parser = ArgumentParser(description="Testing script parameters")
        model = ModelParams(parser, sentinel=True)
        pipeline = PipelineParams(parser) # 渲染用

        parser.add_argument("--iteration", default=-1, type=int)
        parser.add_argument("--skip_train", action="store_true")
        parser.add_argument("--skip_test", action="store_true")
        parser.add_argument("--quiet", action="store_true")
        parser.add_argument("--threshold", default=0.7, type=float, help='threshold of label voting')
        parser.add_argument("--gd_interval", default=20, type=int, help='interval of performing gaussian decomposition')
        self.args = get_combined_args(parser, model_path)

        self.dataset = model.extract(self.args)
        self.dataset.model_path = self.args.model_path

        if progress is not None: progress(0.1, desc="Load Gaussian Scenes")
        gaussians = GaussianModel(self.dataset.sh_degree)
        self.scene = Scene(self.dataset, gaussians, load_iteration=self.args.iteration, shuffle=False)
        self.dataset.white_background = True
        bg_color = [1,1,1] if self.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if progress is not None: progress(0.4, desc="Rendering Images")
        self.images = {}
        for view in self.scene.getTrainCameras():
            render_pkg = render(view, self.scene.gaussians, pipeline, background)
            render_image = render_pkg["render"].permute(1, 2, 0).detach().cpu().numpy()
            render_image = (255 * np.clip(render_image, 0, 1)).astype(np.uint8)
            self.images[view.image_name] = render_image

        if progress is not None: progress(0.7, desc="Get Sam Features")
        self.sam_features = extract_sam_features_pt(model_path, self.predictor, self.scene.getTrainCameras(), self.scene.gaussians)
        self.current_image = list(self.images.keys())[0]
        self.predictor.set_image(self.images[self.current_image])

        self.result_gaussian  = self.load_scene_gaussians()

        # self.scene.gaussian:           场景的gaussian，每次分割都从此开始，不从中间的分割结果再次分割。
        # self.last_gaussian:            合并后的分割结果，是从curr_gaussian一个个累加的
        # self.saved_curr_gs_path:       保存的当前分割结果的路径 (用于展示)
        # self.saved_last_gs_path:       保存的合并分割结果的路径 (用于展示)
        # self.saved_curr_ckpt_gs_path:  暂存checkpoint路径       (用于恢复)
        # self.saved_last_ckpt_gs_path:  暂存checkpoint路径       (用于恢复)
        # self.用户用object名字保存的结果，直接存硬盘，展示路径用self.saved_last_gs_path

        self.last_gaussian = None
        self.saved_curr_gs_path = os.path.join(model_path, f'objects/tmp/tmp_curr.ply')
        self.saved_last_gs_path = os.path.join(model_path, f'objects/tmp/tmp_last.ply')
        self.saved_curr_ckpt_gs_path = os.path.join(model_path, f'objects/tmp/tmp_curr_ckpt.ply')
        self.saved_last_ckpt_gs_path = os.path.join(model_path, f'objects/tmp/tmp_last_ckpt.ply')
        os.makedirs(os.path.join(model_path, f'objects/tmp'), exist_ok=True)
        os.system(f"rm -r {self.saved_curr_gs_path      }")
        os.system(f"rm -r {self.saved_curr_ckpt_gs_path }")
        os.system(f"rm -r {self.saved_last_gs_path      }")
        os.system(f"rm -r {self.saved_last_ckpt_gs_path }")

        self.clear_all_mark()
        if progress is not None: progress(1, desc="Finished")
        print("加载完成")

    def load_gaussians(self, path):
        gaussians = GaussianModel(self.dataset.sh_degree)
        gaussians.load_ply(path)
        return gaussians

    def load_curr_gaussians(self):
        if os.path.isfile(self.saved_curr_gs_path):
            gaussians = GaussianModel(self.dataset.sh_degree)
            gaussians.load_ply(self.saved_curr_gs_path)
            return gaussians
        else:
            print(f"Curr Gaussian Not Found at: {self.saved_curr_gs_path}")
            return None

    def load_scene_gaussians(self):
        gaussians = GaussianModel(self.dataset.sh_degree)
        gaussians.load_ply(
            os.path.join(self.scene.model_path,
            "point_cloud",
            "iteration_" + str(self.scene.loaded_iter),
            "point_cloud.ply"))
        return gaussians

    def get_current_state(self):
        mark = self.seg2d_mark[self.current_image]
        return self.current_image, mark['points']
    
    def add_point(self, evt: gr.SelectData, is_original=True):
        """添加点，支持从原图或mask图添加"""
        print(f"点击事件: {evt}")
        if evt is None:
            print("收到空的点击事件")
            return self._render_original(), self._render_mask(), self._render_selected_mask()
        mark = self.seg2d_mark[self.current_image]
        x, y = evt.index
        print(f"On Add Point: curr_image={self.current_image}, hist_idx={mark['history_index']}, hist_len={len(mark['history'])}, nPoint={len(mark['points'])}, point=({x},{y})")
        mark['points'].append([x, y])
        mark["history_index"] += 1
        
        # 如果场景存在，更新3D点标注
        if self.scene is not None:
            self._update_3d_prompts()
            
        self._get_mask()
        self._save_state()
        self._new_loaded = False
        return self._render_original(), self._render_mask(), self._render_selected_mask()
    
    def undo(self):
        """撤销当前图片的最后一次操作"""
        mark = self.seg2d_mark[self.current_image]
        print(f"On Undo: {self.current_image}, {mark['history_index']}, {len(mark['history'])}, {len(mark['points'])}")
        
        # 检查是否有历史记录可撤销
        if mark['history_index'] > 0: # 此时若刚做了一次，为0
            mark['history_index'] -= 1
            prev_state = mark['history'][mark['history_index']]
            mark['points'] = prev_state[0][:]
            # 恢复3D点标注
            if len(prev_state) > 2 and prev_state[2] is not None:
                mark['prompts_3D'] = prev_state[2].clone() if prev_state[2] is not None else empty_3D.clone()
        else:
            mark["points"] = []
            mark['prompts_3D'] = empty_3D.clone()
            self._save_state()

        self._get_mask()

        # 如果场景存在，更新3D点标注
        if self.scene is not None and len(mark['points']): self._update_3d_prompts()
        return self._render_original(), self._render_mask(), self._render_selected_mask()
            # {i:len(self.seg2d_mark[i]["points"]) for i in self.seg2d_mark.keys()}

    def redo(self):
        """重做当前图片的操作"""
        mark = self.seg2d_mark[self.current_image]
        print(f"On Redo: {self.current_image}, {mark['history_index']}, {len(mark['history'])}, {len(mark['points'])}")
        
        # 检查是否有历史记录可重做
        if mark['history_index'] < len(mark['history']) - 1:
            mark['history_index'] += 1
            next_state = mark['history'][mark['history_index']]
            mark['points'] = next_state[0][:]
            # 恢复3D点标注
            if len(next_state) > 2 and next_state[2] is not None:
                mark['prompts_3D'] = next_state[2].clone() if next_state[2] is not None else empty_3D.clone()
            self._get_mask()
                
        # 如果场景存在，更新3D点标注
        if self.scene is not None and len(mark['points']): self._update_3d_prompts()
        return self._render_original(), self._render_mask(), self._render_selected_mask()

    def set_mask_id(self, mask_id):
        """设置当前选择的mask索引"""
        self.maskid = mask_id
        return self._render_selected_mask()
    
    def set_mode_id(self, mask_id):
        """设置当前选择的mask索引"""
        self.segmode = mask_id
        self._get_mask()
        return self._render_original(), self._render_mask(), self._render_selected_mask()
    
    def clear_current(self):
        """清空当前图片的所有标注"""
        mark = self.seg2d_mark[self.current_image]
        mark['points'] = []
        mark['prompts_3D'] = empty_3D.clone()  # 清空3D点标注
        self._save_state()
        self._get_mask()
        return self._render_original(), self._render_mask(), self._render_selected_mask()

    def clear_all_mark(self):
        self.seg2d_mark = {
            name: {
                'points': [],                     # 当前视角的点标注
                'history': [],                    # 历史记录 [(points, masks), ...]
                'history_index': -1,              # 历史记录索引，用于撤销/重做
                'prompts_3D': empty_3D.clone()    # 存储3D点标注
            } for name in self.images.keys()
        }
        self.record = { # 每次点击，根据不同的点击组合，会创建一个record结构。
            "prompts": None,              # torch.Tensor([N, 3]), 保存用于生成该记录的 3D 点
            "mvmask": {
                # maskid: {               # 0, 1, 2
                #     "multiview": {...}, # multiview_masks     (用于3D分割)
                #     "sam_masks": {...}, # multiview_sam_masks (用于3D分割)
                # }
            },
            "sam_all": {}                 # {img1_name: mask1, ...}（用于可视化）
        }
        if self.has_ckpt is False:
            self.ckpt_seg2d_mark = { name: {'points': [],'history': [],'history_index': -1,'prompts_3D': empty_3D.clone()} for name in self.images.keys()}
            self.ckpt_record = {"prompts": None, "mvmask": {},"sam_all": {}}
        self._new_loaded = True

    def clear_all(self):
        self.clear_all_mark()
        return self._render_original(), self._render_mask(), self._render_selected_mask()

    def _update_3d_prompts(self):
        mark = self.seg2d_mark[self.current_image]
        if not mark['points'] or self.scene is None:
            mark['prompts_3D'] = empty_3D.clone()
            return
            
        current_camera = None
        for camera in self.scene.getTrainCameras():
            if camera.image_name == self.current_image:
                current_camera = camera
                break
                
        if current_camera is None:
            print(f"找不到当前视角对应相机: {self.current_image}")
            return
        xyz = self.scene.gaussians.get_xyz
        prompts_3D = generate_3d_prompts(xyz, current_camera, mark['points'])
        mark['prompts_3D'] = prompts_3D
        
    def update_image(self, image_name):
        self.current_image = image_name
        return self._render_original(), self._render_mask(), self._render_selected_mask()
    
    def toggle_mask(self):
        if self.masks[self.current_image] is not None:
            self.masks[self.current_image] = None
        elif self.points[self.current_image]:
            self._get_mask()
        return self._render_display()

    def generate_multiview_masks(self, prompts_3D, text_prompt = None, progress:gr.Progress=None):
        """
        sam_mask_all_levels (cuda) : list[torch.int64(id,H,W)] , 每个视角下的渲染图的sam mask
        sam_masks           (cuda) : list[torch.int64(H,W)]    , 每个视角下的渲染图的sam mask
        multiview_masks     (cuda) : list[torch.int64(N,1)]    , 每个视角下，每个Gaussians点的mask  
        """
        # point guided, masks[id,H,W,C=1], 2D点标注
        def self_prompt_seg_torch(point_prompts, sam_feature): # point_prompts: torch([N,2])
            input_point = point_prompts[None, :, :] # Batch, N, 2
            input_label = torch.ones((input_point.shape[0],input_point.shape[1]), device=self.predictor.device)

            predictor.features = sam_feature
            masks_torch_batch, _, _ = predictor.predict_torch(
                point_coords=input_point, # [1,N,2]
                point_labels=input_label, # [1,N]
                multimask_output=True,
            )
            masks_torch = masks_torch_batch[0] # [batch, id, H, W, C]
            return_mask = (masks_torch[:, :, :, None]*255).to(torch.uint8) # [id,H,W,C=1]
            return return_mask / 255
        
        scene = self.scene
        mask_id = self.maskid
        cameras = scene.getTrainCameras()
        gaussians = scene.gaussians
        sam_masks = []
        multiview_masks = []
        sam_mask_all_levels = []
        for i, view in tqdm(enumerate(cameras), desc="generate multiview masks"):
            image_name = view.image_name # added
            prompts_2d = project_to_2d(view, prompts_3D)
            sam_mask_all_level = self_prompt_seg_torch(prompts_2d, self.sam_features[image_name]) # torch([id=3,H,W,C=1])
            sam_mask_all_levels.append(sam_mask_all_level)
            sam_mask = sam_mask_all_level[mask_id].long()[:,:,0] # torch[H,W]
            sam_masks.append(sam_mask)
            point_mask, indices_mask = mask_inverse(gaussians.get_xyz, view, sam_mask) # TODO: gaussians.get_xyz.require_grad=True; cuda,cuda,cuda
            multiview_masks.append(point_mask.unsqueeze(-1))
        return sam_mask_all_levels, sam_masks, multiview_masks

    def _get_multilayer_mask(self, img_name):
        if self._new_loaded: return None
        return self.record["sam_all"].get(img_name)

    def _get_single_view_seg(self, progress:gr.Progress=None):
        prompts_3D = self.seg2d_mark[img_name]['prompts_3D']  # torch.Tensor[N,3], 单视角
        tmp_sam_masks  = self.record["sam_all"].get(img_name)[self.maskid]['sam_masks'].clone()
        tmp_multiviews = self.record["sam_all"].get(img_name)[self.maskid]['multiview'].clone()
        sam_all, sam_masks, multiview_masks = self.generate_multiview_masks(prompts_3D)
        self.record["sam_all"][img_name] = {"sam_masks": sam_masks, "multiview": multiview_masks}
        self.saved_curr_gs_path = "tmp/saved_current_gs_single_view"
        self.seg_gaussian(threshold=0.7, object_name=self.saved_curr_gs_path, progress=progress)
        self.record["sam_all"][img_name] = {"sam_masks": tmp_sam_masks, "multiview": tmp_multiviews}
        del sam_masks, multiview_masks
        return self.saved_curr_gs_path

    def _get_multi_view_seg(self, progress:gr.Progress=None):
        self.saved_curr_gs_path = "tmp/saved_current_gs_multi_view"
        self.seg_gaussian(threshold=0.7, object_name=self.saved_curr_gs_path, progress=progress)
        return self.saved_curr_gs_path
    
    def _get_mask(self, progress:gr.Progress=None):
        """
        根据prompts_3D和maskid作为索引，保存multiview_sam_masks, multiview_masks
        根据prompts_3D作为索引，保存sam_mask_all_level
        其中prompts_3D为[N,3]的3D点集，索引过程中不需要保证点集顺序一致，对于十分相近的点认为是同一个点。
        历史记录不再依赖栈结构，而是点集匹配。

        sam_mask_all_level : 用于显示,不需要mask_id(显示时做筛选)
        multiview_sam_masks: 是选中的mask,用于3D分割,需要mask_id
        multiview_masks    : 是3D点的mask,用于3D分割,需要mask_id
        """
        if self.segmode == 0:   # Single
            img_name = self.current_image
            prompts_3D_tensor = self.seg2d_mark[img_name]['prompts_3D']  # torch.Tensor[N,3], 单视角
        elif self.segmode == 1: # Multi
            prompts_3D_list = []
            for img_name in self.images.keys():
                prompts_3D = self.seg2d_mark[img_name]['prompts_3D']  # torch.Tensor[N,3]
                if prompts_3D.shape[0] > 0: prompts_3D_list.append(prompts_3D)  # 检查是否有3D点标注
            if prompts_3D_list: prompts_3D_tensor = torch.cat(prompts_3D_list, dim=0)  # [M,3]
            else: prompts_3D_tensor = torch.empty((0, 3), dtype=torch.float32, device="cuda")

        if progress is not None: progress(0, desc="prepare Prompt 3D")
        sam_alls, sam_masks, multiviews = self.generate_multiview_masks(prompts_3D_tensor, progress=progress)
        print("multiview_masks generated\n")
        sam_all_dict = { # img_name -> torch.tensor(id,H,W), 仅显示时转numpy
            img: m.squeeze(-1) if m is not None else None
            for img, m in zip(self.images.keys(), sam_alls)
        }
        self.record["prompts"] = prompts_3D_tensor
        self.record["sam_all"] = sam_all_dict
        self.record["mvmask"][self.maskid] = {
            "multiview": multiviews,
            "sam_masks": sam_masks,
        }
        log()
        return

    def _save_state(self):
        """保存当前标注到历史记录"""
        mark = self.seg2d_mark[self.current_image]
        current_state = (
            mark['points'][:]          if mark['points'] else [],                     # list[N,2] -> int
            mark['prompts_3D'].clone() if 'prompts_3D' in mark else empty_3D.clone()  # Tensor[N,3]
        )
        if mark['history_index'] < len(mark['history']) - 1:
            mark['history'] = mark['history'][:mark['history_index'] + 1]
        mark['history'].append(current_state)
        mark['history_index'] = len(mark['history']) - 1

    def _render_original(self):
        """渲染原始图像，只显示点标注"""
        img = self.images[self.current_image].copy()
        mark = self.seg2d_mark[self.current_image]
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i, (x, y) in enumerate(mark['points']):
            cv2.circle(img, (x, y), 8, (0, 0, 0), -1)
            cv2.circle(img, (x, y), 6, (0, 255, 255), -1)
            cv2.putText(img, str(i+1), (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(img, str(i+1), (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        return img

    def _render_mask(self):
        """渲染带有mask的图像，不包含图例"""
        img = self.images[self.current_image].copy()
        mark = self.seg2d_mark[self.current_image]
        
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        try:
            mask_multilayer = self._get_multilayer_mask(self.current_image).cpu().numpy()
            mask_colors = [
                np.array([255, 50, 50]),    # R
                np.array([50, 255, 50]),    # G
                np.array([50, 50, 255])     # B
            ]
            mask_color_strength = 0.6 # [0,1]
            combined_mask = np.zeros_like(img)
            priority_mask = np.ones(img.shape[:2], dtype=np.int32) * 999
            masks_count = min(len(mask_multilayer), len(mask_colors))
            for i in range(masks_count):
                mask = mask_multilayer[i]
                mask_area = mask > 0.5
                valid_area = np.logical_and(mask_area, priority_mask > i)
                if np.any(valid_area):
                    priority_mask[valid_area] = i
                    combined_mask[valid_area] = mask_colors[i] * mask_color_strength + img[valid_area] * (1-mask_color_strength)
            mask_any = np.any(combined_mask > 0, axis=2)
            img[mask_any] = combined_mask[mask_any]

        except Exception as e:
            if not self._new_loaded:
                print(f"Mask invalid: {self.current_image}")
                print(e)

        for i, (x, y) in enumerate(mark['points']):
            cv2.circle(img, (x, y), 8, (0, 0, 0), -1)
            cv2.circle(img, (x, y), 6, (0, 255, 255), -1)
            cv2.putText(img, str(i+1), (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(img, str(i+1), (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
        return img

    def _render_selected_mask(self):
        """渲染选中的单个mask应用到原图的效果"""
        img = self.images[self.current_image].copy()
        if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        try:
            mask_multilayer = self._get_multilayer_mask(self.current_image).cpu().numpy()
            mask_id = self.maskid
            if mask_id < len(mask_multilayer):
                mask = mask_multilayer[mask_id]
                mask_area = mask > 0.5
                mask_colors = [
                    np.array([255, 50, 50]),    # R
                    np.array([50, 255, 50]),    # G
                    np.array([50, 50, 255])     # B
                ]
                mask_color = mask_colors[mask_id % len(mask_colors)]
                mask_color_strength = 0.6
                
                if np.any(mask_area):
                    img[mask_area] = mask_color * mask_color_strength + img[mask_area] * (1-mask_color_strength)
        except Exception as e:
            if not self._new_loaded:
                print(f"Mask invalid: {self.current_image}")
                print(e)
        return img

    ## Multi-view label voting 多视角（multi-view）标签投票融合
    def ensemble(self, threshold=0.7):
        multiview_masks = self.record["mvmask"][self.maskid]["multiview"]
        multiview_masks = torch.cat(multiview_masks, dim=1)
        vote_labels,_ = torch.mode(multiview_masks, dim=1)
        matches = torch.eq(multiview_masks, vote_labels.unsqueeze(1))
        ratios = torch.sum(matches, dim=1) / multiview_masks.shape[1]
        ratios_mask = ratios > threshold
        labels_mask = (vote_labels == 1) & ratios_mask
        indices_mask = torch.where(labels_mask)[0].detach().cpu()

        return vote_labels, indices_mask

    # 每次做2D->3D分割，都是从一个新的gaussians开始，分割到curr_gaussian里，保存到saved_curr_gs_path里
    # add_to_result时，把last_gaussian保存到saved_last_gs_path里
    # 用户主动保存时，把last_gaussian保存到object_name里
    # seg_gaussian只处理curr_gaussian
    def seg_gaussian(self, threshold, progress:gr.Progress = None):

        if progress is not None: progress(0, desc="Seg3D - Loading Gaussian")
        cameras = self.scene.getTrainCameras()
        curr_gaussian = self.load_scene_gaussians()

        if progress is not None: progress(0.2, desc="Seg3D - Get Mask")
        self._get_mask(progress=progress) # 需要match maskid
        _, final_mask = self.ensemble(threshold)

        if progress is not None: progress(0.4, desc="Seg3D - Decomposition")
        for i, view in tqdm(enumerate(cameras)):
            if self.args.gd_interval != -1 and i % self.args.gd_interval == 0:
                input_mask = self.record["mvmask"][self.maskid]["sam_masks"][i]
                curr_gaussian = gaussian_decomp(curr_gaussian, view, input_mask, final_mask.to('cuda'))

        if progress is not None: progress(0.6, desc="Seg3D - Segment")
        save_gs(curr_gaussian, final_mask, self.saved_curr_gs_path)

        return self.saved_curr_gs_path

    def add_to_result(self):
        if self.last_gaussian is None: # 第一次，直接复制给结果
            self.last_gaussian = self.load_curr_gaussians()
            if self.last_gaussian is not None: # None即无文件
                os.system(f"cp {self.saved_curr_gs_path} {self.saved_last_gs_path}")
        else: # 非第一次，叠加到结果
            curr_gaussian = self.load_curr_gaussians()
            if curr_gaussian is not None: # None即无文件
                self.last_gaussian = self.last_gaussian + curr_gaussian
                self.last_gaussian.save_ply(self.saved_last_gs_path)
        return self.saved_last_gs_path

    def save_object(self, object_name):
        if self.last_gaussian is not None:
            path = os.path.join(self.scene.model_path, f'objects/{object_name}/fg.ply')
            self.last_gaussian.save_ply(path)
            return path
        else:
            print("self.last_gaussian is None.")
            return None

    # 标注情况暂存在变量里，Gaussian存为文件
    def save_ckpt(self, progress:gr.Progress = None):
        if progress is not None: progress(0, desc="Save Annotation")
        self.has_ckpt = True
        self.ckpt_seg2d_mask = {}
        for key, value in self.seg2d_mark.items():
            self.ckpt_seg2d_mask[key] = {
                'points': value['points'][:],
                'history': [(points[:], prompts_3D.clone() if prompts_3D is not None else None) 
                           for points, prompts_3D in value['history']],
                'history_index': value['history_index'],
                'prompts_3D': value['prompts_3D'].clone() if value['prompts_3D'] is not None else empty_3D.clone()
            }
        self.ckpt_record = {
            "prompts": self.record["prompts"].clone() if self.record["prompts"] is not None else None,
            "mvmask": {
                mask_id: {
                    "multiview": [mask.clone() for mask in masks["multiview"]],
                    "sam_masks": [mask.clone() for mask in masks["sam_masks"]]
                }
                for mask_id, masks in self.record["mvmask"].items()
            },
            "sam_all": {
                img_name: mask.clone() if mask is not None else None
                for img_name, mask in self.record["sam_all"].items()
            }
        }
        if progress is not None: progress(0.3, desc="Save Current Gaussian")
        if os.path.isfile(self.saved_curr_gs_path): os.system(f"cp {self.saved_curr_gs_path} {self.saved_curr_ckpt_gs_path}")
        else: os.system(f"rm -f {self.saved_curr_ckpt_gs_path}")
        if progress is not None: progress(0.7, desc="Save Result Gaussian")
        if os.path.isfile(self.saved_last_gs_path): os.system(f"cp {self.saved_last_gs_path} {self.saved_last_ckpt_gs_path}")
        else: os.system(f"rm -f {self.saved_last_ckpt_gs_path}")
        if progress is not None: progress(1  , desc="Finished")
        print("Checkpoint Saved")

    # 标注情况暂存在变量里，Gaussian存为文件
    def load_ckpt(self, progress:gr.Progress = None):
        if progress is not None: progress(0, desc="Load Annotation")
        if self.has_ckpt:
            self.seg2d_mask = self.ckpt_seg2d_mask
            self.record = self.ckpt_record
            if progress is not None: progress(0, desc="Load Current Gaussian")
            if os.path.isfile(self.saved_curr_ckpt_gs_path):
                os.system(f"cp {self.saved_curr_ckpt_gs_path} {self.saved_curr_gs_path}")
            if progress is not None: progress(0, desc="Load Result Gaussian")
            if os.path.isfile(self.saved_last_ckpt_gs_path):
                self.last_gaussian = self.load_gaussians(self.saved_last_ckpt_gs_path)
                os.system(f"cp {self.saved_last_ckpt_gs_path} {self.saved_last_gs_path}")
            if progress is not None: progress(0, desc="Checkpoint Loaded")
        else:
            if progress is not None: progress(1, desc="No Checkpoint")
            print("No Checkpoint")
        return self.saved_curr_gs_path, self.saved_last_gs_path

def create_gradio_interface(default_model_paths, predictor):
    tool = GradioAnnotationTool(model_path=None, predictor=predictor)
    
    with gr.Blocks(
        css="""
            .no-padding-border {
                padding: 0 !important;
                height: 40px !important;
            }
            #model-path-dropdown {
                margin: 0 !important;
            }
            #object-name-input {
                margin: 0 !important;
                height: 38px !important;
                padding: 4px !important;
            }
        """
    ) as demo:
        with gr.Row():
            original_display      = gr.Image(label="Origin", interactive=False)
            mask_display          = gr.Image(label="Layer Mask", interactive=False)
            selected_mask_display = gr.Image(label="Mask Result", interactive=False)
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    undo_btn = gr.Button("撤销标注")
                    redo_btn = gr.Button("重做标注")
                with gr.Row():
                    seg_curr = gr.Button("分割为3D")
                    add_curr = gr.Button("加入最终结果")

            with gr.Column(scale=1):
                with gr.Row():
                    cls_cur_btn = gr.Button("清空当前视角标注")
                    cls_all_btn = gr.Button("清空所有视角标注")
                with gr.Row():
                    ckp_save = gr.Button("保存暂存点")
                    ckp_rest = gr.Button("恢复暂存点")

            with gr.Column(scale=1):
                with gr.Row():
                    mask_selector = gr.Radio(
                        choices=["S", "M", "L"], 
                        label="SAM Mask Layer", 
                        value="S",
                        interactive=True
                    )
                    mode_selector = gr.Radio(
                        choices=["Single", "Multi"], 
                        label="Annotation Mode", 
                        value="Single",
                        interactive=True
                    )

        with gr.Row():
            with gr.Column(scale=3):
                model_path_input = gr.Dropdown(
                    choices=default_model_paths,
                    allow_custom_value=True,
                    value="",
                    show_label=False,
                    interactive=True,
                    scale=1,
                    min_width=40,
                    container=False,
                    elem_classes="no-padding-border",
                    elem_id="model-path-dropdown"
                )
            with gr.Column(scale=1):
                model_path_btn = gr.Button("打开3D模型")
            with gr.Column(scale=2):
                object_name_input = gr.Textbox(
                    placeholder="Your Object Name", 
                    show_label=False, 
                    interactive=True, 
                    submit_btn="Save to Object File",
                    scale=1,
                    min_width=38,
                    # container=False,
                    # elem_classes="no-padding",
                    elem_id="object-name-input"  # 添加唯一ID以便样式定位
                )
        with gr.Row():
            with gr.Column(scale=2):
                gallery = gr.Gallery(
                    label="Gallery",
                    value=[(tool.images[k],k) for k in sorted(tool.images.keys(), key=lambda x: int(x) if x.isdigit() else x)],
                    columns=4,
                    object_fit="contain",
                    height="auto",
                    allow_preview=False,
                )
            with gr.Column(scale=1):
                with gr.Row(): curr_gs = LitModel3D(label="临时结果", exposure=10.0, height=300)
                with gr.Row(): download_curr_gs = gr.DownloadButton(label="下载", interactive=False)
            with gr.Column(scale=1):
                with gr.Row(): last_gs = LitModel3D(label="最终结果", exposure=10.0, height=300)
                with gr.Row(): download_last_gs = gr.DownloadButton(label="下载", interactive=False)
        
        def on_mask_select(choice):
            if choice == "S": mask_id = 0
            elif choice == "M": mask_id = 1
            elif choice == "L": mask_id = 2
            selected_mask = tool.set_mask_id(mask_id)
            return selected_mask

        def on_mode_select(choice):
            if choice == "Single": mask_id = 0
            elif choice == "Multi": mask_id = 1
            original, mask, selected_mask = tool.set_mode_id(mask_id)
            return original, mask, selected_mask

        def get_available_mask_choices(tool):
            # mark = tool.seg2d_mark[tool.current_image]
            # if tool._get_multilayer_mask() is None: return ["S"]
            # mask_names =  ["S", "M", "L"]
            # return mask_names[:len(tool._get_multilayer_mask())]
            return ["S", "M", "L"]
            
        def add_point_from_original(evt: gr.SelectData):
            orig, mask, selected = tool.add_point(evt, is_original=True)
            available_masks = get_available_mask_choices(tool)
            mask_id = tool.maskid
            if mask_id >= len(available_masks): mask_id = len(available_masks)-1
            return orig, mask, selected

        def add_point_from_mask(evt: gr.SelectData):
            orig, mask, selected = tool.add_point(evt, is_original=False)
            available_masks = get_available_mask_choices(tool)
            mask_id = tool.maskid
            if mask_id >= len(available_masks): mask_id = len(available_masks)-1
            return orig, mask, selected

        def load_with_progress(model_path, progress=gr.Progress(track_tqdm=True)):
            torch.cuda.empty_cache()
            tool.load_gaussian_scene(model_path, progress=progress)
            orig = tool._render_original()
            mask = tool._render_mask()
            selected = tool._render_selected_mask()
            available_masks = get_available_mask_choices(tool)
            return \
                gr.update(value = [(tool.images[k],k) for k in sorted(tool.images.keys(), key=lambda x: int(x) if x.isdigit() else x)]),\
                orig, mask, selected

        def save_ckpt(progress=gr.Progress(track_tqdm=True)):
            tool.save_ckpt(progress)
            return gr.update(interactive=True), gr.update(interactive=True)

        def load_ckpt(progress=gr.Progress(track_tqdm=True)):
            path1, path2 = tool.load_ckpt(progress)
            orig, mask, selected = tool._render_original(), tool._render_mask(), tool._render_selected_mask()
            return orig, mask, selected, path1, path1, path2, path2

        def get_curr_seg(progress=gr.Progress(track_tqdm=True)):
            path = tool.seg_gaussian(0.7, progress=progress)
            return path, gr.update(value=path, interactive=True)

        def add_curr_seg():
            path = tool.add_to_result()
            return path, gr.update(value=path, interactive=True)

        # 加载模型
        model_path_btn.click(
            fn=load_with_progress,
            inputs=model_path_input,
            outputs=[gallery, original_display, mask_display, selected_mask_display],
        )

        # 标点
        original_display.select(
            fn=add_point_from_original,
            outputs=[original_display, mask_display, selected_mask_display]
        )
        
        mask_display.select(
            fn=add_point_from_mask,
            outputs=[original_display, mask_display, selected_mask_display]
        )
        
        # 选择模式
        mask_selector.change(
            fn=on_mask_select,
            inputs=[mask_selector],
            outputs=[selected_mask_display]
        )

        mode_selector.change(
            fn=on_mode_select,
            inputs=[mask_selector],
            outputs=[original_display, mask_display, selected_mask_display, selected_mask_display]
        )

        # 操作
        undo_btn.click(fn=lambda: tool.undo(),   outputs=[original_display, mask_display, selected_mask_display])
        redo_btn.click(fn=lambda: tool.redo(),   outputs=[original_display, mask_display, selected_mask_display])
        cls_cur_btn.click(fn=tool.clear_current, outputs=[original_display, mask_display, selected_mask_display])
        cls_all_btn.click(fn=tool.clear_all,     outputs=[original_display, mask_display, selected_mask_display])

        seg_curr.click(fn=get_curr_seg, outputs=[curr_gs, download_curr_gs])
        add_curr.click(fn=add_curr_seg, outputs=[last_gs, download_last_gs])

        ckp_save.click(fn=save_ckpt, outputs=[ckp_save, ckp_rest])
        ckp_rest.click(fn=load_ckpt, outputs=[
            original_display, mask_display, selected_mask_display,
            curr_gs, download_curr_gs, last_gs, download_last_gs
        ])

        def save_gaussian(object_name):
            tool.save_object(object_name)

        def seg_gaussian_with_prompt(object_name, progress=gr.Progress(track_tqdm=True)):
            yield gr.update(value=f"Processing: '{object_name}'...", interactive=False)
            tool.seg_gaussian(threshold=0.7, object_name=object_name, progress=progress)
            yield gr.update(value=object_name, placeholder="Your Object Name", interactive=True)

        object_name_input.submit( # 保存Gaussian
            fn=save_gaussian,
            inputs=object_name_input,
        ).then( # 显示Gaussian
            fn=lambda: (tool.saved_curr_gs_path, gr.update(value=tool.saved_curr_gs_path, interactive=True)),
            outputs=[curr_gs, download_curr_gs]
        ).then( # 显示Gaussian
            fn=lambda: (tool.saved_last_gs_path, gr.update(value=tool.saved_last_gs_path, interactive=True)),
            outputs=[last_gs, download_last_gs]
        )
        
        # Gallery选择事件处理
        def on_gallery_select(evt: gr.SelectData):
            try:
                index = evt.index
                image_name = sorted(tool.images.keys(), key=lambda x: int(x) if x.isdigit() else x)[index]
                orig, mask, selected = tool.update_image(image_name)
                available_masks = get_available_mask_choices(tool)
                mask_id = tool.maskid
                if mask_id >= len(available_masks): mask_id = len(available_masks)-1
                return orig, mask, selected, gr.update(choices=available_masks, value=available_masks[mask_id] if available_masks else None)
            except Exception as e:
                print(f"Error while select in gallery: {e}")
                orig = tool._render_original()
                mask = tool._render_mask()
                selected = tool._render_selected_mask()
                return orig, mask, selected, gr.update(choices=["S"])
            
        gallery.select(fn=on_gallery_select, outputs=[original_display, mask_display, selected_mask_display, mask_selector])

        # 初始显示
        def init_display():
            orig, mask, selected = None, None, None
            available_masks = get_available_mask_choices(tool)
            return orig, mask, selected, gr.update(choices=available_masks, value=available_masks[0] if available_masks else None)
        
        demo.load(fn=init_display, outputs=[original_display, mask_display, selected_mask_display, mask_selector])
        
    return demo



# I'm using this for trellis model segmentation
# As it's a Y-Up generation model
# output model would be lying down.


def find_model_folders(base_path, relative_to="."):
    model_paths = []
    if os.path.exists(base_path):
        # 如果relative_to是None,则相对于当前目录
        if relative_to is None: relative_to = "."
        for root, dirs, files in os.walk(base_path):
            if os.path.basename(root) == "model":
                # 获取相对于relative_to的路径
                rel_path = os.path.relpath(root, relative_to)
                model_paths.append(rel_path)
    return model_paths

if __name__ == "__main__":

    base_dir = "../TRELLIS/results"
    default_model_paths = find_model_folders(base_dir)

    predictor = load_sam()

    demo = create_gradio_interface(default_model_paths, predictor)
    demo.launch(server_name="0.0.0.0", server_port=None,
        allowed_paths = ["./", base_dir])