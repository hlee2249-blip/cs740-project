"""
Please install the following packages:
conda create -n ftgs python=3.12
conda activate ftgs
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python PyYAML plyfile numpy==1.26.4
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0

Usage:
# The following command will generate the test views for the 001_1_seq0 sequence. We recommend referring to the contents in the output folder to prepare for an accurate submission.
python3 render_ftgs.py --ply_path ./compression/test/FreeTimeGS/001_1_seq0.ply --data_dir ./compression/test/001_1_seq0
# You can also add --debug to visualize the difference between the generated views and the ground truth views in validation set.
python3 render_ftgs.py --ply_path ./compression/test/FreeTimeGS/001_1_seq0.ply --data_dir ./compression/test/001_1_seq0 --output_dir ./debug --debug
"""


import os
import json
from dataclasses import dataclass
from tqdm import tqdm
import torch
import numpy as np
import cv2
from plyfile import PlyData
from tqdm import trange

import matplotlib.pyplot as plt


from gsplat.rendering import rasterization

from utils.easy_utils import read_camera_new


def compute_marginal_t(t: torch.Tensor, mu_t: torch.Tensor, cov_t: torch.Tensor):
    return torch.exp(-0.5 * (t - mu_t) ** 2 / cov_t)


@dataclass
class Gaussian3D:
    active_sh_degree: int = 0
    positions: torch.Tensor = torch.empty(0, 3)
    features: torch.Tensor = torch.empty(0, 1, 3)
    opacities: torch.Tensor = torch.empty(0, 1)
    scales: torch.Tensor = torch.empty(0)
    rotations: torch.Tensor = torch.empty(0, 4)


@dataclass
class Gaussian4D(Gaussian3D):
    ts: torch.Tensor = torch.empty(0)
    scales_t: torch.Tensor = torch.empty(0)
    motion: torch.Tensor = torch.empty(0, 3)

    def marginalize_to_3d(self, t, marginal_t_threshold=0.05):
        if isinstance(t, float) or isinstance(t, int):
            t = torch.tensor(t, device=self.ts.device)

        positions = self.positions + self.motion * (t - self.ts)
        # def compute_marginal_t(t: torch.Tensor, mu_t: torch.Tensor, cov_t: torch.Tensor):
        #     return torch.exp(-0.5 * (t - mu_t) ** 2 / cov_t)
        marginal_t = compute_marginal_t(t, self.ts, self.scales_t**2)
        mask = (marginal_t > marginal_t_threshold).flatten()
        marginal_opacities = self.opacities * marginal_t
        gaussian_data = Gaussian3D(
            active_sh_degree=self.active_sh_degree,
            positions=positions[mask],
            features=self.features[mask],
            scales=self.scales[mask],
            rotations=self.rotations[mask],
            opacities=marginal_opacities[mask],
        )
        return gaussian_data
    
    def load_ply(self, path, device="cuda"):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, -1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        ts = np.zeros((xyz.shape[0], 1))
        ts[:, 0] = np.asarray(plydata.elements[0]["t"])

        t_scales = np.zeros((xyz.shape[0], 1))
        t_scales[:, 0] = np.asarray(plydata.elements[0]["t_scale"])

        motion_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("motion_")]
        motion_names = sorted(motion_names, key = lambda x: int(x.split('_')[-1]))
        motions = np.zeros((xyz.shape[0], len(motion_names)))
        for idx, attr_name in enumerate(motion_names):
            motions[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self.positions = torch.tensor(xyz, dtype=torch.float, device=device) 
        features_dc = torch.tensor(features_dc, dtype=torch.float, device=device)
        features_extra = torch.tensor(features_extra, dtype=torch.float, device=device)
        features = torch.cat([features_dc, features_extra], dim=-1)
        self.features = features.transpose(1, 2).contiguous()
        self.opacities = torch.sigmoid(torch.tensor(opacities, dtype=torch.float, device=device))
        self.scales = torch.exp(torch.tensor(scales, dtype=torch.float, device=device))
        self.rotations = torch.tensor(rots, dtype=torch.float, device=device)
        self.ts = torch.tensor(ts, dtype=torch.float, device=device)
        self.scales_t = torch.exp(torch.tensor(t_scales, dtype=torch.float, device=device))
        self.motion = torch.tensor(motions, dtype=torch.float, device=device)
        self.active_sh_degree = int(np.sqrt(self.features.shape[1]) - 1)

        return self

def stats(name, t):
    print(f"\n[{name}] shape={tuple(t.shape)}")
    print("  min:", t.min().item())
    print("  max:", t.max().item())
    print("  mean:", t.mean().item())
    print("  std:", t.std().item())


def plot_hist(name, tensor, bins=50):
    arr = tensor.detach().cpu().numpy().flatten()
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        print(f"[Warning] {name} is empty or only NaN/Inf")
        return

    plt.figure(figsize=(6,4))
    plt.hist(arr, bins=bins)
    plt.title(f"{name} distribution")
    plt.xlabel(name)
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

def plot_hist_log(name, tensor, bins=50):
    arr = tensor.detach().cpu().numpy().flatten()
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]  # log에 음수/0 들어가면 안 됨

    log_arr = np.log10(arr)

    plt.figure(figsize=(6,4))
    plt.hist(log_arr, bins=bins)
    plt.title(f"{name} distribution (log10 scale)")
    plt.xlabel(f"log10({name})")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()



def render_camera_view(ply_path, camera, output_path, frames, fps=60, near=0.01, far=10000, device="cuda", t_code=None, gt_path=None, mask_path=None):
    gaussian_4d = Gaussian4D().load_ply(ply_path, device=device)
    print("Gaussian4D loaded")

    stats("positions", gaussian_4d.positions)
    stats("opacities", gaussian_4d.opacities)
    stats("scales", gaussian_4d.scales)
    stats("rotations", gaussian_4d.rotations)
    stats("ts", gaussian_4d.ts)
    stats("scales_t", gaussian_4d.scales_t)
    stats("motion", gaussian_4d.motion)


    fields = {
    # "positions_x": gaussian_4d.positions[:,0],
    # "positions_y": gaussian_4d.positions[:,1],
    # "positions_z": gaussian_4d.positions[:,2],

    # "opacities": gaussian_4d.opacities.squeeze(-1),

    # "scales_x": gaussian_4d.scales[:,0],
    # "scales_y": gaussian_4d.scales[:,1],
    # "scales_z": gaussian_4d.scales[:,2],

    # "rotations_all": gaussian_4d.rotations,  

    "ts": gaussian_4d.ts.squeeze(-1),
    "scales_t": gaussian_4d.scales_t.squeeze(-1),

    # "motion_x": gaussian_4d.motion[:,0],
    # "motion_y": gaussian_4d.motion[:,1],
    # "motion_z": gaussian_4d.motion[:,2],
    }

    for name, tensor in fields.items():
        plot_hist(name, tensor)
        plot_hist_log(name, tensor)



    K = torch.from_numpy(camera["K"]).float().to(device)
    W, H = camera["W"], camera["H"]
    R = torch.from_numpy(camera["R"]).float().to(device)
    T = torch.from_numpy(camera["T"]).float().to(device)

    w2c = torch.eye(4, device=device)
    w2c[:3, :3] = R
    w2c[:3, 3] = T.squeeze(-1)

    os.makedirs(os.path.abspath(output_path), exist_ok=True)
    for frame_idx in trange(frames, desc=f"Rendering {output_path}"):
        t = frame_idx / fps
        if t_code is not None:
            # adjust the rendering time according to the timecode
            gaussian_3d = gaussian_4d.marginalize_to_3d(t - t_code)
        else:
            gaussian_3d = gaussian_4d.marginalize_to_3d(t)

        # # ---- GPU device debugging ----
        # print("positions device:", gaussian_3d.positions.device)
        # print("features device:", gaussian_3d.features.device)
        # print("K device:", K.device)
        # print("Using rasterization on:", device)
        # # ------------------------------

        
        render_colors, alphas, meta = rasterization(
            gaussian_3d.positions,
            gaussian_3d.rotations,
            gaussian_3d.scales,
            gaussian_3d.opacities.squeeze(-1),
            gaussian_3d.features,
            w2c[None],
            K[None],
            W,
            H,
            near_plane=near,
            far_plane=far,
            sh_degree=gaussian_3d.active_sh_degree,
        )
        img = (render_colors[0] * 255).clamp(0, 255).to(torch.uint8)[..., [2, 1, 0]].cpu().numpy()
        output_name = os.path.join(os.path.abspath(output_path), f"{frame_idx:06d}.jpg")

        # masking
        if mask_path is not None:
            mask_file = os.path.join(mask_path, f"{frame_idx:06d}.png")

            if os.path.exists(mask_file):
                mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    img = cv2.bitwise_and(img, mask_3ch)


        if gt_path is not None:
            gt_image = os.path.join(os.path.abspath(gt_path), f"{frame_idx:06d}.jpg")
            gt_image = cv2.imread(gt_image).astype(np.float32)
            img = img.astype(np.float32)
            diff_image = np.abs(img - gt_image).max(axis=-1).astype(np.uint8)
            diff_image = cv2.applyColorMap(diff_image, cv2.COLORMAP_JET)
            img = np.concatenate([img, gt_image, diff_image], axis=1)

        cv2.imwrite(output_name, img)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--ply_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--near", type=float, default=0.05) #default = 0.05
    # NOTE: you can set larger near to avoid floating artifacts!
    parser.add_argument("--far", type=float, default=10000) #default = 10000
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--use_mask", action="store_true", default=False)
    args = parser.parse_args()

    train_cameras = read_camera_new(args.data_dir, intri_name="train_intri.yml", extri_name="train_extri.yml")
    test_cameras = read_camera_new(args.data_dir, intri_name="test_intri.yml", extri_name="test_extri.yml")

    # NOTE: load timecode if exists
    if os.path.exists(os.path.join(args.data_dir, "timecode.json")):
        time_code = json.load(open(os.path.join(args.data_dir, "timecode.json")))
        print(f"Time offset loaded from {os.path.join(args.data_dir, 'timecode.json')}")
    else:
        time_code = None

    # render train views
    # n_cameras = len(list(train_cameras.keys()))
    # camera_names = sorted(list(train_cameras.keys()))
    # for view_idx in range(n_cameras):
    #     cam_name = camera_names[view_idx]
    #     camera = train_cameras[cam_name]
    #     T = np.array([camera["T"]]).reshape(3, 1)
    #     R = np.array([camera["R"]]).reshape(3, 3)
    #     K = np.array([camera["K"]]).reshape(3, 3)
    #     W = int(K[0, 2] * 2)
    #     H = int(K[1, 2] * 2)
    #     camera = {
    #         "T": T,
    #         "R": R,
    #         "K": K,
    #         "H": H,
    #         "W": W,
    #     }
    #     output_path = os.path.join(args.output_dir, f"{cam_name}")
    #     os.makedirs(output_path, exist_ok=True)
    #     render_camera_view(args.ply_path, camera, output_path, args.frames, args.fps, args.near, args.far, args.device)

    # render test views
    n_cameras = len(list(test_cameras.keys()))
    camera_names = sorted(list(test_cameras.keys()))
    np.random.shuffle(camera_names)
    for view_idx in range(n_cameras):
        cam_name = camera_names[view_idx]
        camera = test_cameras[cam_name]
        T = np.array([camera["T"]]).reshape(3, 1)
        R = np.array([camera["R"]]).reshape(3, 3)
        K = np.array([camera["K"]]).reshape(3, 3)
        W = int(K[0, 2] * 2)
        H = int(K[1, 2] * 2)
        camera = {
            "T": T,
            "R": R,
            "K": K,
            "H": H,
            "W": W,
        }
        if time_code is not None:
            t_code = time_code[cam_name]
        else:
            t_code = None
        output_path = os.path.join(args.output_dir, f"{cam_name}")
        os.makedirs(output_path, exist_ok=True)
        if args.debug:
            gt_path = os.path.join(args.data_dir, "images", cam_name)
        else:
            gt_path = None

        # for masking
        if args.use_mask:
            mask_path = os.path.join(args.data_dir, "masks", cam_name)
            if not os.path.exists(mask_path):
                print(f"Warning: Mask directory not found: {mask_path}")
                mask_path = None
        else:
            mask_path = None


        render_camera_view(args.ply_path, camera, output_path, args.frames, args.fps, args.near, args.far, args.device, t_code, gt_path, mask_path)
