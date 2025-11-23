"""
easymocap/easyvolcap utility functions
"""

import os
from os.path import dirname, exists, join

import cv2
import numpy as np
import torch

from .base_utils import dotdict


class FileStorage(object):
    def __init__(self, filename, isWrite=False):
        version = cv2.__version__
        self.major_version = int(version.split(".")[0])
        self.second_version = int(version.split(".")[1])

        if isWrite:
            os.makedirs(dirname(filename), exist_ok=True)
            self.fs = open(filename, "w")
            self.fs.write("%YAML:1.0\r\n")
            self.fs.write("---\r\n")
        else:
            assert exists(filename), filename
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        self.isWrite = isWrite

    def __del__(self):
        try:
            if self.isWrite:
                self.fs.close()
            else:
                cv2.FileStorage.release(self.fs)
        except Exception:
            pass

    def _write(self, out):
        self.fs.write(out + "\r\n")

    def write(self, key, value, dt="mat"):
        if dt == "mat":
            self._write("{}: !!opencv-matrix".format(key))
            self._write("  rows: {}".format(value.shape[0]))
            self._write("  cols: {}".format(value.shape[1]))
            self._write("  dt: d")
            self._write("  data: [{}]".format(", ".join(["{:.10f}".format(i) for i in value.reshape(-1)])))
        elif dt == "list":
            self._write("{}:".format(key))
            for elem in value:
                self._write('  - "{}"'.format(elem))
        elif dt == "real":
            if isinstance(value, np.ndarray):
                value = value.item()
            self._write("{}: {:.10f}".format(key, value))  # as accurate as possible
        else:
            raise NotImplementedError

    def read(self, key, dt="mat"):
        if dt == "mat":
            output = self.fs.getNode(key).mat()
        elif dt == "list":
            results = []
            n = self.fs.getNode(key)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == "":
                    val = str(int(n.at(i).real()))
                if val != "none":
                    results.append(val)
            output = results
        elif dt == "real":
            output = self.fs.getNode(key).real()
        else:
            raise NotImplementedError
        return output

    def close(self):
        self.__del__(self)


def read_camera_new(data_root, intri_name="intri.yml", extri_name="extri.yml"):
    return read_camera(join(data_root, intri_name), join(data_root, extri_name))


def read_camera(intri_path: str, extri_path: str = None, cam_names=[]) -> dotdict:
    if extri_path is None:
        extri_path = join(intri_path, "extri.yml")
        intri_path = join(intri_path, "intri.yml")
    assert exists(intri_path), intri_path
    assert exists(extri_path), extri_path

    intri = FileStorage(intri_path)
    extri = FileStorage(extri_path)
    cams = dotdict()
    cam_names = sorted(intri.read("names", dt="list"))
    for cam in cam_names:
        # Intrinsics
        cams[cam] = dotdict()
        cams[cam].K = intri.read("K_{}".format(cam))
        cams[cam].H = int(intri.read("H_{}".format(cam), dt="real")) or -1
        cams[cam].W = int(intri.read("W_{}".format(cam), dt="real")) or -1
        cams[cam].invK = np.linalg.inv(cams[cam]["K"])

        # Extrinsics
        Tvec = extri.read("T_{}".format(cam))
        Rvec = extri.read("R_{}".format(cam))
        if Rvec is not None:
            R = cv2.Rodrigues(Rvec)[0]
        else:
            R = extri.read("Rot_{}".format(cam))
            Rvec = cv2.Rodrigues(R)[0]
        RT = np.hstack((R, Tvec))

        cams[cam].R = R
        cams[cam].T = Tvec
        cams[cam].C = -R.T @ Tvec.squeeze()
        cams[cam].RT = RT
        cams[cam].Rvec = Rvec
        cams[cam].P = cams[cam].K @ cams[cam].RT

        # Distortion
        D = intri.read("D_{}".format(cam))
        if D is None:
            D = intri.read("dist_{}".format(cam))
        cams[cam].D = D

        # Time input
        cams[cam].t = extri.read("t_{}".format(cam), dt="real") or 0  # temporal index, might all be 0
        cams[cam].v = extri.read("v_{}".format(cam), dt="real") or 0  # temporal index, might all be 0

        # Bounds, could be overwritten
        cams[cam].n = extri.read("n_{}".format(cam), dt="real") or 0.0001  # temporal index, might all be 0
        cams[cam].f = extri.read("f_{}".format(cam), dt="real") or 1e6  # temporal index, might all be 0
        cams[cam].bounds = extri.read("bounds_{}".format(cam))
        cams[cam].bounds = (
            np.array([[-1e6, -1e6, -1e6], [1e6, 1e6, 1e6]]) if cams[cam].bounds is None else cams[cam].bounds
        )

        # CCM
        cams[cam].ccm = intri.read("ccm_{}".format(cam))
        cams[cam].ccm = np.eye(3) if cams[cam].ccm is None else cams[cam].ccm

    return dotdict(cams)


def write_camera_new(data_root: str, Ks, Rs, Ts, camera_names, intri_name="intri.yml", extri_name="extri.yml"):
    cameras = {}
    if isinstance(Ks, torch.Tensor):
        Ks, Rs, Ts = Ks.cpu().numpy(), Rs.cpu().numpy(), Ts.cpu().numpy()
    for K, R, T, camera_name in zip(Ks, Rs, Ts, camera_names):
        cameras[camera_name] = dotdict(K=K, R=R, T=T)

    write_camera(cameras, data_root, intri_name, extri_name)


def write_camera(cameras: dict, path: str, intri_name: str = "intri.yml", extri_name: str = "extri.yml"):
    os.makedirs(path, exist_ok=True)
    intri_path = join(path, intri_name)
    extri_path = join(path, extri_name)
    intri = FileStorage(intri_path, True)
    extri = FileStorage(extri_path, True)
    cam_names = [key_.split(".")[0] for key_ in cameras.keys()]
    intri.write("names", cam_names, "list")
    extri.write("names", cam_names, "list")

    warnings = []

    for key_, val in cameras.items():
        if not isinstance(val, dotdict):
            val = dotdict(val)

        # Skip special keys
        if key_ == "basenames":
            continue
        # if key_ == 'avg_R': continue
        # if key_ == 'avg_T': continue

        key = key_.split(".")[0]
        # Intrinsics
        intri.write("K_{}".format(key), val.K)
        if "H" in val:
            intri.write("H_{}".format(key), val.H, "real")
        if "W" in val:
            intri.write("W_{}".format(key), val.W, "real")

        # Distortion
        if "D" not in val:
            if "dist" in val:
                val.D = val.dist
            else:
                val.D = np.zeros((5, 1))
        if val.D.shape[0] == 1 and val.D.shape[1] == 4:
            val.D = np.concatenate([val.D.T, np.zeros_like(val.D.T[:1])])
        intri.write("D_{}".format(key), val.D.reshape(5, 1))

        # Extrinsics
        if "R" in val and "Rvec" in val:
            warnings.append(key)
            val.Rvec = cv2.Rodrigues(val.R)[0]
        elif "R" in val and "Rvec" not in val:
            val.Rvec = cv2.Rodrigues(val.R)[0]
        elif "R" not in val and "Rvec" in val:
            val.R = cv2.Rodrigues(val.Rvec)[0]
        else:
            raise ValueError(f"Neither R nor Rvec is present for camera {key}")

        extri.write("R_{}".format(key), val.Rvec)
        extri.write("Rot_{}".format(key), val.R)
        extri.write("T_{}".format(key), val.T.reshape(3, 1))

        # Temporal
        if "t" in val:
            extri.write("t_{}".format(key), val.t, "real")

        # Bounds
        if "n" in val:
            extri.write("n_{}".format(key), val.n, "real")
        if "f" in val:
            extri.write("f_{}".format(key), val.f, "real")
        if "bounds" in val:
            extri.write("bounds_{}".format(key), val.bounds)

        # Color correction matrix
        if "ccm" in val:
            intri.write("ccm_{}".format(key), val.ccm)

        # Fisheye distrotion (deep view video)
        if "rdist" in val:
            intri.write("rdist_{}".format(key), val.rdist)

    if warnings:
        print(f"[Warning] Both R and Rvec were present for {len(warnings)} camera(s): {', '.join(warnings)}. Using R and computing Rvec from it.")