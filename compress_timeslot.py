"""
Time Slot별 FreeTimeGS Gaussian을 HAC++로 압축하는 스크립트

사용법:
python compress_timeslot.py \
    --ply_path ./compression/FreeTimeGS/001_1_seq0.ply \
    --data_dir ./compression/test/001_1_seq0 \
    --output_dir ./output/compressed_timeslots \
    --num_slots 5 \
    --video_length 5.0
"""

import os
import sys
import argparse
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from dataclasses import dataclass
from typing import List, Tuple

# FreeTimeGS의 Gaussian4D 클래스 가져오기
from render_ftgs_new import Gaussian4D


def extract_gaussians_by_timeslot(
    gaussian_4d: Gaussian4D,
    slot_start: float,
    slot_end: float,
    sigma_multiplier: float = 3.0
) -> Tuple[dict, int]:
    """
    특정 time slot에 활성화된 Gaussian들을 추출합니다.

    Args:
        gaussian_4d: FreeTimeGS의 4D Gaussian 데이터
        slot_start: 시작 시간 (초)
        slot_end: 끝 시간 (초)
        sigma_multiplier: 시간 분산의 배수 (기본 3-sigma)

    Returns:
        추출된 Gaussian 데이터와 개수
    """
    # 각 Gaussian의 생존 구간 계산
    # birth_time = t - sigma_multiplier * t_scale
    # death_time = t + sigma_multiplier * t_scale

    ts = gaussian_4d.ts.squeeze(-1)  # [N]
    scales_t = gaussian_4d.scales_t.squeeze(-1)  # [N]

    birth_times = ts - sigma_multiplier * scales_t
    death_times = ts + sigma_multiplier * scales_t

    # time slot과 겹치는 Gaussian 선택
    # 조건: birth_time < slot_end AND death_time > slot_start
    mask = (birth_times < slot_end) & (death_times > slot_start)

    # 마스크 적용
    extracted = {
        'positions': gaussian_4d.positions[mask].cpu().numpy(),
        'opacities': gaussian_4d.opacities[mask].cpu().numpy(),
        'features': gaussian_4d.features[mask].cpu().numpy(),
        'scales': gaussian_4d.scales[mask].cpu().numpy(),
        'rotations': gaussian_4d.rotations[mask].cpu().numpy(),
        'ts': gaussian_4d.ts[mask].cpu().numpy(),
        'scales_t': gaussian_4d.scales_t[mask].cpu().numpy(),
        'motion': gaussian_4d.motion[mask].cpu().numpy(),
        'active_sh_degree': gaussian_4d.active_sh_degree
    }

    return extracted, mask.sum().item()


def save_timeslot_ply(extracted_data: dict, output_path: str):
    """
    추출된 Gaussian을 PLY 파일로 저장합니다.
    """
    n_gaussians = extracted_data['positions'].shape[0]

    if n_gaussians == 0:
        print(f"Warning: No Gaussians found for this time slot. Skipping.")
        return False

    # PLY 속성 구성
    xyz = extracted_data['positions']
    opacities = extracted_data['opacities']

    # Opacity를 logit으로 변환 (저장 시)
    opacities_logit = np.log(opacities / (1 - opacities + 1e-10))

    # Features (SH coefficients)
    features = extracted_data['features']  # [N, num_sh, 3]
    features_dc = features[:, 0, :]  # [N, 3]
    features_rest = features[:, 1:, :].reshape(n_gaussians, -1)  # [N, rest*3]

    # Scales를 log로 변환 (저장 시)
    scales = extracted_data['scales']
    scales_log = np.log(scales)

    rotations = extracted_data['rotations']

    # Temporal parameters
    ts = extracted_data['ts']
    scales_t = extracted_data['scales_t']
    scales_t_log = np.log(scales_t)
    motion = extracted_data['motion']

    # dtype 정의
    dtype_full = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('opacity', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
    ]

    # f_rest 추가
    for i in range(features_rest.shape[1]):
        dtype_full.append((f'f_rest_{i}', 'f4'))

    # scale 추가
    for i in range(scales.shape[1]):
        dtype_full.append((f'scale_{i}', 'f4'))

    # rotation 추가
    for i in range(4):
        dtype_full.append((f'rot_{i}', 'f4'))

    # temporal parameters 추가
    dtype_full.append(('t', 'f4'))
    dtype_full.append(('t_scale', 'f4'))
    for i in range(3):
        dtype_full.append((f'motion_{i}', 'f4'))

    # 데이터 배열 생성
    elements = np.zeros(n_gaussians, dtype=dtype_full)
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['opacity'] = opacities_logit.squeeze()
    elements['f_dc_0'] = features_dc[:, 0]
    elements['f_dc_1'] = features_dc[:, 1]
    elements['f_dc_2'] = features_dc[:, 2]

    for i in range(features_rest.shape[1]):
        elements[f'f_rest_{i}'] = features_rest[:, i]

    for i in range(scales.shape[1]):
        elements[f'scale_{i}'] = scales_log[:, i]

    for i in range(4):
        elements[f'rot_{i}'] = rotations[:, i]

    elements['t'] = ts.squeeze()
    elements['t_scale'] = scales_t_log.squeeze()
    for i in range(3):
        elements[f'motion_{i}'] = motion[:, i]

    # PLY 저장
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(output_path)

    return True


def generate_hac_training_command(
    ply_path: str,
    data_dir: str,
    output_dir: str,
    slot_idx: int,
    lmbda: float = 0.001,
    iterations: int = 30000
) -> str:
    """
    HAC++ 학습 명령어를 생성합니다.
    """
    cmd = f"""python HAC-plus-main/train.py \\
    -s {data_dir} \\
    -m {output_dir}/slot_{slot_idx} \\
    --iterations {iterations} \\
    --lmbda {lmbda} \\
    --voxel_size 0.001 \\
    --feat_dim 50 \\
    --n_offsets 10 \\
    --use_feat_bank \\
    --test_iterations {iterations} \\
    --save_iterations {iterations}"""

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Time Slot별 FreeTimeGS 압축")
    parser.add_argument("--ply_path", type=str, required=True,
                        help="FreeTimeGS PLY 파일 경로")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="원본 데이터셋 디렉토리 (이미지, 카메라 정보 포함)")
    parser.add_argument("--output_dir", type=str, default="./output/compressed_timeslots",
                        help="출력 디렉토리")
    parser.add_argument("--num_slots", type=int, default=5,
                        help="분할할 time slot 개수")
    parser.add_argument("--video_length", type=float, default=5.0,
                        help="비디오 총 길이 (초)")
    parser.add_argument("--lmbda", type=float, default=0.001,
                        help="Rate-distortion trade-off (높을수록 더 압축)")
    parser.add_argument("--sigma_multiplier", type=float, default=3.0,
                        help="시간 분산 배수 (기본 3-sigma)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="연산 장치")
    parser.add_argument("--run_training", action="store_true",
                        help="HAC++ 학습 자동 실행")

    args = parser.parse_args()

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # FreeTimeGS PLY 로드
    print(f"Loading FreeTimeGS PLY from {args.ply_path}...")
    gaussian_4d = Gaussian4D().load_ply(args.ply_path, device=args.device)

    total_gaussians = gaussian_4d.positions.shape[0]
    print(f"Total Gaussians: {total_gaussians:,}")

    # 시간 통계
    ts = gaussian_4d.ts.squeeze(-1)
    scales_t = gaussian_4d.scales_t.squeeze(-1)
    print(f"Temporal center (t) range: [{ts.min().item():.3f}, {ts.max().item():.3f}]")
    print(f"Temporal scale (t_scale) range: [{scales_t.min().item():.4f}, {scales_t.max().item():.4f}]")

    # Time slot 분할
    slot_duration = args.video_length / args.num_slots
    training_commands = []

    print(f"\n{'='*60}")
    print(f"Splitting into {args.num_slots} time slots (each {slot_duration:.2f}s)")
    print(f"{'='*60}\n")

    for i in range(args.num_slots):
        slot_start = i * slot_duration
        slot_end = (i + 1) * slot_duration

        print(f"Time Slot {i}: [{slot_start:.2f}s, {slot_end:.2f}s]")

        # 해당 time slot의 Gaussian 추출
        extracted, count = extract_gaussians_by_timeslot(
            gaussian_4d, slot_start, slot_end, args.sigma_multiplier
        )

        print(f"  - Extracted Gaussians: {count:,} ({100*count/total_gaussians:.1f}%)")

        if count == 0:
            print(f"  - Skipping empty slot")
            continue

        # PLY 저장
        slot_ply_path = os.path.join(args.output_dir, f"slot_{i}.ply")
        success = save_timeslot_ply(extracted, slot_ply_path)

        if success:
            print(f"  - Saved to: {slot_ply_path}")

            # HAC++ 학습 명령어 생성
            cmd = generate_hac_training_command(
                slot_ply_path, args.data_dir, args.output_dir, i, args.lmbda
            )
            training_commands.append(cmd)

        print()

    # 학습 스크립트 저장
    script_path = os.path.join(args.output_dir, "train_all_slots.sh")
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write("# HAC++ Training Script for Time Slots\n")
        f.write(f"# Generated for {args.ply_path}\n\n")

        for i, cmd in enumerate(training_commands):
            f.write(f"echo 'Training slot {i}...'\n")
            f.write(cmd + "\n\n")

    os.chmod(script_path, 0o755)
    print(f"Training script saved to: {script_path}")

    # 선택적으로 학습 실행
    if args.run_training:
        print("\n" + "="*60)
        print("Starting HAC++ training for all slots...")
        print("="*60 + "\n")
        os.system(f"bash {script_path}")
    else:
        print("\n" + "="*60)
        print("To start training, run:")
        print(f"  bash {script_path}")
        print("="*60)


if __name__ == "__main__":
    main()
