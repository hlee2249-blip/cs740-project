"""
Simple Spatio-Temporal Compression for FreeTimeGS
Course Project Level Implementation

학습 없이 바로 적용 가능한 압축 방법:
1. Time Slot 분할
2. Quantization (양자화)
3. Pruning (가지치기)
4. Entropy Coding (DEFLATE/LZMA)

사용법:
python simple_compression.py \
    --ply_path ./compression/FreeTimeGS/001_1_seq0.ply \
    --output_dir ./output/simple_compressed \
    --num_slots 5 \
    --video_length 5.0 \
    --quantization_bits 8 \
    --prune_threshold 0.01
"""

import os
import sys
import argparse
import numpy as np
import torch
import lzma
import gzip
import json
from plyfile import PlyData, PlyElement
from dataclasses import dataclass
from typing import Dict, Tuple
import struct

# FreeTimeGS의 Gaussian4D 클래스 가져오기
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_ftgs_new import Gaussian4D, Gaussian3D


class SimpleCompressor:
    """학습 없이 바로 적용 가능한 간단한 압축기"""

    def __init__(self, quantization_bits=8, prune_threshold=0.01):
        self.quantization_bits = quantization_bits
        self.prune_threshold = prune_threshold

    def quantize(self, data: np.ndarray, bits: int = 8) -> Tuple[np.ndarray, dict]:
        """
        Uniform Quantization: float32 -> int8/int16

        Returns:
            quantized_data: 양자화된 데이터
            metadata: 복원에 필요한 정보 (min, max, scale)
        """
        data_min = data.min()
        data_max = data.max()

        # Avoid division by zero
        if data_max - data_min < 1e-10:
            data_max = data_min + 1e-10

        # Scale to [0, 2^bits - 1]
        max_val = (1 << bits) - 1
        scale = max_val / (data_max - data_min)

        quantized = np.round((data - data_min) * scale).astype(
            np.uint8 if bits <= 8 else np.uint16
        )

        metadata = {
            'min': float(data_min),
            'max': float(data_max),
            'scale': float(scale),
            'bits': bits,
            'shape': list(data.shape)
        }

        return quantized, metadata

    def dequantize(self, quantized: np.ndarray, metadata: dict) -> np.ndarray:
        """양자화된 데이터를 복원"""
        return quantized.astype(np.float32) / metadata['scale'] + metadata['min']

    def prune_gaussians(self, gaussian_4d: Gaussian4D) -> Gaussian4D:
        """
        낮은 opacity를 가진 Gaussian 제거
        """
        opacities = gaussian_4d.opacities.squeeze(-1)
        mask = opacities > self.prune_threshold

        pruned = Gaussian4D()
        pruned.positions = gaussian_4d.positions[mask]
        pruned.opacities = gaussian_4d.opacities[mask]
        pruned.features = gaussian_4d.features[mask]
        pruned.scales = gaussian_4d.scales[mask]
        pruned.rotations = gaussian_4d.rotations[mask]
        pruned.ts = gaussian_4d.ts[mask]
        pruned.scales_t = gaussian_4d.scales_t[mask]
        pruned.motion = gaussian_4d.motion[mask]
        pruned.active_sh_degree = gaussian_4d.active_sh_degree

        return pruned, mask.sum().item()

    def extract_timeslot(
        self,
        gaussian_4d: Gaussian4D,
        slot_start: float,
        slot_end: float,
        sigma_multiplier: float = 3.0
    ) -> Tuple[Gaussian4D, int]:
        """특정 time slot에 활성화된 Gaussian 추출"""
        ts = gaussian_4d.ts.squeeze(-1)
        scales_t = gaussian_4d.scales_t.squeeze(-1)

        birth_times = ts - sigma_multiplier * scales_t
        death_times = ts + sigma_multiplier * scales_t

        mask = (birth_times < slot_end) & (death_times > slot_start)

        extracted = Gaussian4D()
        extracted.positions = gaussian_4d.positions[mask]
        extracted.opacities = gaussian_4d.opacities[mask]
        extracted.features = gaussian_4d.features[mask]
        extracted.scales = gaussian_4d.scales[mask]
        extracted.rotations = gaussian_4d.rotations[mask]
        extracted.ts = gaussian_4d.ts[mask]
        extracted.scales_t = gaussian_4d.scales_t[mask]
        extracted.motion = gaussian_4d.motion[mask]
        extracted.active_sh_degree = gaussian_4d.active_sh_degree

        return extracted, mask.sum().item()

    def compress_slot(self, gaussian_4d: Gaussian4D) -> Tuple[bytes, dict]:
        """
        단일 time slot의 Gaussian을 압축

        Returns:
            compressed_bytes: 압축된 바이트 스트림
            metadata: 복원에 필요한 메타데이터
        """
        # 1. Pruning
        pruned, n_after_prune = self.prune_gaussians(gaussian_4d)

        # 2. Convert to numpy
        data = {
            'positions': pruned.positions.cpu().numpy(),
            'opacities': pruned.opacities.cpu().numpy(),
            'features': pruned.features.cpu().numpy(),
            'scales': pruned.scales.cpu().numpy(),
            'rotations': pruned.rotations.cpu().numpy(),
            'ts': pruned.ts.cpu().numpy(),
            'scales_t': pruned.scales_t.cpu().numpy(),
            'motion': pruned.motion.cpu().numpy(),
        }

        # 3. Quantization
        quantized_data = {}
        metadata = {
            'active_sh_degree': pruned.active_sh_degree,
            'n_gaussians': pruned.positions.shape[0],
            'quantization': {}
        }

        for key, arr in data.items():
            q_arr, q_meta = self.quantize(arr, self.quantization_bits)
            quantized_data[key] = q_arr
            metadata['quantization'][key] = q_meta

        # 4. Serialize to bytes
        buffer = b''
        for key in ['positions', 'opacities', 'features', 'scales',
                    'rotations', 'ts', 'scales_t', 'motion']:
            buffer += quantized_data[key].tobytes()

        # 5. LZMA compression (best ratio)
        compressed = lzma.compress(buffer, preset=9)

        return compressed, metadata

    def decompress_slot(self, compressed: bytes, metadata: dict) -> Gaussian4D:
        """압축된 데이터를 복원"""
        # 1. LZMA decompression
        buffer = lzma.decompress(compressed)

        # 2. Deserialize
        n = metadata['n_gaussians']
        offset = 0
        data = {}

        for key in ['positions', 'opacities', 'features', 'scales',
                    'rotations', 'ts', 'scales_t', 'motion']:
            q_meta = metadata['quantization'][key]
            shape = q_meta['shape']
            dtype = np.uint8 if q_meta['bits'] <= 8 else np.uint16
            size = int(np.prod(shape)) * dtype().itemsize

            q_arr = np.frombuffer(buffer[offset:offset+size], dtype=dtype)
            q_arr = q_arr.reshape(shape)
            offset += size

            # Dequantize
            data[key] = self.dequantize(q_arr, q_meta)

        # 3. Create Gaussian4D
        gaussian = Gaussian4D()
        gaussian.positions = torch.tensor(data['positions'], dtype=torch.float32)
        gaussian.opacities = torch.tensor(data['opacities'], dtype=torch.float32)
        gaussian.features = torch.tensor(data['features'], dtype=torch.float32)
        gaussian.scales = torch.tensor(data['scales'], dtype=torch.float32)
        gaussian.rotations = torch.tensor(data['rotations'], dtype=torch.float32)
        gaussian.ts = torch.tensor(data['ts'], dtype=torch.float32)
        gaussian.scales_t = torch.tensor(data['scales_t'], dtype=torch.float32)
        gaussian.motion = torch.tensor(data['motion'], dtype=torch.float32)
        gaussian.active_sh_degree = metadata['active_sh_degree']

        return gaussian


def calculate_original_size(gaussian_4d: Gaussian4D) -> int:
    """원본 Gaussian 데이터 크기 계산 (bytes)"""
    n = gaussian_4d.positions.shape[0]

    # 각 파라미터의 크기 (float32 = 4 bytes)
    size = 0
    size += n * 3 * 4  # positions
    size += n * 1 * 4  # opacities
    size += gaussian_4d.features.numel() * 4  # features
    size += gaussian_4d.scales.numel() * 4  # scales
    size += n * 4 * 4  # rotations
    size += n * 1 * 4  # ts
    size += n * 1 * 4  # scales_t
    size += n * 3 * 4  # motion

    return size


def main():
    parser = argparse.ArgumentParser(description="Simple Spatio-Temporal Compression")
    parser.add_argument("--ply_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output/simple_compressed")
    parser.add_argument("--num_slots", type=int, default=5)
    parser.add_argument("--video_length", type=float, default=5.0)
    parser.add_argument("--quantization_bits", type=int, default=8,
                        choices=[4, 8, 16])
    parser.add_argument("--prune_threshold", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load FreeTimeGS
    print(f"Loading FreeTimeGS from {args.ply_path}...")
    gaussian_4d = Gaussian4D().load_ply(args.ply_path, device=args.device)

    total_gaussians = gaussian_4d.positions.shape[0]
    original_size = calculate_original_size(gaussian_4d)

    print(f"Total Gaussians: {total_gaussians:,}")
    print(f"Original size: {original_size / 1024 / 1024:.2f} MB")

    # Initialize compressor
    compressor = SimpleCompressor(
        quantization_bits=args.quantization_bits,
        prune_threshold=args.prune_threshold
    )

    # Compress each time slot
    slot_duration = args.video_length / args.num_slots
    total_compressed_size = 0
    all_metadata = {'slots': [], 'params': vars(args)}

    print(f"\n{'='*60}")
    print(f"Compressing {args.num_slots} time slots")
    print(f"{'='*60}\n")

    for i in range(args.num_slots):
        slot_start = i * slot_duration
        slot_end = (i + 1) * slot_duration

        print(f"Slot {i}: [{slot_start:.2f}s, {slot_end:.2f}s]")

        # Extract time slot
        slot_gaussian, n_extracted = compressor.extract_timeslot(
            gaussian_4d, slot_start, slot_end
        )

        if n_extracted == 0:
            print(f"  - Empty slot, skipping")
            continue

        slot_original_size = calculate_original_size(slot_gaussian)

        # Compress
        compressed, metadata = compressor.compress_slot(slot_gaussian)
        compressed_size = len(compressed)

        # Save compressed data
        slot_file = os.path.join(args.output_dir, f"slot_{i}.bin")
        with open(slot_file, 'wb') as f:
            f.write(compressed)

        # Calculate metrics
        ratio = slot_original_size / compressed_size

        print(f"  - Gaussians: {n_extracted:,} -> {metadata['n_gaussians']:,} (after pruning)")
        print(f"  - Original: {slot_original_size/1024:.1f} KB")
        print(f"  - Compressed: {compressed_size/1024:.1f} KB")
        print(f"  - Ratio: {ratio:.1f}x")
        print()

        total_compressed_size += compressed_size

        # Store metadata
        slot_meta = {
            'slot_idx': i,
            'slot_start': slot_start,
            'slot_end': slot_end,
            'n_original': n_extracted,
            'n_after_prune': metadata['n_gaussians'],
            'original_size': slot_original_size,
            'compressed_size': compressed_size,
            'compression_ratio': ratio,
            'metadata': metadata
        }
        all_metadata['slots'].append(slot_meta)

    # Save metadata
    meta_file = os.path.join(args.output_dir, "metadata.json")
    with open(meta_file, 'w') as f:
        json.dump(all_metadata, f, indent=2)

    # Summary
    print("="*60)
    print("COMPRESSION SUMMARY")
    print("="*60)
    print(f"Original total size: {original_size/1024/1024:.2f} MB")
    print(f"Compressed total size: {total_compressed_size/1024/1024:.2f} MB")
    print(f"Overall compression ratio: {original_size/total_compressed_size:.1f}x")
    print(f"\nOutput saved to: {args.output_dir}")
    print("="*60)

    # Test decompression
    print("\nTesting decompression...")
    test_slot = 0
    test_file = os.path.join(args.output_dir, f"slot_{test_slot}.bin")
    if os.path.exists(test_file):
        with open(test_file, 'rb') as f:
            test_compressed = f.read()

        test_meta = all_metadata['slots'][test_slot]['metadata']
        recovered = compressor.decompress_slot(test_compressed, test_meta)
        print(f"Slot {test_slot} decompressed: {recovered.positions.shape[0]} Gaussians")
        print("Decompression test PASSED!")


if __name__ == "__main__":
    main()
