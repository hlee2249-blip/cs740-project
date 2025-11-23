#!/bin/bash

# HAC++ Training Script for Time Slots
# Generated for ./compression/FreeTimeGS/001_1_seq0.ply

echo 'Training slot 0...'
python HAC-plus-main/train.py \
    -s ./compression/test/001_1_seq0 \
    -m ./output/compressed_timeslots/slot_0 \
    --iterations 30000 \
    --lmbda 0.001 \
    --voxel_size 0.001 \
    --feat_dim 50 \
    --n_offsets 10 \
    --use_feat_bank \
    --test_iterations 30000 \
    --save_iterations 30000

echo 'Training slot 1...'
python HAC-plus-main/train.py \
    -s ./compression/test/001_1_seq0 \
    -m ./output/compressed_timeslots/slot_1 \
    --iterations 30000 \
    --lmbda 0.001 \
    --voxel_size 0.001 \
    --feat_dim 50 \
    --n_offsets 10 \
    --use_feat_bank \
    --test_iterations 30000 \
    --save_iterations 30000

echo 'Training slot 2...'
python HAC-plus-main/train.py \
    -s ./compression/test/001_1_seq0 \
    -m ./output/compressed_timeslots/slot_2 \
    --iterations 30000 \
    --lmbda 0.001 \
    --voxel_size 0.001 \
    --feat_dim 50 \
    --n_offsets 10 \
    --use_feat_bank \
    --test_iterations 30000 \
    --save_iterations 30000

echo 'Training slot 3...'
python HAC-plus-main/train.py \
    -s ./compression/test/001_1_seq0 \
    -m ./output/compressed_timeslots/slot_3 \
    --iterations 30000 \
    --lmbda 0.001 \
    --voxel_size 0.001 \
    --feat_dim 50 \
    --n_offsets 10 \
    --use_feat_bank \
    --test_iterations 30000 \
    --save_iterations 30000

echo 'Training slot 4...'
python HAC-plus-main/train.py \
    -s ./compression/test/001_1_seq0 \
    -m ./output/compressed_timeslots/slot_4 \
    --iterations 30000 \
    --lmbda 0.001 \
    --voxel_size 0.001 \
    --feat_dim 50 \
    --n_offsets 10 \
    --use_feat_bank \
    --test_iterations 30000 \
    --save_iterations 30000

