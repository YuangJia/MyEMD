scenes_1=("scene_9")
# scenes_1=("scene_5" "scene_6" "scene_8" "scene_9" "scene_10" "scene_11")
for scene in "${scenes_1[@]}"; do
    echo "=== Processing $scene ==="
    # 训练
     CUDA_VISIBLE_DEVICES=1 python train.py \
         -s /root/autodl-tmp/GaussianPro/euvs_data/trainset/level1/${scene} \
         -m /root/autodl-tmp/GaussianPro/output/level1/${scene} \
         --flatten_loss --position_lr_init 0.000016 --scaling_lr 0.001 \
         --percent_dense 0.0005 --port 1021 --dataset waymo \
         --normal_loss --sky_seg --depth_loss --propagation_interval 30 \
         --depth_error_min_threshold 0.8 --depth_error_max_threshold 1.0 \
         --propagated_iteration_begin 0 --propagated_iteration_after 0 \
         --patch_size 20 --lambda_l1_normal 0.001 --lambda_cos_normal 0.001 \
#         --start_checkpoint /root/autodl-tmp/GaussianPro/output/level1/scene_11/chkpnt25000.pth
    echo "=== Finished $scene ==="
done

#scenes_2=("scene_1")
#for scene in "${scenes_2[@]}"; do
#    echo "=== Processing $scene ==="
##    # 训练
#     CUDA_VISIBLE_DEVICES=0 python train.py \
#         -s /root/autodl-tmp/GaussianPro/euvs_data/trainset/level2/${scene} \
#         -m /root/autodl-tmp/GaussianPro/output/level2/${scene} \
#         --flatten_loss --position_lr_init 0.000016 --scaling_lr 0.001 \
#         --percent_dense 0.0005 --port 1021 --dataset waymo \
#         --normal_loss --sky_seg --depth_loss --propagation_interval 30 \
#         --depth_error_min_threshold 0.8 --depth_error_max_threshold 1.0 \
#         --propagated_iteration_begin 0 --propagated_iteration_after 0 \
#         --patch_size 20 --lambda_l1_normal 0.001 --lambda_cos_normal 0.001
#    # 渲染
##    python render.py \
##         -m /root/autodl-tmp/GaussianPro/output/level1/${scene}
#
#    echo "=== Finished $scene ==="
#done
