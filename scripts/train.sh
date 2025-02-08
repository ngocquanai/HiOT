#!/bin/bash

#SBATCH --job-name=new_hvit_rev_living17_partial_90
#SBATCH --mail-user=seulki@umich.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24g
#SBATCH --time=240:00:00
#SBATCH --account=stellayu
#SBATCH --partition=stellayu
#SBATCH --output=./logs/%x-%j.log

export PYTHONPATH=deit/:$PYTHONPATH

module purge
module load cuda/11.6.2 cudnn/11.6-v8.4.1

source activate cast

# python deit/main_suppix_hier.py \
#   --model cast_small \
#   --batch-size 256 \
#   --epochs 300 \
#   --num-superpixels 196 --num_workers 8 \
#   --data-set INAT21-MINI-HIER-SUPERPIXEL \
#   --data-path /nfs/turbo/coe-stellayu/shared_data/iNat2021/ \
#   --output_dir /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/inat21_hcast_300 \
#   --filename inat21_hcast_300.csv \
#   --globalkl --gk_weight 0.5 \
#   --resume /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/inat21_hcast_300/checkpoint.pth \

# python deit/main_suppix_hier.py \
#   --model cast_small \
#   --batch-size 256 \
#   --epochs 300 \
#   --num-superpixels 196 --num_workers 8 \
#   --data-set BREEDS-HIER-SUPERPIXEL \
#   --breeds_sort living17 \
#   --data-path /nfs/turbo/coe-stellayu/shared_data/ILSVRC2012/imagenet \
#   --output_dir /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/living17_hcast_300 \
#   --filename living17_hcast_300.csv \
#   --globalkl --gk_weight 0.5 \
#   --sourcefile '_train_source.txt' \
#   --resume /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/living17_cast_small_nofinetune_gk/best_checkpoint.pth 

# python deit/main_suppix_hier.py \
#   --model cast_small \
#   --batch-size 256 \
#   --epochs 100 \
#   --num-superpixels 196 --num_workers 8 \
#   --lr 0.001 \
#   --warmup-lr 0.0001 \
#   --data-set AIR-HIER-SUPERPIXEL \
#   --data-path /home/seulki/ \
#   --output_dir time \
#   --filename time_air_hcast.csv \
#   --globalkl --gk_weight 0.5 \
#   --finetune /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/best_checkpoint.pth


#python -W ignore -m torch.distributed.launch \
 # --nproc_per_node=2 \
 # --use_env deit/main_suppix.py \
 #python -W ignore deit/main_suppix.py \

#BIRD
# python deit/main_suppix.py \
#   --model cast_small \
#   --batch-size 256 \
#   --epochs 100 \
#   --num-superpixels 196 \
#   --num_workers 8 \
#   --data-set AIR-SUPERPIXEL \
#   --category family \
#   --data-path /home/seulki/ \
#   --output_dir /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/air_cast_linearprobe \
#   --finetune /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/best_checkpoint.pth

  # python deit/main_suppix.py \
  # --model cast_small \
  # --batch-size 256 \
  # --epochs 100 \
  # --num-superpixels 196 \
  # --num_workers 8 \
  # --data-set INAT18-SUPERPIXEL \
  # --data-path /nfs/turbo/coe-stellayu/shared_data/iNat2018/ \
  # --output_dir /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/inat_vit_hier \
  # --resume /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/inat_vit_hier/checkpoint.pth

#python deit/main_hier.py \
#  --model deit_small_patch16_224 \
#  --batch-size 256 \
#  --epochs 100 \
#  --num_workers 8 \
#  --data-set INAT18-HIER \
#  --data-path /nfs/turbo/coe-stellayu/shared_data/iNat2018/ \
#  --output_dir /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/inat_vit_hier  \
#  --resume /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/inat_vit_hier/checkpoint.pth 

# python deit/main_hier.py \
#   --model deit_small_patch16_224 \
#   --batch-size 256 \
#   --epochs 100 \
#   --num_workers 8 \
#   --data-set INAT18-HIER \
#   --data-path /nfs/turbo/coe-stellayu/shared_data/iNat2018/ \
#   --output_dir tmp --filename inat_vit_hier_resume_ddp.csv  \
#   --resume /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/inat_vit_hier_ddp/best_checkpoint.pth \
#   --eval


# python deit/main_suppix_hier_global.py \
#   --model cast_small \
#   --batch-size 256 \
#   --epochs 100 \
#   --num-superpixels 196 --num_workers 12 \
#   --data-set AIR-HIER-SUPERPIXEL \
#   --lr 0.001 \
#   --warmup-lr 0.0001 \
#   --data-path /home/seulki \
#   --output_dir /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/cast_air_new_global \
#   --finetune /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/best_checkpoint.pth 
#  --globalkl --gk_weight 1


# python deit/main_suppix_hier.py \
#   --model cast_small \
#   --batch-size 256 \
#   --epochs 100 \
#   --num-superpixels 196 --num_workers 8 \
#   --lr 0.001 \
#   --warmup-lr 0.0001 \
#   --data-set AIR-HIER-SUPERPIXEL \
#   --imb_type bal \
#   --data-path /nfs/turbo/coe-stellayu/shared_data \
#   --output_dir /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/ab_loss_air_flatonly \
#   --filename ab_loss_air_flatonly.csv \
#   --globalkl --gk_weight 0.5 \
#   --finetune /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/best_checkpoint.pth

# python deit/main_suppix_hier.py \
#   --model cast_small \
#   --batch-size 256 \
#   --epochs 100 \
#   --num-superpixels 196 --num_workers 8 \
#   --lr 0.001 \
#   --warmup-lr 0.0001 \
#   --data-set AIR-HIER-SUPERPIXEL-LT \
#   --imb_type bal \
#   --data-path /nfs/turbo/coe-stellayu/shared_data \
#   --output_dir /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/small-bal-air \
#   --globalkl --gk_weight 0.5 \
#   --finetune /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/best_checkpoint.pth


 # --finetune /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/best_checkpoint.pth  \
  # /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/best_checkpoint.pth
#cast_b_coco.pth.tar
 #/nfs/turbo/coe-stellayu/seulki/CAST/tmp/snapshots/moco/imagenet1k/cast_base/lincls/checkpoint.pth.tar
#/nfs/turbo/coe-stellayu/seulki/CAST/tmp/snapshots/deit/imagenet1k/cast_small_deep/best_checkpoint.pth
#/nfs/turbo/coe-stellayu/seulki/CAST/tmp/snapshots/moco/imagenet1k/cast_base/checkpoint_0099.pth.tar air_castbase_0099_gk_0.5
# /nfs/turbo/coe-stellayu/seulki/CAST/tmp/snapshots/moco/coco/cast_small/checkpoint_0399.pth.tar  air_castsmall_coco399_gk_0.5

# Birds
# python deit/main_suppix_hier.py \
#   --model cast_base \
#   --batch-size 256 \
#   --epochs 100 \
#   --num-superpixels 196 --num_workers 8 \
#   --lr 0.001 \
#   --warmup-lr 0.0001 \
#   --data-set AIR-HIER-SUPERPIXEL \
#   --data-path /home/seulki/ \
#   --output_dir /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/cast_base_air_deit_nogk \
#   --finetune /home/seulki/.cache/torch/hub/checkpoints/deit_base_patch16_224-b5f2ef4d.pth 
#   #--globalkl --gk_weight 0.5



#  python deit/main_suppix_hier_partial.py \
#   --model cast_small \
#   --batch-size 256 \
#   --epochs 100 \
#   --num-superpixels 196 --num_workers 8 \
#   --data-set BIRD-HIER-SUPERPIXEL \
#   --proportion 0.45 \
#   --data-path /nfs/turbo/coe-stellayu/shared_data/CUB_200_2011/CUB_200_2011/images_split \
#   --finetune /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/best_checkpoint.pth \
#   --output_dir /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/partial_bird_90_hcast \
#   --globalkl --gk_weight 0.5 
  
  # python deit/main_suppix_hier_partial.py \
  # --model cast_small \
  # --batch-size 256 \
  # --epochs 100 \
  # --num-superpixels 196 --num_workers 8 \
  # --data-set BREEDS-HIER-SUPERPIXEL \
  # --breeds_sort living17 \
  # --proportion 0.9 \
  # --data-path /nfs/turbo/coe-stellayu/shared_data/ILSVRC2012/imagenet \
  # --output_dir /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/partial_hcast_living17_90 \
  # --filename partial_hcast_living17_90.csv \
  # --globalkl --gk_weight 0.5 

  #   --lr 0.001 \
  # --warmup-lr 0.0001 \
  

  # --data-path /home/seulki/CUB_200_2011/images \
  #   --finetune /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/best_checkpoint.pth \

## Eval Code
# python deit/main_suppix_hier_global.py \
#   --model cast_small \
#   --batch-size 256 \
#   --epochs 100 \
#   --num-superpixels 196 --num_workers 8 \
#   --data-set AIR-HIER-SUPERPIXEL \
#   --lr 0.001 \
#   --warmup-lr 0.0001 \
#   --data-path /home/seulki/ \
#   --output_dir tmp --filename cast_air_new_global_gk1.csv \
#   --resume /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/cast_air_new_global_gk1/best_checkpoint.pth \
#   --eval 



# python deit/main_suppix_hier.py \
#   --model cast_small \
#   --batch-size 256 \
#   --epochs 100 \
#   --num-superpixels 196 --num_workers 8 \
#   --data-set BREEDS-HIER-SUPERPIXEL \
#   --breeds_sort living17 \
#   --data-path /nfs/turbo/coe-stellayu/shared_data/ILSVRC2012/imagenet \
#   --output_dir /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/hcast_rev_living17 \
#   --filename hcast_rev_living17.csv \
#   --globalkl --gk_weight 0.5 \
#   --sourcefile '_train_source.txt'

# python deit/main_hier.py \
#   --model deit_small_patch16_224 \
#   --batch-size 256 \
#   --epochs 100 \
#   --seed 0 \
#   --num_workers 8 \
#   --data-set BREEDS-HIER \
#   --breeds_sort living17 \
#   --data-path /nfs/turbo/coe-stellayu/shared_data/ILSVRC2012/imagenet \
#   --output_dir /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/new_hvit_rev_living17 \
#   --filename new_hvit_rev_living17.csv \
#   --sourcefile '_train_source.txt' 
  # --resume /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/vit_hier_living17_ab_2/best_checkpoint.pth \
  # --eval


# python deit/main_suppix_hier.py \
#   --model cast_small \
#   --batch-size 512 \
#   --epochs 1 \
#   --num-superpixels 196 --num_workers 8 \
#   --data-set INAT21-MINI-HIER-SUPERPIXEL \
#   --data-path /nfs/turbo/coe-stellayu/shared_data/iNat2021 \
#   --output_dir /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/inat21_hcast_ddp_sh \
#   --filename inat21_mini_hcast_2.csv \
#   --globalkl --gk_weight 0.5 \
#   --resume /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/inat21_hcast_ddp_sh/best_checkpoint.pth \
#   --eval

# python deit/main_hier.py \
#   --model deit_small_patch16_224 \
#   --batch-size 256 \
#   --epochs 1 \
#   --num_workers 8 \
#   --data-set INAT21-MINI-HIER \
#   --data-path /nfs/turbo/coe-stellayu/shared_data/iNat2021 \
#   --output_dir /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/inat21_mini_vit_hier --filename inat21_mini_vit_hier.csv \
#   --resume /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/inat21_mini_vit_hier/best_checkpoint.pth \
#   --eval

  #--finetune /home/seulki/.cache/torch/hub/checkpoints/deit_small_patch16_224-cd65a155.pth


  # --resume /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/living17_cast_small_nofinetune_gk/best_checkpoint.pth \
  # --path-yn --eval 

  # --imb_type bal \
  # --img_max 26 \
  # python deit/main_suppix_hier.py \
  # --model cast_small \

  # --batch-size 256 \
  # --epochs 1 \
  # --num-superpixels 196 \
  # --num_workers 8 \
  # --data-set INAT18-HIER-SUPERPIXEL \
  # --data-path /nfs/turbo/coe-stellayu/shared_data/iNat2018/ \
  # --output_dir tmp --filename cast_inat_orilr.csv \
  # --resume /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/cast_inat_orilr/best_checkpoint.pth \
  # --eval 


# python deit/main_suppix_hier.py \
#   --model cast_small \
#   --batch-size 256 \
#   --epochs 100 \
#   --num-superpixels 196 --num_workers 8 \
#   --data-set BREEDS-HIER-SUPERPIXEL-LT \
#    --imb_type exp \
#   --breeds_sort living17 \
#   --data-path /nfs/turbo/coe-stellayu/shared_data/ILSVRC2012/imagenet \
#   --output_dir /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/lt_hcast_living \
#    --filename lt_hcast_living.csv 
 #   --img_max 510 \

# python deit/main_hier.py \
#   --model deit_small_patch16_224 \
#   --batch-size 256 \
#   --epochs 1 \
#   --seed 0 \
#   --num_workers 8 \
#   --data-set BREEDS-HIER \
#   --breeds_sort living17 \
#   --data-path /nfs/turbo/coe-stellayu/shared_data/ILSVRC2012/imagenet \
#   --output_dir tmp --filename lt_bs_living17_vit_hier.csv \
#   --resume /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/lt_bs_living17_vit_hier/best_checkpoint.pth \
#   --eval


#eval 
# python deit/main_hier.py \
#   --model deit_small_patch16_224 \
#   --batch-size 256 \
#   --epochs 1 \
#   --seed 0 \
#   --num_workers 8 \
#   --data-set AIR-HIER \
#   --data-path /nfs/turbo/coe-stellayu/shared_data \
#   --output_dir tmp --filename lt_air_vit_hier.csv \
#   --resume /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/lt_air_vit_hier/best_checkpoint.pth \
#   --eval

#/CUB_200_2011/images

# python deit/main_hier_partial.py \
#   --model deit_small_patch16_224 \
#   --batch-size 256 \
#   --epochs 100 \
#   --seed 0 \
#   --num_workers 8 \
#   --data-set BIRD-HIER \
#   --data-path /nfs/turbo/coe-stellayu/shared_data/CUB_200_2011/CUB_200_2011/images_split \
#   --output_dir /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/hvit_rev_cub_partial_0.45 --filename hvit_rev_cub_partial_0.45.csv \
#   --finetune /home/seulki/.cache/torch/hub/checkpoints/deit_small_patch16_224-cd65a155.pth \
#   --proportion 0.45

  python deit/main_hier_partial.py \
  --model deit_small_patch16_224 \
  --batch-size 256 \
  --epochs 100 \
  --num_workers 8 \
  --data-set BREEDS-HIER \
  --breeds_sort living17 \
  --proportion 0.9 \
  --data-path /nfs/turbo/coe-stellayu/shared_data/ILSVRC2012/imagenet \
  --output_dir /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/new_hvit_rev_living17_partial_90 \
  --filename new_hvit_rev_living17_partial_90.csv 


# python deit/main_suppix_hier.py \
#   --model cast_small \
#   --batch-size 256 \
#   --epochs 100 \
#   --num-superpixels 196 --num_workers 8 \
#   --data-set BIRD-HIER-SUPERPIXEL \
#   --data-path /nfs/turbo/coe-stellayu/shared_data/CUB_200_2011/CUB_200_2011/images_split \
#   --output_dir /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/hcast_rev_cub --filename hcast_rev_cub.csv \
#   --finetune /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/best_checkpoint.pth \
#   --globalkl --gk_weight 0.5 


# python deit/main.py \
#   --model deit_small_patch16_224 \
#   --batch-size 256 \
#   --epochs 100 \
#   --seed 0 \
#   --num_workers 8 \
#   --data-set AIR \
#   --category order \
#   --data-path /nfs/turbo/coe-stellayu/shared_data \
#   --output_dir /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/air_vit_order_2409 \
#   --finetune /home/seulki/.cache/torch/hub/checkpoints/deit_small_patch16_224-cd65a155.pth

# python deit/main_hier.py \
#   --model deit_small_patch16_224 \
#   --batch-size 256 \
#   --epochs 100 \
#   --seed 0 \
#   --num_workers 8 \
#   --data-set AIR-HIER-LT \
#   --imb_type bal \
#   --img_max 26 \
#   --data-path /nfs/turbo/coe-stellayu/shared_data \
#   --output_dir /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/bal_small_air_vit_hier \
#   --filename bal_small_air_vit_hier.csv \
#   --finetune /home/seulki/.cache/torch/hub/checkpoints/deit_small_patch16_224-cd65a155.pth
 
 # --finetune /home/seulki/.cache/torch/hub/checkpoints/deit_base_patch16_224-b5f2ef4d.pth


# python deit/main_hier.py \
#   --model deit_small_patch16_224 \
#   --batch-size 256 \
#   --epochs 100 \
#   --seed 0 \
#   --num_workers 8 \
#   --data-set BREEDS-HIER-LT \
#   --imb_type exp \
#   --breeds_sort living17 \
#   --data-path /nfs/turbo/coe-stellayu/shared_data/ILSVRC2012/imagenet \
#   --output_dir /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/lt_bs_living17_vit_hier \
#   --filename lt_bs_living17_vit_hier.csv 


# python deit/main_suppix.py \
#   --model cast_small \
#   --batch-size 256 \
#   --epochs 100 \
#   --num-superpixels 196 --num_workers 8 \
#   --data-set BREEDS-SUPERPIXEL \
#   --breeds_sort living17 \
#   --category family \
#   --data-path /nfs/turbo/coe-stellayu/shared_data/ILSVRC2012/imagenet \
#   --output_dir /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/living17_cast_flat_family 

# python deit/main_suppix.py \
#   --model cast_small \
#   --batch-size 256 \
#   --epochs 100 \
#   --num-superpixels 196 \
#   --num_workers 8 \
#   --data-set AIR-SUPERPIXEL \
#   --category family \
#   --data-path /nfs/turbo/coe-stellayu/shared_data \
#   --output_dir tmp \
#   --filename air_cast_flat_family.csv \
#   --resume /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/air_flat/best_checkpoint.pth \
#   --eval

# python deit/main.py \
#   --model deit_small_patch16_224 \
#   --batch-size 256 \
#   --epochs 1 \
#   --num_workers 8 \
#   --data-set AIR \
#   --category family \
#   --data-path /home/seulki/ \
#   --output_dir tmp \
#   --filename air_vit_flat_family.csv \
#   --resume /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/air_vit_small_flat/best_checkpoint.pth \
#   --eval

  # python deit/main_suppix.py \
  # --model cast_small \
  # --batch-size 256 \
  # --epochs 100 \
  # --num-superpixels 196 --num_workers 8 \
  # --data-set BREEDS-SUPERPIXEL \
  # --breeds_sort living17 \
  # --category family \
  # --data-path /nfs/turbo/coe-stellayu/shared_data/ILSVRC2012/imagenet \
  # --output_dir tmp --filename cast_living17_flat_family.csv \
  # --resume /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/living17_cast_flat_family/checkpoint.pth \
  # --eval 

  # python deit/main.py \
  # --model deit_small_patch16_224 \
  # --batch-size 256 \
  # --epochs 100 \
  # --seed 0 \
  # --num_workers 8 \
  # --data-set BREEDS \
  # --breeds_sort living17 \
  # --category family \
  # --data-path /nfs/turbo/coe-stellayu/shared_data/ILSVRC2012/imagenet \
  # --output_dir tmp --filename vit_living17_flat_family.csv \
  # --resume /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/living17_vit_family/best_checkpoint.pth \
  # --eval

  # python deit/main_posthoc.py \
  # --model deit_small_patch16_224 \
  # --batch-size 256 \
  # --epochs 100 \
  # --seed 0 \
  # --num_workers 8 \
  # --data-set AIR-HIER \
  # --category name \
  # --breeds_sort living17 \
  # --data-path /nfs/turbo/coe-stellayu/shared_data \
  # --output_dir tmp --filename posthoc_air.csv \
  # --resume /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/air_vit_name_2409/best_checkpoint.pth \
  # --auxmodel /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/air_vit_family_2409/best_checkpoint.pth \
  # --eval

# python deit/main.py \
#   --model deit_small_patch16_224 \
#   --batch-size 256 \
#   --epochs 1 \
#   --seed 0 \
#   --num_workers 8 \
#   --data-set AIR \
#   --category family \
#   --data-path /nfs/turbo/coe-stellayu/shared_data/ \
#   --output_dir tmp --filename posthoc_air.csv \
#   --resume /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/air_vit_flat_family/best_checkpoint.pth \
#   --eval

  # python deit/main_posthoc.py \
  # --model deit_small_patch16_224 \
  # --batch-size 256 \
  # --epochs 100 \
  # --seed 0 \
  # --num_workers 8 \
  # --data-set BREEDS-HIER \
  # --category family \
  # --breeds_sort living17 \
  # --data-path /nfs/turbo/coe-stellayu/shared_data/ILSVRC2012/imagenet \
  # --output_dir tmp --filename posthoc_living17.csv \
  # --resume /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/living17_vit_small_name/best_checkpoint.pth \
  # --auxmodel /nfs/turbo/coe-stellayu/seulki/CAST/snapshots/living17_vit_family/best_checkpoint.pth \
  # --eval
