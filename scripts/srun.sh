
export PYTHONPATH=/lustre/scratch/client/movian/research/users/quanpn2/public/HiOT

torchrun --nproc_per_node=2 deit/main_suppix_hier.py \
  --model cast_small \
  --batch-size 512 \
  --epochs 100 \
  --num-superpixels 196 --num_workers 12 \
  --data-set INAT21-MINI-HIER-SUPERPIXEL \
  --data-path ../dataset/ \
  --output_dir ./output/inat21_mini_hcast \
  --ot_loss --ot_weight 0.5 \
  --finetune best_checkpoint.pth \
  --tree_path ./data/inat21_3tree.json --distributed
  

