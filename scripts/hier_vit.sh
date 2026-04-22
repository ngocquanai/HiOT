export PYTHONPATH=/lustre/scratch/client/movian/research/users/quanpn2/public/HiOT
torchrun --nproc_per_node=1 --master_port=15463 deit/main_hier.py \
  --model deit_small_patch16_224 \
  --batch-size 256 \
  --epochs 100 \
  --num_workers 12 \
  --data-set INAT21-MINI-HIER \
  --data-path ../dataset/ \
  --output_dir ./output/inat21_mini_hvit \
  --finetune best_checkpoint.pth --distributed

