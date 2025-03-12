# Visually Consistent Hierarchical Image Classification (H-CAST)
By [Seulki Park](https://sites.google.com/view/seulkipark/home), [Youren Zhang](https://www.linkedin.com/in/youren-zhang-92a447251/), [Stella X. Yu](https://web.eecs.umich.edu/~stellayu/), [Sara Beery](https://beerys.github.io/), and [Jonathan Huang](http://www.jonathan-huang.org)   
Official implementation of ["Visually Consistent Hierarchical Image Classification"](https://openreview.net/forum?id=IRcv4yFX6z), ICLR 2025.

## 🔍 Overview
Our method ensures that all levels of hierarchical classification, from fine-grained species recognition to broader category distinctions, are **grounded in consistent visual cues** through segmentation. This shared visual foundation improves prediction consistency across the taxonomy, enhancing accuracy at all levels.   
<img src="images/prior_vs_ours.png" width="700">



## 🛠️ Installation
- Python: 3.10
- CUDA: 12.1
- PyTorch: 2.1.2
- DGL: 2.4.0
- GCC: 11.2.0 (Recommended to avoid errors when running DGL)

Create a conda environment with the following command:
```
# create conda env
> conda create -n hcast python=3.10
> conda activate hcast
> pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121


# install dgl (https://www.dgl.ai/pages/start.html)
> pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html
```


## ▶️  Usage
- ImageNet-pretrained [CAST](https://openreview.net/forum?id=IRcv4yFX6z)-small model can be downloaded from: [Link](https://huggingface.co/twke/CAST/blob/main/snapshots/deit/imagenet1k/cast_small/best_checkpoint.pth)

### [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/)
- arrange_birds.py: Split the CUB dataset into separate train and test folders ('images' -> 'images_split').
```
export PYTHONPATH=deit/:$PYTHONPATH
export PYTHONPATH=deit/dataset/:$PYTHONPATH

python deit/main_suppix_hier.py \
  --model cast_small \
  --batch-size 256 \
  --epochs 100 \
  --num-superpixels 196 --num_workers 8 \
  --globalkl --gk_weight 0.5 \
  --data-set BIRD-HIER-SUPERPIXEL \
  --data-path /data/CUB_200_2011/images_split \
  --output_dir ./output/bird_hcast \
  --finetune best_checkpoint.pth      # location of ImageNet-pretrained CAST checkpoint
```

### Aircraft
```
To-be updated
```
### BREEDS
```
To-be updated
```

---

## 🔗 Code Base
This repository is heavily based on **[CAST](https://github.com/twke18/CAST.git)**.  

---

## 🚀 Upcoming Updates
We are actively working on improving this repository! More updates will be released soon. **Stay tuned!** 🔥


## ✅ TODO (Upcoming Updates)
- [ ] Add support for more datasets
- [ ] Add evaluation script
- [ ] Add our checkpoints


---

## 📢 Citation
If you find this repository helpful, please consider citing our work:
```
@inproceedings{
    park2025visually,
    title={Visually Consistent Hierarchical Image Classification},
    author={Seulki Park and Youren Zhang and Stella X. Yu and Sara Beery and Jonathan Huang},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=7HEMpBTb3R}
}
```
Thank you for your support! 🚀
