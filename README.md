# Visually Consistent Hierarchical Image Classification (H-CAST)
By [Seulki Park](https://sites.google.com/view/seulkipark/home), [Youren Zhang](https://www.linkedin.com/in/youren-zhang-92a447251/), [Stella X. Yu](https://web.eecs.umich.edu/~stellayu/), [Sara Beery](https://beerys.github.io/), and [Jonathan Huang](http://www.jonathan-huang.org)

Official implementation of ["Visually Consistent Hierarchical Image Classification"](https://openreview.net/forum?id=IRcv4yFX6z), ICLR 2025.


ImageNet-pretrained [CAST](https://openreview.net/forum?id=IRcv4yFX6z)-small model can be downloaded from: [Link](https://huggingface.co/twke/CAST/blob/main/snapshots/deit/imagenet1k/cast_small/best_checkpoint.pth)


## Installation

Create a conda environment with the following command:
```
# initiate conda env
> conda update conda
> conda env create -f environment.yaml
> conda activate hcast

# install dgl (https://www.dgl.ai/pages/start.html)
> pip install dgl==1.1.3+cu116 -f https://data.dgl.ai/wheels/cu116/dgl-1.1.3%2Bcu116-cp39-cp39-manylinux1_x86_64.whl
```