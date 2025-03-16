# Stochastic Resetting Mitigates Latent Gradient Bias of SGD from Label Noise

[![arxiv]([![image](https://github.com/user-attachments/assets/fa6179cf-a5f4-4ee2-a044-529a983ba351)
](http://img.shields.io/badge/arXiv-2003.04166-B31B1B.svg))](https://arxiv.org/abs/2406.00396v3)

Authors: Youngkyoung Bae<sup>\*</sup>, Yeongwoo Song<sup>\*</sup>, and Hawoong Jeong<br>
<sub>\* Equal contribution</sub>

This repository is the official implementation of ``Stochastic Resetting Mitigates Latent Gradient Bias of SGD from Label Noise`` ([arXiv:2406.00396v2](https://arxiv.org/abs/2406.00396v2)).

## Getting started

We implemented our algorithm (basically) with ``Python 3.10.13`` and ``PyTorch==1.12.1``.

Download our repository and install its dependencies.

```
git clone https://github.com/qodudrud/stochastic-resetting.git
cd stochastic-resetting
conda env create --file env.yml
conda activate resetting
```

## Datasets

We use the following datasets in our experiments:

- ciFAIR-10
  - Paper: [Image Classification with Small Datasets: Overview and Benchmark](https://ieeexplore.ieee.org/abstract/document/9770050)
  - Link: [Github](https://github.com/lorenzobrigato/gem) (note that we manually included this repo in our code.)
- CIFAR-10/100
  - Paper: [Learning multiple layers of features from tiny images](https://www.cs.utoronto.ca/~kriz/learning-features-2009-TR.pdf)
  - Link: [cifar-10/100](https://www.cs.toronto.edu/~kriz/cifar.html)
- CIFAR-10N/100N
  - Paper: [Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations](https://openreview.net/forum?id=TBWA6PLJZQm&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2022%2FConference%2FAuthors%23your-submissions))
  - Link: [Github](https://github.com/UCSC-REAL/cifar-10-100nhttps://noisylabels.com)
- ANIMAL-10N
  - Paper: [SELFIE: Refurbishing Unclean Samples for Robust Deep Learning](http://proceedings.mlr.press/v97/song19b.html)
  - Link: [animal-10n official homepage](https://dm.kaist.ac.kr/datasets/animal-10n/)

Download the datasets and put them inside the `data` folder.

For convinience, we share the collections of the above datasets through the following link; [link to the collections](https://www.dropbox.com/scl/fi/wyuwhr5kld7y0erv445tx/resetting_data_collection.zip?rlkey=eldhhz3j8ehi3pjk62c59h6o0&st=m5ln0a9v&dl=0)

## Execution

You can train the models by running the file ``main.py``. Below, we provide an example.

```
python -u main.py \
  --save-path results\
  --data cifar10 \
  --model resnet34 \
  --opt sgd \
  --best-metric loss \
  --opt-reset 1 \
  --tot-iter 50000 \
  --log-iter 10 \
  --lr 0.1 \
  --lr-schedule cosineannealing \
  --weight-decay 0.0005 \
  --batch-size 256 \
  --test-batch-size 4096 \
  --reset-prob 0.001 \
  --threshold-iter 5000 \
  --warmup-iter 0 \
  --momentum 0.9 \
  --seed 1 \
  --noise-rate 0.6 \
  --adaptive 1 \
  --loss-type ce \
```
