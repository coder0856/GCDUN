# Gates-Controlled Deep Unfolding Network for Image Compressed Sensing
## Abstract
Deep Unfolding Networks(DUNs) have demonstrated remarkable success in compressed sensing by integrating opti
mization solvers with deep neural networks. The issue of infor
mation loss during the unfolding process has received significant
 attention. To address this issue, many advanced deep unfolding
 networks utilize memory mechanisms to augment the information
 transmission during iterations. However, most of these networks
 only use the memory module to enhance the proximal mapping
 process instead of adjusting the entire iteration. In this paper,
 we propose an LSTM-inspired proximal gradient descent mod
ule called the Gates-Controlled Iterative Module (GCIM), lead
ing to a Gates-Controlled Deep Unfolding Network (GCDUN) for
 compressed sensing. We utilize the gate units to modulate the
 information flow through the iteration by forgetting the redun
dant information before the gradient descent, providing necessary
 features for the proximal mapping stage, and selecting the key
 information for the next stage. To reduce parameters, we propose
 a parameter-friendly version called Recurrent Gates-Controlled
 Deep Unfolding Networks (RGCDUN), which also achieves great
 performance but with much fewer parameters. Extensive experi
ments manifest that our networks achieve excellent performance.
 The source codes are available at https://github.com/coder0856/GCDUN

## Requirements
python == 3.8

torch == 1.11.0+cu113
## Acknowledgements
Our codes are built on [FSOINet](https://github.com/cwjjun/FSOINet), [ISTA-Net](https://github.com/jianzhangcs/ISTA-Net-PyTorch) and [OPINE-Net](https://jianzhang.tech/projects/OPINENet). We are sincerely thankful to the authors for sharing their codes.
