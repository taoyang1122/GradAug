# GradAug: A New Regularization Method for Deep Neural Networks (NeurIPS'20) [[arXiv]](https://arxiv.org/abs/2006.07989)
This work proposes to utilize randomly transformed training samples to regularize a set of sub-networks. The motivation is that a well-generalized network, and its sub-networks, should recognize transformed images as the same object. The proposed method is simple, general yet effective. It achieves state-of-the-art performance on ImageNet and Cifar classification, and can further improve downstream tasks such as object detection and instance segmentation. The effectiveness is also validated on model robustness and low data regimes.
# Install
- Pytorch 1.0.0+, torchvision, Numpy, pyyaml
- Follow the PyTorch [example](https://github.com/pytorch/examples/tree/master/imagenet) to prepare ImageNet dataset.
# Run
1. ImageNet experiments are conducted on 8 GPUs. To train ResNet-50,
```
python train.py app:configs/resnet50_randwidth.yml
```
2. Cifar experiments are conducted on 2 GPUs. 

To train WideResNet-28-10,
```
python train_cifar.py app:configs/wideresnet_randwidth.yml
```
To train PyramidNet-200,
```
python train_cifar.py app:configs/pyramidnet_randwidth.yml
```
# Citation
If you find this useful in your work, please consider citing,
```
@article{yang2020gradaug,
  title={GradAug: A New Regularization Method for Deep Neural Networks},
  author={Yang, Taojiannan and Zhu, Sijie and Chen, Chen},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
