# GradAug: A New Regularization Method for Deep Neural Networks (NeurIPS'20) [[arXiv]](https://arxiv.org/abs/2006.07989)
This work proposes to utilize randomly transformed training samples to regularize a set of sub-networks. The motivation is that a well-generalized network, and its sub-networks, should recognize transformed images as the same object. The proposed method is simple, general yet effective. It achieves state-of-the-art performance on ImageNet and Cifar classification, and can further improve downstream tasks such as object detection and instance segmentation. The effectiveness is also validated on model robustness and low data regimes.
# Install
- Pytorch 1.0.0+, torchvision, Numpy, pyyaml
- Follow the PyTorch [example](https://github.com/pytorch/examples/tree/master/imagenet) to prepare ImageNet dataset.
# Run
1. ImageNet experiments are conducted on 8 GPUs.

To train ResNet-50,
```
python train.py app:configs/resnet50_randwidth.yml
```
To test a pre-trained model,

Modify `test_only: False` to `test_only: True` in .yml file to enable testing. 

Modify `pretrained: /PATH/TO/YOUR/WEIGHTS` to assign pre-trained weights.

2. Cifar experiments are conducted on 2 GPUs. 

To train WideResNet-28-10,
```
python train_cifar.py app:configs/wideresnet_randwidth.yml
```
To train PyramidNet-200,
```
python train_cifar.py app:configs/pyramidnet_randwidth.yml
```

3. Generate sub-networks by random depth.
```
python train_cifar.py app:configs/resnet_randdepth,yml
```
# Results
1. ImageNet classification accuacy. Note that we report the final-epoch results.

|Model|FLOPs|Top-1|Top-5|
|-----|-----|-----|-----|
|ResNet-50|4.1 G|76.32|92.95|
|+Dropblock|4.1 G|78.13|94.02|
|+Mixup|4.1 G|77.9|93.9|
|+CutMix|4.1 G|78.60|94.08|
|+StochDepth|4.1 G|77.53|93.73|
|+ShakeDrop|4.1 G|77.5|-|
|+GradAug ([Model](https://drive.google.com/file/d/1_0aXi30ET4OrQY0j6KxTYXK0lkgo70Fh/view?usp=sharing))|4.1 G|**78.78**|**94.43**|
|+bag of tricks|4.3 G|79.29|94.38|
|+GradAug+CutMix ([Model](https://drive.google.com/file/d/1cmB3ZIK5-pWt2I37-Pf0I721xZq8MF_4/view?usp=sharing))|4.1 G|**79.67**|**94.93**|

2. Cifar-100 classification accuracy. Note that we report the final-epoch results.

|WideResNet-28-10|Top-1|Top-5|
|----------------|-----|-----|
|Baseline|81.53|95.59|
|+Mixup|82.5|-|
|+CutMix|84.08|96.28|
|+ShakeDrop|81.65|96.19|
|+GradAug ([Model](https://drive.google.com/file/d/1KgsNhBDMGwzg4ursr3Zi57ltuiYspM0b/view?usp=sharing))|83.98|96.17|
|+GradAug+CutMix ([Model](https://drive.google.com/file/d/1Tkr7TLQtst8XnwqPixTXiwLHjN_LbYQQ/view?usp=sharing))|**85.35**|**96.85**|

|PyramidNet-200|Top-1|Top-5|
|--------------|-----|-----|
|Baseline|83.49|94.31|
|+Mixup|84.37|96.01|
|+CutMix|84.83|96.73|
|+ShakeDrop|84.57|97.08|
|+GradAug|84.98|97.08|
|+GradAug+CutMix ([Model](https://drive.google.com/file/d/1MZXH1u3fVijPF_yj7VoskqjNPOXBscQq/view?usp=sharing))|**86.24**|**97.33**|

3. Experiments of generating sub-networks by random depth.

|ResNet-110|Cifar-10|Cifar-100|
|----------|--------|---------|
|Baseline|93.59|72.24|
|+StochDepth|94.75|75.02|
|+GradAug-Randdepth|**94.85**|**77.63** ([Model](https://drive.google.com/file/d/1dMw7p2E_sthg3MN0XhURjvi3VZWWTK3h/view?usp=sharing))|

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
