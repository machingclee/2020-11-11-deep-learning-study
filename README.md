# Deep Learning Study
Record what I have learnt from the beginning. Deprecated folder contains material I search from the net but I cannot recall what it is exactly doing.

### 2020-10-10-multiclassification
Implemented the whole back-prop update process from scratch. Derivations of important formulars are recorded in my blog post:
https://checkerlee.blogspot.com/2020/09/derive-formula-of-displaystyle.html

The back-propagation formula is based on calculating ![equation](https://latex.codecogs.com/svg.latex?\delta_\ell) at the <img src="http://www.sciweavers.org/tex2img.php?eq=%5Cell&bc=White&fc=Black&im=png&fs=12&ff=cmbright&edit=0" align="center" border="0" alt="\ell" width="11" height="15" /> layer, and passing it to the <img src="http://www.sciweavers.org/tex2img.php?eq=%5Cell-1&bc=White&fc=Black&im=png&fs=12&ff=cmbright&edit=0" align="center" border="0" alt="\ell-1" width="39" height="15" />-th layer with the following formula: 

<img src="http://www.sciweavers.org/tex2img.php?eq=%5Cdelta_%7B%5Cell%7D%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%5Ccdot%20%5CPhi%5E%7B%5B%5Cell%5D%7D%7B%7D%27%28U%5E%7B%5B%5Cell%5D%7D%29%20%2A%20%5Cleft%5BW%5E%7B%5B%5Cell%2B1%5DT%7D%20%5Ccdot%20%5Cdelta_%7B%5Cell%2B1%7D%5Cright%5D%5Cquad%20%5Ctext%7Bwith%7D%5Cquad%20%5Cfrac%7B%5Cpartial%20%5Cmathcal%20L%7D%7B%5Cpartial%20W%5E%7B%5B%5Cell%5D%7D%7D%20%3D%20%5Cdelta_%5Cell%20Y%5E%7B%5B%5Cell-1%5DT%7D&bc=White&fc=Black&im=png&fs=12&ff=cmbright&edit=0" align="center" border="0" alt="\delta_{\ell} = \frac{1}{m}\cdot \Phi^{[\ell]}{}'(U^{[\ell]}) * \left[W^{[\ell+1]T} \cdot \delta_{\ell+1}\right]\quad \text{with}\quad \frac{\partial \mathcal L}{\partial W^{[\ell]}} = \delta_\ell Y^{[\ell-1]T}" width="450" height="36" />


### 2020-10-17-bounding-box-regression
Study how to train a model to draw bounding box of specific object.

### 2020-11-10-first-CNN-shallownet
One Conv layer structure for identifying animals of 3 classes. Also learn how to serialize my model and load my trained weights. Average accuracy is just about 70%. study purpose.

### 2020-11-11-LeNet-implementation
Implement LeNet and train it through the mnist dataset of 0-9.

### 2020-11-12-MiniVGGNet
Implement a similified version of VGG Net. Added dropout layer, added momentum and nesterov acceleration in SGD. Also introduce BatchNormalization to see difference. The validation accuracy is about 0.82.

Some reference for me in this stage:
- [Dropout on convolutional layers is weird](https://towardsdatascience.com/dropout-on-convolutional-layers-is-weird-5c6ab14f19b2)
- [Deep learning for pedestrians: backpropagation in CNNs](https://arxiv.org/abs/1811.11987)
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
- [Andrew Ng's Gradient Descent With Momentum (C2W2L06) video](https://www.youtube.com/watch?v=k8fTYJPd3_I)
- [機器/深度學習-基礎數學(三):梯度最佳解相關算法(gradient descent optimization algorithms)](https://medium.com/@chih.sheng.huang821/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E5%9F%BA%E7%A4%8E%E6%95%B8%E5%AD%B8-%E4%B8%89-%E6%A2%AF%E5%BA%A6%E6%9C%80%E4%BD%B3%E8%A7%A3%E7%9B%B8%E9%97%9C%E7%AE%97%E6%B3%95-gradient-descent-optimization-algorithms-b61ed1478bd7)
