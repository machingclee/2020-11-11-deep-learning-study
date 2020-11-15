# Deep Learning Study
Record what I have learnt from the beginning. Deprecated folder contains material I search from the net but I cannot recall what it is exactly doing.

### 2020-10-10-multiclassification
Implemented the whole back-prop update process from scratch. The back-propagation formula is based on calculating ![equation](http://latex.codecogs.com/svg.latex?\delta_\ell) at the ![equation](http://latex.codecogs.com/svg.latex?\ell)-th layer, and passing it to the ![equation](http://latex.codecogs.com/svg.latex?(\ell-1))-th layer with the following formula: 

![equation](https://latex.codecogs.com/svg.latex?%20\delta_{\ell}%20=%20\frac{1}{m}\cdot%20\Phi^{[\ell]}{}%27(U^{[\ell]})%20*%20\left[W^{[\ell+1]T}%20\cdot%20\delta_{\ell+1}\right]\quad%20\text{with}\quad%20\frac{\partial%20\mathcal%20L}{\partial%20W^{[\ell]}}%20=%20\delta_\ell%20Y^{[\ell-1]T})

Derivations of this formular is recorded in my blog post:
https://checkerlee.blogspot.com/2020/09/derive-formula-of-displaystyle.html

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
