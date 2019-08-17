# SRCNN-Partial-Convolution
Using Partial Convolution with Zero-Padding in SRCNN 

## Prerequisites
 * Tensorflow
 * Scipy version > 0.18 ('mode' option from scipy.misc.imread function)
 * h5py
 * matplotlib

## Usage
For training, `python main.py --is_train True`
<br>
For testing, `python main.py --is_train False`

## Result
經過訓練15000次，達到效果如下：
![orig](https://github.com/tegg89/SRCNN-Tensorflow/blob/master/result/orig.png)<br>
Bicubic interpolated image:
![bicubic](https://github.com/tegg89/SRCNN-Tensorflow/blob/master/result/bicubic.png)<br>
Super-resolved image:
![srcnn](https://github.com/tegg89/SRCNN-Tensorflow/blob/master/result/srcnn.png)

## References
* [tegg89/SRCNN-Tensorflow](https://github.com/tegg89/SRCNN-Tensorflow) 
<br>
