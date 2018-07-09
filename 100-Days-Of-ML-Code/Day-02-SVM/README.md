# Support Vector Machines
SVM is the most important classification algorithm used for various applications. The goal of SVM model is to design the best hyperplane(straight line) for a linearly separable binary sets in such a way that it divides the training vector into two separate classes. The best hyperplane is determined by the one with maximum margin(straight line distance between hyperplane and closest elements from hyperplane) from both classes.
In simple terms the two classes should be clearly distinguishable from each other.

## All about SVMs
The support vectors are nothing but the non-zero points which are used to calculate the parameters of the hyperplane(w) that is vectors which support the classification model.
In real world problems, the sets used to be classified, are mostly non linearly separable. So SVM has various kernels for converting the not separable ones to separable ones.
Kernel is simply an arbitrary function that returns a number as output to classify. Kernels capture the similarity and domain knowledge. Kernels such as Polynomial, Gaussian or Sigmoidal etc. can be used.
Selection of the kernel plays an important role here and varies with the requirements of given data. Usually Mercer condition is used to select a good kernel function.

#### Sigmoidal kernel usually performs with lower accuracy than Gaussian kernel as we see below in the implementation as well.

## My Implementation
I have initally implemented the same logisitc regression model that I used in Day 01 to compare the accuracy and performance difference between both.
This is the linear kernel SVM model used for linearly modelled data implemented [here](https://github.com/ditsme/Machine-Learning/blob/master/100-Days-Of-ML-Code/Day-02-SVM/Salary%20Analysis%20using%20SVM.ipynb).
This is the accuracy that I got after implementing SVM.

![image](https://user-images.githubusercontent.com/32769743/42462173-19693ef2-83c0-11e8-9380-60d548c90cbb.png)

The other two kernels-Gaussian and Sigmoidal are also implemented [here](https://github.com/ditsme/Machine-Learning/blob/master/100-Days-Of-ML-Code/Day-02-SVM/Wine_quality_Gaussian_kernel.ipynb) and [here](https://github.com/ditsme/Machine-Learning/blob/master/100-Days-Of-ML-Code/Day-02-SVM/WIne_quality_sigmoid_kernel.ipynb) to predict red wine quality. The data used for this is [ Kaggle](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009).

This is the implementation of Gaussian kernel using Google Colab:

![image](https://user-images.githubusercontent.com/32769743/42463078-bf335582-83c2-11e8-86a0-7e5e7ea25703.png)

This is the implementation of Sigmoidal kernel using Google Colab:

![image](https://user-images.githubusercontent.com/32769743/42463185-0a8d2c1a-83c3-11e8-9ee7-e6b3f52f291a.png)
