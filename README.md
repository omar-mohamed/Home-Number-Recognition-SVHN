# Home-Number-Recognition-SVHN
A deep learning project to recognize home numbers in an image.

# Problem:

Given an image, the task is to identify the number in the image. The number to be identified is a sequence of digits, s = s1; s2; : : : ; sn. When determining the accuracy of a digit transcriber, we compute the proportion of the input images for which the length n of the sequence and every element si of the sequence is predicted correctly.


![g1](https://user-images.githubusercontent.com/6074821/41202631-58d5afae-6ccc-11e8-96ec-c995d6cdcb7e.jpg)


# Dataset:

SVHN is a real-world image dataset for developing machine learning and object recognition algorithms with minimal requirement on data preprocessing and formatting. SVHN is obtained from house numbers in Google Street View images.

Link: [SVHN_Dataset](http://ufldl.stanford.edu/housenumbers/)

### Dataset Splitting:

- The original SVHN dataset contains around 33k training image and around 13k test image.
- Download extra training examples from SVHN website around(202k images)
- Add 150k image to training, 50k to test, and around 2300 to validation set.
- Now the training set is around 183k picture
- The test set is around 63k picture
- The validation set is around 2300 picture

# Preprocessing:

- Cropping using the bounding boxes provided in the dataset to 32x32 
- Grayscale and Normalization

![g2](https://user-images.githubusercontent.com/6074821/41202695-6a6d5b62-6ccd-11e8-891d-8422b3233800.jpg)

# Training Architecture:

![g3](https://user-images.githubusercontent.com/6074821/41202747-93b4f5f2-6ccd-11e8-94cd-84663b64711a.jpg)

For more details on the architecture please check our presentation slides.


# Results:

- Reached 89.4% on the 63k test images.
- Reached 91% on 2300 images validation set.

![g4](https://user-images.githubusercontent.com/6074821/41202776-098abe74-6cce-11e8-97bb-8886b580481c.jpg)

![g5](https://user-images.githubusercontent.com/6074821/41202779-14b23408-6cce-11e8-91d8-51527cf457dc.jpg)


# GUI Interface:

We made a GUI interface that takes a picture and classify it using our saved model.

![g6](https://user-images.githubusercontent.com/6074821/41202790-427634d4-6cce-11e8-817b-30b6f128068d.jpg)

# Future Work:

- Try different architectures, meta-parameters, and loss functions to increase accuracy.
- Try to add numbers localization so it can detect numbers regardless of zoom or scale.

# References:

- [Ian Goodfellow Paper, Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42241.pdf)

# Environment Used:
- Python 3.6.1
- Tensorflow 1.3
