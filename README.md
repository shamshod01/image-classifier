# To Run The Project Copy and Paste code to Google Colab Environment

##Read Briefly about code structure and Project itself below
##Introduction:
This project aims to classify images by applying CNN(Convolutional Neural Network) to various image datasets, and to improve performance by using methods such as network structure design, hyperparameter tuning, batch normalization, and dropout.
The program uses Python 3 and utilizes the PyTorch library.The code is written using Google Colab and include the following features:
-   Dataset download part
-   Model class
-   Iteratively training and testing the model
-   Drawing the learning curve showing the model accuracy according to the epoch

From the given 3 datasets (CIFAR-10, CIFAR-100, SVHN) in this project was used the SVHN dataset.  SVHN is short for Street View House Numbers, and consists of license plate numbers collected from Google Street View. There are a total of 10 classes (0~9), and it consists of 32x32 images of a total of 99289 (73257 trains, 26032 tests).
The CNN model contains 5 Convolutional layers, as optimization algorithm uses Adam, activation function is Rectified Linear Unit.

##Methods:
Defining the Model (Convolutional Neural Network)
I used a convolutional neural network, using the nn.Conv2d class from PyTorch.
The 2D convolution is a fairly simple operation at heart: you start with a kernel, which is simply a small matrix of weights. This kernel “slides” over the 2D input data, performing an elementwise multiplication with the part of the input it is currently on, and then summing up the results into a single output pixel.

nn.Sequential is used to chain the layers and activations functions into a single network architecture.
After each CNN layer used Dropout. In this way, the model can reduce the tendency of some neurons to be over-dependent, and overfitting is reduced because different neurons are used for each training data. I found p = 0.2 more suitable for this model.
In addition, on each layer used Batch normalization to acquire following features:
- Learning becomes more stable as it is less affected by parameter scale or initial value
- Learning rate can be set relatively large, so learning can be faster
- Since the input range is fixed, the vanishing gradient phenomenon is less likely to occur even if sigmoid is used as the activation function.
- It also has a regularization effect.

And as I mentioned above, the activation function is Rectified Linear Unit.
Considering that based on where you're running this notebook, your default device could be a CPU or GPU the following line of code was also included
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
On downloading data batch_size is equal 256 and learning rate is 0.001.