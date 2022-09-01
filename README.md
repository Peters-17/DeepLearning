# DeepLearning
Score 100/100
# Assignment Goals
• Implement and train a convolutional neural network (CNN), specifically LeNet
• Understand and count the number of trainable parameters in CNN
• Explore different training configurations such as batch size, learning rate and training epochs.
• Design and customize your own deep network for scene recognition
# Summary
Your implementation in this assignment might take one or two hours to run. We highly recommend to start
working on this assignment early! In this homework, we will explore building deep neural networks, particularly Convolutional Neural Networks (CNNs), using PyTorch. Helper code is provided in this assignment.
In this HW, you are still expected to use the Pytorch library and virtual environment for programming. You can
find the relevant tutorials in HW6 Part I. We omit here to avoid redundancy.
Design a CNN model for MiniPlaces Dataset
In this part, you will design a simple CNN model along with your own convolutional network for a more realistic
dataset – MiniPlaces, again using PyTorch.
# Dataset
MiniPlaces is a scene recognition dataset developed by MIT. This dataset has 120K images from 100 scene categories. The categories are mutually exclusive. The dataset is split into 100K images for training, 10K images for
validation, and 10K for testing.
![image](https://user-images.githubusercontent.com/85666623/187816843-65826e80-7bfc-4734-a2a9-9bef8b291d0d.png)

# Part I Creating LeNet-5
Background: LeNet was one of the first CNNs and its success was foundational for further research into deep
learning. We are implementing an existing architecture in the assignment, but it might be interesting to think
about why early researchers chose certain kernel sizes, padding, and strides which we learned about conceptually
in class.
## In this part, you have to implement a classic CNN model, called LeNet-5 in Pytorch for the MiniPlaces dataset.
We use the following layers in this order:
1. One convolutional layer with the number of output channels to be 6, kernel size to be 5, stride to be 1,
followed by a relu activation layer and then a 2D max pooling layer (kernel size to be 2 and stride to be 2).
2. One convolutional layer with the number of output channels to be 16, kernel size to be 5, stride to be 1,
followed by a relu activation layer and then a 2D max pooling layer (kernel size to be 2 and stride to be 2).
3. A Flatten layer to convert the 3D tensor to a 1D tensor.
4. A Linear layer with output dimension to be 256, followed by a relu activation function.
5. A Linear layer with output dimension to be 128, followed by a relu activation function.
6. A Linear layer with output dimension to be the number of classes (in our case, 100).
You have to fill in the LeNet class in student code.py. You are expected to create the model following this tutorial, which is different from using nn.Sequential() in the last HW.
To help visualization, see the picture below:
![image](https://user-images.githubusercontent.com/85666623/187570886-68517cb3-daf0-42aa-88eb-c4bb78dd8777.png)

# Part II Count the number of trainable parameters of LeNet-5
Background: As discussed in lecture, fully connected models (like what we made in the previous homework) are
dense with many parameters we need to train. After finishing this part, it might be helpful to think about the
number of parameters in this model compared to the number of parameters in a fully connected model of similar
depth (similar number of layers). Especially, how does the difference in size impact efficiency and accuracy?
In this part, you are expected to return the number of trainable parameters of your created LeNet model in the
previous part. You have to fill in the function count model params in student code.py.
The function output is in the unit of Million(1e6). Please do not use any external library which directly calculates
the number of parameters (other libraries, such as NumPy can be used as helpers)!
Hint: You can use the model.named parameters() to gain the name and the corresponding parameters of a model.
Please do not do any rounding to the result.

# Part III Training LeNet-5 under different configurations
Background: A large part of creating neural networks is designing the architecture (part 1). However, there are
other ways of tuning the neural net to change its performance. In this section, we can see how batch size, learning
rate, and number of epochs impact how well the model learns. As you get your results, it might be helpful to think
about how and why the changes have impacted the training.
Based on the LeNet-5 model created in the previous parts, in this section, you are expected to train the LeNet-5
model under different configurations.

