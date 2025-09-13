# CNet
**Neural network framework implemented from scratch**

By contributing to this project, you agree to follow the rules listed in the [rules document](rules.md). Failure to follow the rules means I will hunt you down and [DATA EXPUNGED].

## Foreword
I ran my first neural network about 2 years ago. The network was really deep, but I didn't have a good computer, so the network took hours to train.

Fast forward to last summer. The first neural network that I wrote entirely by myself took about 8 minutes per epoch on my laptop. A full 50-epoch training session would take 6 hours and 40 minutes, almost my entire working day.  
I needed a way to quickly build and train networks on less powerful hardware, not just computers with the fastest GPUs and compute units.  
Even if speed goals are not met, I still wanted to harness the **full power of C++!!!**

Although widely used, PyTorch has features that I found confusing. The most striking feature is that optimizers (or "criterions", in PyTorch language) are separate from the network. I thought that an optimizer, as part of the network, belongs inside the network instead of outside it. Another confusing feature is that PyTorch, and Tensorflow too, forces users to make custom classes to build networks instead of having pre-built network objects available.

Using **the power of C++**, I created an original neural network framework with these design goals:
- Have the network be the central container. All network components exist inside a network object.
- Enable the framework to be extended using abstract classes and polymorphism
- Use straightforward methods for network configuration
- Implement hot-swapping capability: layers can be added, removed, or edited during training



## Setup Instructions
This project requires the Eigen 3 linear algebra package.

After pulling this repo, download Eigen 3:
[zip](https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip), [tar.gz](https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz),
[Eigen website](https://eigen.tuxfamily.org/index.php?title=Main_Page).
Then, extract the archive and place the folder inside your cloned repo's top-level directory.

Directory structure should be as follows:
```
your-repo-name
├── cnet/
├── <EIGEN FOLDER GOES HERE>/
...
```

To use CNet in your program, include the "network.cpp" file in the "cnet" directory:
```
#include "cnet/network.cpp"
```

## Quick Start Guide

Ensure you have the line `#include "cnet/network.cpp"` at the top of the file containing the `int main()` function. For best results, ensure your file with `main()` is in the cloned repo's top-level directory.

To create a network, create a new Network object:
```
Network net = Network();
```
The network will have no layers, loss calculator, or optimizer. These must be added to the network.

To add a layer to the network:
```
//Create a ReLU activation function object, as a smart pointer
shared_ptr<Relu> relu_activation_function_ptr = make_shared<Relu>();

//Add a layer with 3 inputs, 5 outputs, and ReLU activation
net.add_layer(Layer(3, 5, relu_activation_function_ptr));

//Add a layer with 5 inputs, 10 outputs, and no activation
net.add_layer(Layer(3, 5));

//Add a layer with 10 inputs, 2 outputs, and no activation, without using an explicit constructor
net.add_layer(10, 2);

//Highly recommended: free the activation function pointer
relu_activation_function_ptr.reset();
```

Once added to the network, each layer is assigned an index. The first layer has index 0, the second layer has index 1... and the last layer has index `{networkName}.layer_count()`-1. The index allows for editing individual layers.

Layers are added in sequential order. The first layer added will become the input. The last layer added is the output.  
The `insert_layer` method adds layers in positions other than the back of the network.  
The `remove_layer` method erases a layer at a specified index.

To add a loss calculator, i.e. for Mean Squared Error:
```
net.set_loss_calculator(make_shared<MeanSquaredError>());
```

To add an optimizer, i.e. for Stochastic Gradient Descent (SGD):
```
net.set_optimizer(make_shared<SGD>());
```

Note: Both `set_loss_calculator` and `set_optimizer` use smart pointers, to enable memory-safe polymorphism.


To use the network, enable it:
```
net.enable()
```
The method checks these criteria:
- The network has at least 1 layer
- The network has a defined loss calculator and optimizer 
- No layer has Softmax activation, except for the final layer
- The output dimension of each layer equals the input dimension of the next layer

If any criterion is broken, the `enable` method throws the `illegal_state` exception, a subclass of `runtime_error`.  
If all the criteria are met, the network can be trained, but not edited.

To feed an input vector into the network:
```
//Creates the 3D vector, initialized to [1, 0, 1]
Eigen::VectorXd input(3);
input << 1, 0, 1;

//Compute and store the network's output
Eigen::VectorXd predictions = net.forward(input);
```

To compute the reverse process, carrying out backpropagation using the network's optimizer:
```
//Create a 2D output vector containing the expected outputs from the network
Eigen::VectorXd expected_output(2);
input << 1, 1;

//Initiate backpropagation
net.reverse(predictions, expected_output);
```

To evaluate performance while not training:
```
//Make 3D vector [1, 0, 1]
Eigen::VectorXd test_input(3);
test_input << 1, 0, 1;

//The following two method calls are interchangeable.
Eigen::VectorXd test_output = net.forward(test_input, false);
Eigen::VectorXd test_output = net.predict(test_input);
```
With `network.forward(...)` or `network.forward(..., true)`, intermediate outputs from each layer are stored for the reverse process. The above method calls do not store the layer intermediate outputs, saving memory and computation time.

To edit the network after enabling it:
```
//Disables the network, allowing it to be changed
net.disable();

Add layers, remove layers, change weight/bias matrices...

//Re-enables the network for continued training
net.enable();
```


To compile the network, use the provided Makefile if your have GnuMake installed:
```
make c
```
You may need to change the Makefile variables `MAIN`, `OUTPUT_EXECUTABLE_NAME`, and `EIGEN_DIRECTORY_PATH` to where you put your main function, the desired executable filename, and the name of your Eigen 3 folder.

If you don't have GnuMake and you have the G++ compiler installed, this command should compile your executable. Replace "main.cpp" with your main function's filename and "cnet" with your desited executable's name.
Ensure {name of Eigen 3 folder} is replaced with the actual folder's name:
```
g++ main.cpp  -o cnet  -std=c++14  -I {name of Eigen 3 folder}
```

If you don't have G++, refer to your compiler's documentation on how to include external directories and use the C++14 standard (or later).


## File Structure

All source code files are found in the "cnet" directory.

There are 5 main components:
- `activation_function.cpp`- contains the ActivationFunction class and its subclasses. Smart pointers to ActivationFunctions compute activation functions and their derivatives for layers.
- `layer.cpp`- contains the Layer class, a linear layer of a network. The file also contains activation functions and their derivatives.
- `loss_calculator.cpp`- contains the LossCalculator, which computes output losses and loss gradients for model optimization
- `optimizer.cpp`- contains the Optimizer, for improving model weights given losses and a LossCalculator
- `network.cpp`- contains the Network class, which contains all the other components

The top-level directory contains the Makefile and `main.cpp`, which contains functions for testing.