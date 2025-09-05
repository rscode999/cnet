# CNet
**Neural network framework implemented from scratch**

### Run Instructions
You will need Eigen 3 to make the project work.

After pulling this repo, download Eigen 3 by clicking the link [here](https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip). Then, extract the ZIP archive and place the folder inside your cloned repo.

To use in your program, include the "network.cpp" file:
```
#include "network.cpp"
```

To create a network, create a new Network object:
```
Network net = Network();
```

To create a layer without an activation function:
```
//Creates a layer with 3 inputs and 1 output
Layer layer0 = Layer(3, 1);

//Creates a layer with 3 inputs and 1 output, with the identifier "l0"
Layer layer0 = Layer(3, 1, "l0");
```

To create a layer with an activation function:
```
//Creates a layer with 3 inputs and 1 output, with the ReLU activation function and its derivative, the unit step
Layer layer0 = Layer(3, 1, relu, unit_step);
```

To add a layer:
```
net.add_layer(layer0);
```

Layers are added in sequential order. The first layer added will become the input. The last layer added is the output.


To compute forward inputs of the network, first enable training to ensure layer compatibility:
```
net.enable_training()
```

Then, feed an input vector into the network:
```
//Creates the vector [1, 0, 1]
Eigen::VectorXd input(3);
input << 1, 0, 1;

//Store the network's output
Eigen::VectorXd output = net.forward(input);
```

Backpropagation is not yet implemented.


Compile your program with the provided Makefile (given that you have GnuMake installed) with `make c`.  
If not, run the command: `g++ {main program filename}  -o {output executable name}  -std=c++11  -I {path to Eigen 3 folder}` if using the G++ compiler.

### Project Structure

There are 4 main components:
- `layer.cpp`- contains the Layer class, a linear layer of a network. The file also contains activation functions and their derivatives.
- `loss_calculator.cpp`- contains the LossCalculator, which computes output losses and loss gradients for model optimization
- `optimizer.cpp`- contains the Optimizer, for improving model weights given losses and a LossCalculator
- `network.cpp`- contains the Network class, used for overall networks
