
# Class and Method Documentation

Documentation for each class and its methods.

To use in your program, include the "cnet/core.cpp" file. Example: `#include "cnet/core.cpp"`

All classes and methods are in the `CNet` namespace. The only class or method not in `CNet` is the `illegal_state` exception, which is not in a namespace.

Some functions use the `Eigen` linear algebra package. A short guide for Eigen can be found [here](https://libeigen.gitlab.io/eigen/docs-nightly/GettingStarted.html).

<details>
  <summary>Implementation Details</summary>
  
Some documentation entries have an Implementation Details dropdown.
Implementation Details include where a class or method is defined, or information not known without manually inspecting the source code.

</details>
<br>

[Back to README](../README.md)

## Table of Contents


Method categories:
- Constructor: Creates a new instance of the class
- Getter: Retrieves a class attribute
- Setter: Changes the value of a class attribute
- Methods: Any other function that is not a getter or setter
- Operator Overload: Allows usage of operators on the class

Available classes:

- [Illegal State Exception](#illegal_state)

<br>

- [Network](#network)
    - [Constructor](#constructor)
    - [Getters](#getters)
    - [Setters](#setters)
    - [Methods](#methods)
    - [Operator Overloads](#operator-overloads)
- [ActivationFunction (Abstract class)](#activationfunction)
    - [Constructor](#constructor-1)
    - [Methods](#methods-1)
- [Layer](#layer)
    - [Constructors](#constructor-2)
    - [Getters](#getters-1)
    - [Setters](#setters-1)
    - [Methods](#methods-2)
    - [Operator Overloads](#operator-overloads-1)
- [LossCalculator (Abstract class)](#losscalculator)
    - [Constructor](#constructor-3)
    - [Methods](#methods-3)
- [Optimizer (Abstract class)](#optimizer-1)
    - [Detailed optimizer documentation ->](optimizers.md)

<br>

- [Standalone Methods (not in classes)](#standalone-methods)
  - [Loading and Storing Networks](#loading-and-storing-networks)
  - [Creating Network Components by Name](#creating-network-components-by-name)

---
---
---


## illegal_state

Thrown when a CNet object (i.e. a Network) is not in the proper configuration to call a method.

This exception is *not* part of the `CNet` namespace.

`illegal_state` is a subclass of `std::runtime_error`.

Example: `illegal_state` would be thrown if a Network's `forward` method is called while it's disabled, or if `input_dimension` is called when the Network has no layers.

<details>
  <summary>Implementation Details</summary>
  
The class definition for `illegal_state` is inside the file "network.cpp", in the "cnet" directory.

</details>
<br>

---
---

## Network

A neural network that can be trained and used for predictions.

The user adds layers, a loss calculator, and an optimizer to the network prior to use.

Each layer in a network has a 0-based index number, ranging from 0 to `{networkName}.layer_count()` - 1. The first layer in a network has index 0. The second has index 1, the third has index 2, and so on.  
The input layer is the first layer. The output layer is the last layer.

To train and optimize a network, the network must be enabled via `{networkName}.enable()`.
Once enabled, the network cannot be edited until `{networkName}.disable()` is called. Getter methods may still be called while a network is enabled.


---

### Constructor

#### Default Constructor

*Signature*: `Network()`

Creates an empty network.

The created network is not enabled. It has no layers, loss calculator, or optimizer.

---

### Getters

#### biases\_at

*Signature*: `Eigen::VectorXd biases_at(int layer_number) const`

Returns a deep copy of the bias vector in layer `layer_number`.

**Returns**

* `Eigen::VectorXd`: Biases of layer `layer_number`.

**Parameters**

* `layer_number` (`int`): Layer number to access.  Must be on the interval [0, `{networkName}.layer_count()`-1].


---

#### input\_dimension

*Signature*: `int input_dimension() const`

Returns the number of inputs of this network.

**Returns**

* `int`: Number of inputs.

**Exceptions**

* `illegal_state`: If the network has no layers.

---

#### is\_enabled

*Signature*: `bool is_enabled() const`

Returns whether the network is enabled.

The network must be enabled to train and evaluate with it.

**Returns**

* `bool`: `true` if the network is enabled, `false` otherwise.

---

#### layer\_at (by index)

*Signature*: `Layer layer_at(int layer_number) const`

Returns a deep copy of the layer at `layer_number`.

**Returns**

* `Layer`: Deep copy of the layer.

**Parameters**

* `layer_number` (`int`): Layer number (0-based indexing). Must be on the interval [0, `{networkName}.layer_count()`-1].

---

#### layer\_at (by name)

*Signature*: `Layer layer_at(const std::string& layer_name) const`

Returns the layer whose name is `layer_name`.

The first matching layer name, i.e. the layer with the lowest index number, is returned.

**Returns**

* `Layer`: Matching layer.

**Parameters**

* `layer_name` (`std::string`): Name of layer to find. Must match an existing layer.

**Exceptions**

* `out_of_range`: If no matching layer is found.

---

#### layer\_count

*Signature*: `int layer_count() const`

Returns the number of layers in the network.

**Returns**

* `int`: Number of layers.

---

#### loss\_calculator

*Signature:* `std::shared_ptr<LossCalculator> loss_calculator() const`

Returns a smart pointer to the network's loss calculator.

**Returns**

* `std::shared_ptr<LossCalculator>`: Network's loss calculator.

---

#### optimizer

*Signature*: `std::shared_ptr<Optimizer> optimizer() const`

Returns a `std::shared_ptr` to the network's optimizer.

The smart pointer can be used to directly change the network's hyperparameters.

**Returns:**

* `std::shared_ptr<Optimizer>`: Network's optimizer, as a smart pointer.

---

#### optimizer_hyperparameters

*Signature:* `std::vector<double> optimizer_hyperparameters() const`

Returns a std::vector containing the optimizer's hyperparameters, in the order required by the current optimizer's `set_optimizer_hyperparameters` method.

Example: If the Network uses a SGD optimizer, the output has 3 indices. Index 0 contains the learning rate, index 1 has the momemtum coefficient, and index 2 has the batch size (as a double).

**Returns:**

* `std::vector<double>`: Vector containing the optimizer hyperparameters.

---

#### output\_dimension

*Signature*: `int output_dimension() const`

Returns the number of outputs of this network.

**Returns**

* `int`: Number of outputs.

**Exceptions**

* `illegal_state`: If the network has fewer than 1 layer.

---

#### weights\_at

*Signature*: `Eigen::MatrixXd weights_at(int layer_number) const`

Returns a deep copy of the weight matrix in layer `layer_number`.

**Returns**

* `Eigen::MatrixXd`: Weight matrix at the specified layer.

**Parameters**

* `layer_number` (`int`): Index of layer. Must be on the interval [0, `{networkName}.layer_count()`-1].

---

### Setters

Because they mutate the network's architecture, most setters must be called when the network is disabled.

<details>
  <summary>Implementation Details</summary>
  
Setters that add or remove layers that are not the final layer run in linear time. That is, the runtime of these methods scales linearly with the number of layers in the network.

</details>

#### add\_layer (Layer)

*Signature*: `void add_layer(Layer new_layer)`

Adds `new_layer` to the back of the network.

To use this method, the network must be disabled.

**Parameters**

* `new_layer` (`Layer`): Layer to add.

**Exceptions**

* `illegal_state`: If the network is enabled.

---

#### add\_layer (dimensions + name)

*Signature*: `void add_layer(int input_dimension, int output_dimension, std::string name = "layer")`

Adds a new layer with the given dimensions and name. The new layer has no activation function.

To use this method, the network must be disabled.


<details>
  <summary>Implementation Details</summary>
  
The new layer's activation function is the identity function, f(x)=x, a placeholder that does nothing.

</details>
<br>


**Parameters**

* `input_dimension` (`int`): Number of input nodes. Must be positive.
* `output_dimension` (`int`): Number of output nodes. Must be positive.
* `name` (`std::string`): Layer name. Default: `"layer"`

**Exceptions**

* `illegal_state`: If the network is enabled.

---

#### add\_layer (dimensions + activation + name)

*Signature*: `void add_layer(int input_dimension, int output_dimension, std::shared_ptr<ActivationFunction> activation_function, std::string name = "layer")`

Adds a new layer with specified activation function and name.

To use this method, the network must be disabled.

**Parameters**

* `input_dimension` (`int`): Number of input nodes. Must be positive.
* `output_dimension` (`int`): Number of output nodes. Must be positive.
* `activation_function` (`std::shared_ptr<ActivationFunction>`): Pointer to activation function.
* `name` (`std::string`): Layer name. Default: `"layer"`

**Exceptions**

* `illegal_state`: If the network is enabled.

---

#### disable

*Signature*: `void disable()`

Disables the network, allowing the network to be modified.

---

#### enable

*Signature*: `void enable()`

Enables the network for training and evaluation.

Performs checks on configuration, dimensions, and activations.

To successfully enable the network, the following conditions must be met:

* The network has at least 1 layer.
* A loss calculator and an optimizer are defined.
* The output dimension of each layer must equal the input dimension of the next layer.
* The only layer with Softmax activation is the final (output) layer.

If a condition is broken, an `illegal_state` exception is thrown, with an error message detailing the failed check.

If all conditions are met, the network resets all internal state previously used in training.

**Exceptions**

* `illegal_state`: If checks on layer count, optimizer, loss calculator, layer compatibility, or softmax placement fail.

---

#### insert\_layer\_at

*Signature*: `void insert_layer_at(int new_pos, Layer new_layer)`

Inserts `new_layer` at position `new_pos`.

The current layer at `new_pos`, and all layers after that layer, have their positions increased by 1. The new layer will have a position number of `new_pos`.

To use this method, the network must be disabled.

**Parameters**

* `new_pos` (`int`): Insertion index. Must be on the interval [0, `{networkName}.layer_count()`] (inclusive on both sides).
* `new_layer` (`Layer`): Layer to insert.

**Exceptions**

* `illegal_state`: If the network is enabled.

---

#### remove\_layer

*Signature*: `void remove_layer(std::string removal_name)`

Removes a layer with the name `removal_name`.

The first layer with a matching name, i.e. the one with the lowest index number, is removed.
If no matching layer names are found, throws `out_of_range`.

To use this method, the network must be disabled.

**Parameters**

* `removal_name` (`std::string`): Name of the layer to remove.

**Exceptions**

* `illegal_state`: If the network is enabled.
* `out_of_range`: If no matching layer is found.

---

#### remove\_layer\_at

*Signature*: `void remove_layer_at(int remove_pos)`

Removes the layer at position `remove_pos`.

To use this method, the network must be disabled.

**Parameters**

* `remove_pos` (`int`): Index of the layer. Must be on the interval [0, `{networkName}.layer_count()`-1].

**Exceptions**

* `illegal_state`: If the network is enabled.

---

#### rename\_layer\_at

*Signature*: `void rename_layer_at(int rename_pos, std::string new_name)`

Renames the layer at `rename_pos`.

Unlike most setters, this method can be called when the network is enabled.

**Parameters**

* `rename_pos` (`int`): Layer index to rename. Must be on the interval [0, `{networkName}.layer_count()`-1].
* `new_name` (`std::string`): New name for the chosen layer

---

#### set\_activation\_function\_at

*Signature*: `void set_activation_function_at(int layer_number, std::shared_ptr<ActivationFunction> new_activation_function)`

Sets the activation function for a given layer.

To remove a layer's activation function, set the activation to a `std::shared_ptr<IdentityActivation>`. The `IdentityActivation` applies the identity function f(x)=x to its inputs, effectively doing nothing.

The network must be disabled to use this method.

**Parameters**

* `layer_number` (`int`): Index of the layer to change. Must be on the interval [0, `{networkName}.layer_count()`-1].
* `new_activation_function` (`std::shared_ptr<ActivationFunction>`): Smart pointer to the activation function object.

**Exceptions**

* `illegal_state`: If the network is enabled.

---

#### set\_biases\_at

*Signature*: `void set_biases_at(int layer_number, Eigen::VectorXd new_biases)`

Sets the biases of the specified layer.

To use this method, the network must be disabled.

**Parameters**

* `layer_number` (`int`): Index of the layer. Must be on the interval [0, `{networkName}.layer_count()`-1].
* `new_biases` (`Eigen::VectorXd`): New bias vector. Must be a column vector with `{selected layer}.output_dimension()` elements (dimension depends on the selected layer)

**Exceptions**

* `illegal_state`: If the network is enabled.

---

#### set\_loss\_calculator

*Signature*: `void set_loss_calculator(std::shared_ptr<LossCalculator> new_calculator)`

Sets the network's loss calculator to `new_calculator`, replacing any existing loss calculator.

To use this method, the network must be disabled.

**Parameters**

* `new_calculator` (`std::shared_ptr<LossCalculator>`): Smart pointer to new loss calculator.

**Exceptions**

* `illegal_state`: If the network is enabled.

---


#### set\_optimizer

*Signature*: `void set_optimizer(std::shared_ptr<Optimizer> new_optimizer)`

Sets the network's optimizer to `new_optimizer`, replacing any existing optimizer.

To use this method, the network must be disabled.

**Parameters**

* `new_optimizer` (`std::shared_ptr<Optimizer>`): Smart pointer to new optimizer.

**Exceptions**

* `illegal_state`: If the network is enabled.

---

#### set\_optimizer\_hyperparameters
*Signature*: `void set_optimizer_hyperparameters(const std::vector<double>& hyperparameters)`

Sets the optimizer's hyperparameters.
The purpose of each index in `hyperparameters` depends on the optimizer.

Throws `illegal_state` if the network has no defined optimizer.

Example: For SGD optimizers, index 0 is the new learning rate, index 1 is for the new momentum coefficient, and index 2 is the new batch size.

For more information about particular optimizers, including their preconditions, see the [optimizer-specific documentation](optimizers.md).

**Parameters**

* `hyperparameters` (`const std::vector<double>&`): Vector of new hyperparameters to set. Preconditions vary, depending on the specific optimizer used.

**Exceptions**

* `illegal_state`: If the network has no defined optimizer.

---

#### set\_weights\_at

*Signature*: `void set_weights_at(int layer_number, Eigen::MatrixXd new_weights)`

Sets the weight matrix of the specified layer.

To use this method, the network must be disabled.

**Parameters**

* `layer_number` (`int`): Layer index. Must be on the interval [0, `{networkName}.layer_count()`-1].
* `new_weights` (`Eigen::MatrixXd`): New weight matrix. Must have `{selected layer}.output_dimension()` rows and `{selected layer}.input_dimension()` columns.

**Exceptions**

* `illegal_state`: If the network is enabled.

---

### Methods

#### forward

*Signature*: `Eigen::VectorXd forward(const Eigen::VectorXd& input, bool training = true)`

Returns the output of the network for a given input.

If `training` is true, intermediate outputs are stored for backpropagation, allowing `{networkName}.reverse` to be called. Otherwise, the network does not store intermediate outputs, saving memory and computation time.

**Returns**

* `Eigen::VectorXd`: Network's predictions for the given input

**Parameters**

* `input` (`const Eigen::VectorXd&`): Input vector to the network.
* `training` (`bool`): Whether the network is in training mode. Default: `true`.

**Exceptions**

* `illegal_state`: If the network is disabled.

---

#### predict

*Signature*: `Eigen::VectorXd predict(const Eigen::VectorXd& input)`

Returns the predictions for the given input.

When this method is used, the network *does not* internally record intermediate layer outputs for backpropagation, saving memory and computation time.

Equivalent to `{networkName}.forward(input, false)`.

Requires that the network is enabled.

**Returns**

* `Eigen::VectorXd`: Network's predictions for the given input

**Parameters**

* `input` (`const Eigen::VectorXd&`): Input vector to make predictions with.

**Exceptions**

* `illegal_state`: If the network is disabled.

---

#### reverse

*Signature*: `void reverse(const Eigen::VectorXd& predictions, const Eigen::VectorXd& actuals)`

Updates the weights and biases of this network using `predictions` and `actuals`, using the network's optimizer.

Recommendation: Call this method immediately after using the `{networkName}.forward` method, using the network's output as `predictions` and the expected output as `actuals`.  
That way, the stored gradients from `forward` (stored inside the network) will allow `reverse` to update the network gradients properly.

This method requires the network to be enabled. Also, `{networkName}.forward` with `training = true` must have been called since the network was enabled.
If these conditions are not met, the method throws `illegal_state`.

**Parameters**

* `predictions` (`const Eigen::VectorXd&`): Predicted outputs from the network. Must have `{networkName}.output_dimension()` elements.
* `actuals` (`const Eigen::VectorXd&`): Expected outputs that the network should have produced. Must have `{networkName}.output_dimension()` elements.

**Exceptions**

* `illegal_state`: If the network is disabled, or a training forward pass was not completed.

---

### Operator Overloads

#### add-assign (`+=`)

*Signature*: `void operator+=(Layer new_layer)`

Adds `new_layer` as the final layer of the network.

If the network is enabled, this method throws `illegal_state`.

Equivalent to `{networkName}.add_layer(new_layer)`.

Usage Example
```
CNet::Network net;
CNet::Layer new_layer = Layer(3, 5);
net += new_layer;
```

**Parameters**

* `new_layer` (`Layer`): Layer to add to the back of the network.

**Exceptions**

* `illegal_state`: If the network is enabled.

---

#### output stream insertion (`<<`)

*Signature*: `friend std::ostream& operator<<(std::ostream& output_stream, const Network& network)`

Exports `network` to the output stream `output_stream`, returning a reference to `output_stream` with `network` added.

The exported network, as a `std::string`, contains the Network's enabled/disabled status, loss calculator, optimizer, and layers.

Usage Example
```
CNet::Network net;
std::cout << net;
```

**Returns**

* `std::ostream` (`std::ostream`): New output stream containing the network's information.

**Parameters**

* `output_stream` (`std::ostream&`): Output stream to export to.
* `network` (`const Network&`): Network to export.


---
---


## ActivationFunction

Abstract class representing an activation function, along with all functions required to perform backpropagation.

Commonly used when pointed to by a `std::shared_ptr` smart pointer.  
Example of creating a `Relu` object: `std::shared_ptr<Relu> relu = std::make_shared<Relu>();`

Pre-implemented concrete subclasses:
- `IdentityActivation` (name: "none"), a placeholder that does nothing to its input
- `Relu` (name: "relu"), the Rectified Linear Unit (ReLU) function
- `Sigmoid` (name: "sigmoid")
- `Softmax` (name: "softmax"). The only layer that can use `Softmax` activation is the final layer.

Names are accessed using the `name` method.

<br>
<details>
  <summary>Notice to Implementers</summary>
  
The file "activation_function.cpp" contains the method `make_activation_function`. If creating a new activation function, add the function's name to `make_activation_function`. If not, the function can't be stored and loaded to external files.

</details>
<br>

---

### Constructor

#### Default constructor

All pre-implemented subclasses of `ActivationFunction` have a constructor that takes no arguments. Examples: `Relu()`, `Sigmoid()`

ActivationFunctions are commonly used when pointed to by a smart pointer, particularly a `std::shared_ptr`.  
Example:
```std::shared_ptr<Relu> relu_activation = std::make_shared<Relu>();```

---

### Methods

#### compute

*Signature:* `virtual Eigen::VectorXd compute(const Eigen::VectorXd& input) const`

Applies the activation function element-wise to the input.

**Returns**

* `Eigen::VectorXd`: `input` after applying this activation function element-wise.

**Parameters**

* `input` (`const Eigen::VectorXd&`): Values to calculate activations.

---

#### compute\_derivative

*Signature:* `virtual Eigen::VectorXd compute_derivative(const Eigen::VectorXd& input) const`

Applies the derivative of the activation function element-wise to the input.

**Returns**

* `Eigen::VectorXd`: Activation function's derivative applied element-wise to `input`.

**Parameters**

* `input` (`const Eigen::VectorXd&`): Value to calculate.

---

#### name

*Signature:* `virtual std::string name() const`

Returns the unique identifier for the activation function.

`IdentityActivation`'s name is "none".

A function object's name is usually the class's name converted to lowercase, with underscores separating each word. Example: `Relu`'s name is "relu".

<details>
<summary>Implementation Details</summary>

If this method is not overridden, this method returns "none", as with the `IdentityActivation`.

</details>
<br>

**Returns**

* `std::string`: Name of the activation function.


---

#### using\_pre\_activation

*Signature:* `virtual bool using_pre_activation() const`

Indicates whether the activation function should be applied on pre-activation inputs.

<details>
<summary>Implementation Details</summary>

If not overridden, this method returns `false`.

</details>
<br>

**Returns**

* `bool`: `true` if pre-activation input is used, `false` otherwise.




---
---


## Layer

A linear layer in a network.

There is no distinction between an input, hidden, or output layer. A layer's role depends on its position in the network. The first layer is the input. The last layer is the output.

Each layer has a weight matrix and separate bias vector.
They can be manually viewed or updated using getter and setter methods.  
Internally, all layers have activation functions. If an activation is not assigned, the layer's activation is the `IdentityActivation`, a placeholder that does nothing.

---

### Constructor

#### Without activation function

*Signature:* `Layer(int input_dimension, int output_dimension, std::string name = "layer")`

Creates a new Layer and initializes all fields. The activation function is set to the identity activation function, f(x)=x, which does nothing.

All weights and biases are randomly initialized on the uniform interval [-1, 1].

**Parameters**

* `input_dimension` (`int`): Number of inputs that the Layer takes in. Must be positive.
* `output_dimension` (`int`): Number of outputs that the Layer gives. Must be positive.
* `name` (`std::string`): Identifier for this Layer. Default: `"layer"`

---

#### With activation function

*Signature:* `Layer(int input_dimension, int output_dimension, std::shared_ptr<ActivationFunction> activation_function, std::string name = "layer")`

Creates a new Layer and loads it with the provided fields. All elements in the Layer's weights and biases are randomly initialized in the uniform range [-1, 1].

**Parameters**

* `input_dimension` (`int`): Number of inputs that the Layer takes in. Must be positive.
* `output_dimension` (`int`): Number of outputs that the Layer gives. Must be positive.
* `activation_function` (`std::shared_ptr<ActivationFunction>`): Smart pointer to activation function object to use.
* `name` (`std::string`): Identifier for this Layer. Default: `"layer"`

---

### Getters

#### activation\_function

*Signature:* `const std::shared_ptr<ActivationFunction> activation_function() const`

Returns a smart pointer to the layer's activation function.

**Returns**

* `std::shared_ptr<ActivationFunction>`: Smart pointer to the layer's activation function.

---

#### bias\_vector

*Signature:* `Eigen::VectorXd bias_vector() const`

Returns the layer's bias vector, as a `Eigen::VectorXd`.

**Returns**

* `Eigen::VectorXd`: Bias vector, containing `{layerName}.output_dimension()` elements.

---

#### input\_dimension

*Signature:* `int input_dimension() const`

Returns the number of inputs for the layer.

**Returns**

* `int`: Number of input elements.

---

#### name

*Signature:* `std::string name() const`

Returns the display name of the layer.

The name returned is *not* the layer variable's name. The returned name is the one given in the layer's constructor.

Example
```
CNet::Layer my_layer = Layer(1, 2, "layer name");
std::cout << my_layer.name(); //Prints "layer name"
```

**Returns**

* `std::string`: Layer name.

---

#### output\_dimension

*Signature:* `int output_dimension() const`

Returns the number of outputs for the layer.

**Returns**

* `int`: Number of output elements.

---

#### weight\_matrix

*Signature:* `Eigen::MatrixXd weight_matrix() const`

Returns the layer's weight matrix.

**Returns**

* `Eigen::MatrixXd`: Weight matrix, with `{layerName}.output_dimension()` rows and `{layerName}.input_dimension()` columns.

---

### Setters

#### set\_activation\_function

*Signature:* `void set_activation_function(std::shared_ptr<ActivationFunction> new_activation_function)`

Sets the layer's activation function to `new_activation_function`.

To remove a layer's activation function, set the function to a `std::shared_ptr<IdentityActivation>`. The IdentityActivation applies the identity function f(x)=x to its inputs, effectively doing nothing.

**Parameters**

* `new_activation_function` (`std::shared_ptr<ActivationFunction>`): Smart pointer to new activation function.

---

#### set\_bias\_vector

*Signature:* `void set_bias_vector(Eigen::VectorXd new_biases)`

Sets the layer's bias vector.

**Parameters**

* `new_biases` (`Eigen::VectorXd`): New vector of biases. Must have `{layerName}.output_dimension()` elements.

---

#### set\_name

*Signature:* `void set_name(std::string new_name)`

Sets the display name of the layer.

Does not change the layer's variable name.

**Parameters**

* `new_name` (`std::string`): New name for the layer.

---

#### set\_weight\_matrix

*Signature:* `void set_weight_matrix(Eigen::MatrixXd new_weights)`

Sets the layer's weight matrix.

**Parameters**

* `new_weights` (`Eigen::MatrixXd`): New matrix of weights. Must have `{layer}.output_dimension()` rows and `{layer}.input_dimension()` columns.

---

### Methods

#### forward

*Signature:* `Eigen::VectorXd forward(const Eigen::VectorXd& input)`

Returns the result of the linear forward operation for the given input. Does not perform any other operations.

Applies weights and adds biases.

**Important note**: This method does *not* apply the layer's activation function.

<details>
<summary>Implementation Details</summary>

The `Network` (as opposed to each individual `Layer`) applies activations, so the `Network` can store pre- and post-activation layer outputs for optimization.

</details>
<br>

**Returns**

* `Eigen::VectorXd`: Resulting vector of length `{layerName}.output_dimension()`.

**Parameters**

* `input` (`const Eigen::VectorXd&`): Input column vector. Must have `{layerName}.input_dimension()` elements.

---

### Operator Overloads

#### output stream insertion (`<<`)

*Signature:* `friend std::ostream& operator<<(std::ostream& output_stream, const Layer& layer)`

Exports `layer` to the output stream `output_stream`, returning a new output stream with `layer` inside.

Information contained in the new output stream is the layer's name, its input and output dimensions in the format "(`input dimension`, `output dimension`)", and its activation function.

Usage Example
```
CNet::Layer layer = Layer(3, 5);
std::cout << layer;
```

**Returns**

* `std::ostream&`: Output stream with `layer` added.

**Parameters**

* `output_stream` (`std::ostream&`): Output stream to write to.
* `layer` (`Layer`): Layer to export.




---
---


## LossCalculator

Abstract class for calculating loss (error between actual and expected output).

A `std::shared_ptr` smart pointer to a `LossCalculator` can be used by a `Network`.

Pre-implemented concrete subclasses:

* `CrossEntropy`
* `MeanSquaredError`

<br>
<details>
  <summary>Notice to Implementers</summary>
  
The file "loss_calculator.cpp" contains the method `make_loss_calculator`. If creating a new loss calculator, add the calculator's name to `make_loss_calculator`. If not, the calculator can't be stored and loaded to external files.

</details>
<br>

---

### Constructor

#### Default constructor

*Signature:* `{LossCalculator subclass name}()`

All pre-implemented subclasses of `LossCalculator` have a constructor that takes no arguments.

Example: `MeanSquaredError()`

---

### Methods

#### compute\_loss

*Signature:* `virtual double compute_loss(const Eigen::VectorXd& predictions, const Eigen::VectorXd& actuals)`

Returns the loss (error) when measured between `predictions` and `actuals`.

**Returns**

* `double`: Calculator's loss of the model predictions.

**Parameters**

* `predictions` (`const Eigen::VectorXd&`): Model's predictions for a given input
* `actuals` (`const Eigen::VectorXd&`): True values for model predictions. Must have the same number of rows as `predictions`.

---

#### compute\_loss\_gradient

*Signature:* `virtual Eigen::VectorXd compute_loss_gradient(const Eigen::VectorXd& predictions, const Eigen::VectorXd& actuals)`

Returns the gradient of the losses when measured between `predictions` and `actuals`.

**Returns**

* `Eigen::VectorXd`: Calculator's loss gradient of the model predictions.

**Parameters**

* `predictions` (`const Eigen::VectorXd&`): Model's predictions for a given input
* `actuals` (`const Eigen::VectorXd&`): True values for model predictions. Must have the same number of rows as `predictions`.

---

#### name

*Signature:* `virtual std::string name()`

Returns the identifying string of the loss calculator.

<details>
<summary>Implementation Details</summary>

This method is purely virtual. Subclasses must override this method.

</details>
<br>

**Returns**

* `std::string`: Name of the loss calculator (typically, the lowercased class name with underscores separating words, e.g. `"cross_entropy"`).


---
---


## Optimizer

Abstract class for network optimizers.

A `shared_ptr` smart pointer to an `Optimizer` instance can be used by a `Network`.  
Example for SGD optimizer: `std::shared_ptr<SGD> optimizer = make_shared<SGD>();`

Pre-implemented concrete subclasses:

* `SGD`, a Stochastic Gradient Descent optimizer

Further info is in the [optimizer-specific documentation](optimizers.md).

<br>
<details>
  <summary>Notice to Implementers</summary>
  
The file "optimizer.cpp" contains the method `make_optimizer`. If creating a new optimizer, add the optimizer's name to `make_optimizer`. If not, the optimizer can't be stored and loaded to external files.

</details>
<br>

---
---

## Standalone Methods

Methods that are *not* part of a class. 

Includes functions to load and store network configurations to an external file, and functions to create network components by name.

---

### Loading and Storing Networks

Used to store a network's state to an external file.

Loading and storing methods can be used by including "cnet/core.cpp" (they can't be used if "cnet/network.cpp" is included):
```
#include "cnet/core.cpp"
```

Example usage, to save a network to "configuration.txt", then load the saved network:
```
CNet::Network net1;

Add layers, loss calculator, optimizer to `net1`...

//Saves the configuration of `net1` to the file "configuration.txt"
store_network_config("configuration.txt", net1); 

//Loads the configuration stored in "configuration.txt" to `net2`
CNet::Network net2 = load_network_config("configuration.txt");
```

---

#### load\_network\_config

*Signature:* `CNet::Network load_network_config(const std::string& config_filepath)`

 Returns the network whose configuration is specified in the file at `config_filepath` (including the file extension).

The network that is outputted from this method is disabled.

Ensure that the chosen configuration file is in the same format as that produced by the `store_network_config` method.  
If not, this method will not work: the malformed file may trigger an assertion or cause a segfault.

**Returns**

* `CNet::Network`: Network containing the configuration at the specified filepath.

**Parameters**

* `config_filepath` (`std::string`): filepath to load configuration from. The file must be in the format produced by `store_network_config`.

**Exceptions**

* `runtime_error`: If the file at `config_filepath` does not exist, or cannot be opened. There is no check for a file in the incorrect format.

---

#### store\_network\_config

*Signature:* `void store_network_config(const std::string& config_filepath, const CNet::Network& network)`

Stores `network` into the configuration file at the path `config_filepath` (including the file extension).

All leading and trailing whitespace will be removed from layer names.  
The chosen file will be overwritten.

Note: All layers in the network must have names that have no whitespace.
The default layer name is "layer", not the empty string.

**Parameters**

* `config_filepath` (`std::string`): filepath to store configuration at
* `network` (`CNet::Network`): Network object to store at the specified filepath. All layers in the network must have at least 1 non-whitespace character in their names.

**Exceptions**

* `runtime_error`: If the file at `config_filepath` does not exist, or cannot be opened.

---

### Creating Network Components by Name

Factory methods to create components using their names.

All methods return smart pointers to the components, as a `std::shared_ptr`.

Usage example
```
std::shared_ptr<ActivationFunction> relu_activ_ptr = make_activation_function("relu");

std::shared_ptr<Optimizer> sgd_ptr = make_optimizer("sgd", vector<double>{0.005, 0.8, 1});
```

---

#### make\_activation\_function

*Signature:* `std::shared_ptr<ActivationFunction> make_activation_function(const std::string& name)`

Returns a std::shared_ptr to the activation function whose name is `name` (example: "none", "relu", "sigmoid").

This function creates a new pointer.

**Returns**

* `std::shared_ptr`: Newly created smart pointer to the activation function given by `name`

**Parameters**

* `name` (`std::string`): Name of the desired activation function

**Exceptions**

* `runtime_error`: If `name` is not a recognized activation function name.

<br>
<details>
  <summary>Implementation Details</summary>
  
This method is defined in the file "activation_function.cpp".

</details>
<br>

---

#### make\_loss\_calculator

*Signature:* `std::shared_ptr<LossCalculator> make_loss_calculator(const std::string& name)`

Returns a std::shared_ptr to the loss calculator whose name is `name` (example: "mean_squared_error").

This function creates a new pointer.

**Returns**

* `std::shared_ptr`: Newly created smart pointer to the loss calculator given by `name`

**Parameters**

* `name` (`std::string`): Name of the desired loss calculator

**Exceptions**

* `runtime_error`: If `name` is not a recognized loss calculator name.

<br>
<details>
  <summary>Implementation Details</summary>
  
This method is defined in the file "loss_calculator.cpp".

</details>
<br>

---

#### make\_optimizer

*Signature:* `std::shared_ptr<Optimizer> make_optimizer(const std::string& name, const std::vector<double>& hyperparameters)`

Returns a std::shared_ptr to an optimizer whose name is `name` (example: "sgd") and specified by the hyperparameters `hyperparameters`.

This function creates a new pointer.

**Returns**

* `std::shared_ptr`: Newly created smart pointer to the optimizer specified by `name` and `hyperparameters`

**Parameters**

* `name` (`std::string`): Name of the desired loss calculator
* `hyperparameters` (`std::vector<double>`): Hyperparameters to initialize the new optimizer. Must be accepted by the optimizer's `set_hyperparameters` method

**Exceptions**

* `runtime_error`: If `name` is not a recognized optimizer name.

<br>
<details>
  <summary>Implementation Details</summary>
  
This method is defined in the file "optimizer.cpp".

</details>
<br>

---
---
---

[Back to table of contents](#table-of-contents)