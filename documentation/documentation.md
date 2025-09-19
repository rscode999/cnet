
# Class and Method Documentation

Documentation for each class and its methods.

Some functions use the `Eigen` linear algebra package. A short guide for Eigen can be found [here](https://libeigen.gitlab.io/eigen/docs-nightly/GettingStarted.html).  

[Back to README](../README.md)

## Table of Contents


Method categories:
- Constructor: Creates a new instance of the class
- Getter: Retrieves a class attribute
- Setter: Changes the value of a class attribute
- Methods: Any other function that is not a getter or setter
- Operator Overload: Allows usage of operators on the class

Available classes:

- [Network](#network)
    - [Illegal State Exception](#illegal_state)
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
- [Optimizer (Abstract class)](#optimizer)
    - [Detailed optimizer documentation ->](optimizers.md)

---
---
---


## Network

A neural network that can be trained and used for predictions.

The user adds layers, a loss calculator, and an optimizer to the network prior to use.

Each layer in a network has an index number, ranging from 0 to `{networkName}.layer_count()` - 1. The first layer in a network has index 0. The second has index 1, the third has index 2, and so on.  
The input layer is the first layer. The output layer is the last layer.

To use a network, the network must be enabled via `{networkName}.enable()`.
Once enabled, the network cannot be edited until `{networkName}.disable()` is called. Getter methods may still be called while a network is enabled.


---

### illegal_state

An exception thrown only by a Network. Thrown when enable/disable rules are broken.

`illegal_state` is a subclass of `std::runtime_error`.

Example: `illegal_state` would be thrown if a Network's `forward` method is called while it's disabled.

---

### Constructor

#### Default Constructor

*Signature*: `Network()`

Creates an empty network.

The created network is not enabled. It has no layers, loss calculator, or optimizer.

---

### Getters

#### biases\_at

*Signature*: `Eigen::VectorXd biases_at(int layer_number)`

Returns a deep copy of the bias vector in layer `layer_number`.

**Returns**

* `Eigen::VectorXd`: Biases of layer `layer_number`.

**Parameters**

* `layer_number` (`int`): Layer number to access.

**Exceptions**

* `out_of_range`: If `layer_number` is not on the interval [0, `{networkName}.layer_count()`-1].

---

#### input\_dimension

*Signature*: `int input_dimension()`

Returns the number of inputs of this network.

**Returns**

* `int`: Number of inputs.

**Exceptions**

* `illegal_state`: If the network has fewer than 1 layer.

---

#### is\_enabled

*Signature*: `bool is_enabled()`

Returns whether the network is enabled.

The network must be enabled to train and evaluate with it.

**Returns**

* `bool`: `true` if the network is enabled, `false` otherwise.

---

#### layer\_at (by index)

*Signature*: `Layer layer_at(int layer_number)`

Returns a deep copy of the layer at `layer_number`.

**Returns**

* `Layer`: Deep copy of the layer.

**Parameters**

* `layer_number` (`int`): Layer number (0-based indexing). Must be on the interval [0, `{networkName}.layer_count()`-1].

---

#### layer\_at (by name)

*Signature*: `Layer layer_at(std::string layer_name)`

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

#### output\_dimension

*Signature*: `int output_dimension()`

Returns the number of outputs of this network.

**Returns**

* `int`: Number of outputs.

**Exceptions**

* `illegal_state`: If the network has fewer than 1 layer.

---

#### weights\_at

*Signature*: `Eigen::MatrixXd weights_at(int layer_number)`

Returns a deep copy of the weight matrix in layer `layer_number`.

**Returns**

* `Eigen::MatrixXd`: Weight matrix at the specified layer.

**Parameters**

* `layer_number` (`int`): Index of layer. Must be on the interval [0, `{networkName}.layer_count()`-1].

---

### Setters

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

Adds a new layer with the given dimensions and name. The new layer has no activation function.*

To use this method, the network must be disabled.

*The activation function is the identity function, f(x)=x, which does nothing.

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

If all conditions are met, to handle changes in network architecture, 
the network resets all internal state previously used in training.

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
* `new_name` (`std::string`): New name.

---

#### set\_activation\_function\_at

*Signature*: `void set_activation_function_at(int layer_number, std::shared_ptr<ActivationFunction> new_activation_function)`

Sets the activation function for a given layer.

To remove a layer's activation function, set the function to a `shared_ptr<IdentityActivation>`. The `IdentityActivation` applies the identity function f(x)=x to its inputs, effictively doing nothing.

The network must be disabled to use this method.

**Parameters**

* `layer_number` (`int`): Index of the layer. Must be on the interval [0, `{networkName}.layer_count()`-1].
* `new_activation_function` (`std::shared_ptr<ActivationFunction>`): Smart pointer to the function.

**Exceptions**

* `illegal_state`: If the network is enabled.

---

#### set\_biases\_at

*Signature*: `void set_biases_at(int layer_number, Eigen::VectorXd new_biases)`

Sets the biases of the specified layer.

To use this method, the network must be disabled.

**Parameters**

* `layer_number` (`int`): Index of the layer. Must be on the interval [0, `{networkName}.layer_count()`-1].
* `new_biases` (`Eigen::VectorXd`): New bias vector (column vector). Must contain `{networkName}.output_dimension()` elements

**Exceptions**

* `illegal_state`: If the network is enabled.

---

#### set\_loss\_calculator

*Signature*: `void set_loss_calculator(std::shared_ptr<LossCalculator> new_calculator)`

Replaces the current loss calculator.

To use this method, the network must be disabled.

**Parameters**

* `new_calculator` (`std::shared_ptr<LossCalculator>`): Smart pointer to new loss calculator.

**Exceptions**

* `illegal_state`: If the network is enabled.

---


#### set\_optimizer

*Signature*: `void set_optimizer(std::shared_ptr<Optimizer> new_optimizer)`

Replaces the current optimizer.

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

Example: For SGD optimizers, index 0 is the new learning rate, and index 1 is for the new momentum coefficient.

See the [optimizer-specific documentation](optimizers.md) for more information about particular optimizers.

**Parameters**

* `hyperparameters` (`const std::vector<double>&`): Vector of new hyperparameters to set. Preconditions vary, depending on the specific optimizer used.

---

#### set\_weights\_at

*Signature*: `void set_weights_at(int layer_number, Eigen::MatrixXd new_weights)`

Sets the weight matrix of the specified layer.

To use this method, the network must be disabled.

**Parameters**

* `layer_number` (`int`): Layer index. Must be on the interval [0, `{networkName}.layer_count()`-1].
* `new_weights` (`Eigen::MatrixXd`): New weight matrix. Must have `{selectedLayer}.output_dimension()` rows and `{selectedLayer}.input_dimension()` columns.

**Exceptions**

* `illegal_state`: If the network is enabled.

---

### Methods

#### forward

*Signature*: `Eigen::VectorXd forward(const Eigen::VectorXd& input, bool training = true)`

Returns the output of the network for a given input.

If `training` is true, intermediate outputs are stored for backpropagation, allowing `{networkName}.reverse` to be called.

**Returns**

* `Eigen::VectorXd`: Network's output.

**Parameters**

* `input` (`const Eigen::VectorXd&`): Input vector.
* `training` (`bool`): Whether the network is in training mode. Default: `true`.

---

#### predict

*Signature*: `Eigen::VectorXd predict(const Eigen::VectorXd& input)`

Returns the predictions for the given input.

When this method is used, the network *does not* internally record intermediate layer outputs for backpropagation.

Equivalent to `{networkName}.forward(input, false)`.

Requires that the network is enabled.

**Returns**

* `Eigen::VectorXd`: Network's output.

**Parameters**

* `input` (`const Eigen::VectorXd&`): Input vector to make predictions with.

---

#### reverse

*Signature*: `void reverse(const Eigen::VectorXd& predictions, const Eigen::VectorXd& actuals)`

Updates the weights and biases of this network using `predictions` and `actuals`, using the network's optimizer.

This method requires the network to be enabled. Also, `{networkName}.forward` with `training = true` must have been called since the network was enabled.
If these conditions are not met, the method throws `illegal_state`.

**Parameters**

* `predictions` (`const Eigen::VectorXd&`): Predicted outputs from the network. Must have `{networkName}.output_dimension()` elements.
* `actuals` (`const Eigen::VectorXd&`): Expected outputs that the network should have produced. Must have `{networkName}.output_dimension()` elements.

**Exceptions**

* `illegal_state`: If the network is disabled.

---

### Operator Overloads

#### add-assign (`+=`)

*Signature*: `void operator+=(Layer new_layer)`

Adds `new_layer` as the final layer of the network.

If the network is enabled, this method throws `illegal_state`.

Equivalent to `{networkName}.add_layer(new_layer)`.

**Parameters**

* `new_layer` (`Layer`): Layer to add to the back of the network.

**Exceptions**

* `illegal_state`: If the network is enabled.

---

#### output stream insertion (`<<`)

*Signature*: `friend std::ostream& operator<<(std::ostream& os, const Network& network)`

Exports `network` to the output stream `os`, returning a reference to `os` with `network` added.

The exported network, as a `std::string`, contains the Network's enabled/disabled status, loss calculator, optimizer, and layers.

**Returns**

* `std::ostream` (`std::ostream`): New output stream containing the network's information.

**Parameters**

* `os` (`std::ostream&`): Output stream to export to.
* `network` (`const Network&`): Network to export.


---
---


## ActivationFunction

Abstract class representing an activation function, along with all functions required to perform backpropagation.

Commonly used when pointed to by a `shared_ptr` smart pointer.

Pre-implemented concrete subclasses:
- `IdentityActivation` (name: "none"), a placeholder that does nothing to its input
- `Relu` (name: "relu"), the Rectified Linear Unit (ReLU) function
- `Sigmoid` (name: "sigmoid")
- `Softmax` (name: "softmax"). The only layer that can use `Softmax` activation is the final layer.

Names are accessed using the `name` method.

---

### Constructor

#### Default constructor

All pre-implemented subclasses of `ActivationFunction` have a constructor that takes no arguments. Example: `Relu()`, `Sigmoid()`

---

### Methods

#### compute

*Signature:* `virtual Eigen::VectorXd compute(const Eigen::VectorXd& input)`

Applies the activation function element-wise to the input.

**Returns**

* `Eigen::VectorXd`: `input` after applying this activation function element-wise.

**Parameters**

* `input` (`const Eigen::VectorXd&`): Value to calculate.

---

#### compute\_derivative

*Signature:* `virtual Eigen::VectorXd compute_derivative(const Eigen::VectorXd& input)`

Applies the derivative of the activation function element-wise to the input.

**Returns**

* `Eigen::VectorXd`: Activation function's derivative applied element-wise to `input`.

**Parameters**

* `input` (`const Eigen::VectorXd&`): Value to calculate.

---

#### name

*Signature:* `virtual std::string name()`

Returns the unique identifier for the activation function.
If not overridden, returns `"none"`.

`IdentityActivation`'s name is "none".

A function object's name is usually the class's name converted to lowercase, with underscores separating each word. Example: `Relu`'s name is "relu".

**Returns**

* `std::string`: Name of the activation function.

---

#### using\_pre\_activation

*Signature:* `virtual bool using_pre_activation()`

Indicates whether the activation function should be applied on pre-activation inputs.
If not overridden, returns `false`.

**Returns**

* `bool`: `true` if pre-activation input is used, `false` otherwise.


---
---


## Layer

A linear layer in a network.

There is no distinction between an input, hidden, or output layer. A layer's role depends on its position in the network.

Each layer has a weight matrix and separate bias vector.
They can be manually viewed or updated using getter and setter methods.

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

Creates a new Layer and loads it with the provided fields. Weights and biases are randomly initialized in the uniform range [-1, 1].

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

*Signature:* `Eigen::VectorXd bias_vector()`

Returns the layer's bias vector, as a `Eigen::VectorXd`.

**Returns**

* `Eigen::VectorXd`: Bias vector, containing `{layerName}.output_dimension()` elements.

---

#### input\_dimension

*Signature:* `int input_dimension()`

Returns the number of inputs for the layer.

**Returns**

* `int`: Number of input elements.

---

#### name

*Signature:* `std::string name()`

Returns the name of the layer.

**Returns**

* `std::string`: Layer name.

---

#### output\_dimension

*Signature:* `int output_dimension()`

Returns the number of outputs for the layer.

**Returns**

* `int`: Number of output elements.

---

#### weight\_matrix

*Signature:* `Eigen::MatrixXd weight_matrix()`

Returns the layer's weight matrix.

**Returns**

* `Eigen::MatrixXd`: Weight matrix, with `{layerName}.output_dimension()` rows and `{layerName}.input_dimension()` columns.

---

### Setters

#### set\_activation\_function

*Signature:* `void set_activation_function(std::shared_ptr<ActivationFunction> new_activation_function)`

Sets the layer's activation function to `new_activation_function`.

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

Sets the name of the layer.

**Parameters**

* `new_name` (`std::string`): New name for the layer.

---

#### set\_weight\_matrix

*Signature:* `void set_weight_matrix(Eigen::MatrixXd new_weights)`

Sets the layer's weight matrix.

**Parameters**

* `new_weights` (`Eigen::MatrixXd`): New matrix of weights. Must have `{layerName}.output_dimension()` rows and `{layerName}.input_dimension()` columns.

---

### Methods

#### forward

*Signature:* `Eigen::VectorXd forward(const Eigen::VectorXd& input)`

Performs the linear forward operation for the given input.

Applies weights and adds biases.
**Important note**: This method does *not* apply the layer's activation function.

**Returns**

* `Eigen::VectorXd`: Resulting vector of length `{layerName}.output_dimension()`.

**Parameters**

* `input` (`const Eigen::VectorXd&`): Input column vector. Must have `{layerName}.input_dimension()` elements.

---

### Operator Overloads

#### output stream insertion (`<<`)

*Signature:* `friend std::ostream& operator<<(std::ostream& os, const Layer& layer)`

Exports `layer` to the output stream `os`, returning a new output stream with `layer` inside.

**Returns**

* `std::ostream&`: Output stream with `layer` added.

**Parameters**

* `os` (`std::ostream&`): Output stream to write to.
* `layer` (`Layer`): Layer to export.




---
---


## LossCalculator

Abstract class for calculating loss.

A `shared_ptr` smart pointer to a `LossCalculator` can be used by a `Network`.

Pre-implemented concrete subclasses:

* `CrossEntropy`
* `MeanSquaredError`

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

**Returns**

* `std::string`: Name of the loss calculator (typically, the lowercased class name with underscores separating words, e.g. `"cross_entropy"`).


---
---


## Optimizer

Abstract class for network optimizers.

A `shared_ptr` smart pointer to an `Optimizer` instance can be used by a `Network`.

Pre-implemented concrete subclasses:

* `SGD`, a Stochastic Gradient Descent optimizer

Further info is in the [optimizer-specific documentation](optimizers.md).


---
---
---
[Back to table of contents](#table-of-contents)