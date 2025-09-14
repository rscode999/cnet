
# Class and Method Documentation

Documentation for each class and its methods.

[Back to README](README.md)

## Table of Contents


Types of methods:
- Constructor: Creates a new instance of the class
- Getter: Retrieves a class attribute
- Setter: Changes the value of a class attribute
- Methods: Any other function that is not a getter or setter
- Operator Overload: Allows usage of operators on the class

Each section (i.e. Network, ActivationFunction) is for a class.  
The subsections (i.e. Constructor, Getters, Setters) are types of methods that can be called with the class.


- [Network](#network)
    - [Illegal State Exception](#illegal_state)
    - [Constructor](#constructor)
    - [Getters](#getters)
    - [Setters](#setters)
    - [Methods](#methods)
    - [Operator Overloads](#operator-overloads)
- [ActivationFunction](#activationfunction)
    - [Constructor](#constructor-1)
    - [Methods](#methods-1)
- [Layer](#layer)
    - [Constructors](#constructor-2)
    - [Getters](#getters-1)
    - [Setters](#setters-1)
    - [Methods](#methods-2)
    - [Operator Overloads](#operator-overloads-1)
- [LossCalculator](#losscalculator)
    - [Constructor](#constructor-3)
    - [Methods](#methods-3)
- [Optimizer](#optimizer)
    - [Constructor](#constructor-4)
        - [SGD](#sgd-concrete-class)
    - [Methods](#methods-4)


---
---
---


## Network

A neural network that can be trained and used for predictions.

The user adds layers, a loss calculator, and an optimizer to the network prior to use.

Each layer in a network has an index number, ranging from 0 to `{networkName}.layer_count()` - 1. The first layer in a network has index 0. The second has index 1, the third has index 2, and so on.
The input layer is the first layer. The output layer is the last layer.

To use a network, the network must be enabled via `{networkName}.enable()`.
Once enabled, the network cannot be edited until `{networkName}.disable()` is called.


---

### illegal_state

An exception thrown only by a Network. Thrown when enable/disable rules are broken.

Class name: `illegal_state`. A subclass of `std::runtime_error`.

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

*Signature*: `VectorXd biases_at(int layer_number)`

Returns a deep copy of the bias vector in layer `layer_number`.

**Returns**

* `VectorXd`: Biases of layer `layer_number`.

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

*Signature*: `MatrixXd weights_at(int layer_number)`

Returns a deep copy of the weight matrix in layer `layer_number`.

**Returns**

* `MatrixXd`: Weight matrix at the specified layer.

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

Adds a new layer with the given dimensions and name. The new layer has no activation function.

To use this method, the network must be disabled.

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

Disables the network, allowing the network to be modified. The network's stored intermediate outputs from training are cleared.

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

*Signature*: `void remove_layer(std::string layer_name)`

Removes a layer with the name `layer_name`.

The first layer with a matching name, i.e. the one with the lowest index number, is removed.
If no matching layer names are found, throws `out_of_range`.

To use this method, the network must be disabled.

**Parameters**

* `layer_name` (`std::string`): Name of the layer to remove.

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

To use this method, the network must be disabled.

**Parameters**

* `layer_number` (`int`): Index of the layer. Must be on the interval [0, `{networkName}.layer_count()`-1].
* `new_activation_function` (`std::shared_ptr<ActivationFunction>`): Smart pointer to the function.

**Exceptions**

* `illegal_state`: If the network is enabled.

---

#### set\_biases\_at

*Signature*: `void set_biases_at(int layer_number, MatrixXd new_biases)`

Sets the biases of the specified layer.

To use this method, the network must be disabled.

**Parameters**

* `layer_number` (`int`): Index of the layer. Must be on the interval [0, `{networkName}.layer_count()`-1].
* `new_biases` (`MatrixXd`): New bias vector (column vector). Must have `output_dimension()` rows and 1 column.

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

#### set\_weights\_at

*Signature*: `void set_weights_at(int layer_number, MatrixXd new_weights)`

Sets the weight matrix of the specified layer.

To use this method, the network must be disabled.

**Parameters**

* `layer_number` (`int`): Layer index. Must be on the interval [0, `{networkName}.layer_count()`-1].
* `new_weights` (`MatrixXd`): New weight matrix. Must have `{selectedLayer}.output_dimension()` rows and `{selectedLayer}.input_dimension()` columns.

**Exceptions**

* `illegal_state`: If the network is enabled.

---

### Methods

#### forward

*Signature*: `VectorXd forward(const MatrixXd& input, bool training = true)`

Returns the output of the network for a given input.

If `training` is true, intermediate outputs are stored for backpropagation.

**Returns**

* `VectorXd`: Network's output.

**Parameters**

* `input` (`MatrixXd`): Input vector.
* `training` (`bool`): Whether the network is in training mode. Default: `true`.

---

#### predict

*Signature*: `VectorXd predict(const MatrixXd& input)`

Returns the predictions for the given input.

When this method is used, the network *does not* internally record intermediate layer outputs for backpropagation.

Equivalent to `{networkName}.forward(input, false)`.

Requires that the network is enabled.

**Returns**

* `VectorXd`: Network's output.

**Parameters**

* `input` (`MatrixXd`): Input vector to make predictions with.

---

#### reverse

*Signature*: `void reverse(const MatrixXd& predictions, const MatrixXd& actuals)`

Updates the weights and biases of this network using `predictions` and `actuals`, using the network's optimizer.

This method requires the network to be enabled. Also, `{networkName}.forward` with `training = true` must have been called since the network was enabled.
If these conditions are not met, the method throws `illegal_state`.

**Parameters**

* `predictions` (`MatrixXd`): Predicted outputs from the network. Must have `{networkName}.output_dimension()` rows and 1 column.
* `actuals` (`MatrixXd`): Expected outputs that the network should have produced. Must have `{networkName}.output_dimension()` rows and 1 column.

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

The exported network, as a std::string, contains its enabled/disabled status, loss calculator, optimizer, and all layers.

**Returns**

* `std::ostream` (`std::ostream`): New output stream containing the network's information.

**Parameters**

* `os` (`std::ostream&`): Output stream to export to.
* `network` (`Network`): Network to export.


---
---


## ActivationFunction

Abstract class representing an activation function, along with all functions required to perform backpropagation.

Pre-implemented concrete subclasses:
- `IdentityActivation`, a placeholder that does nothing to its input
- `Relu`, the Rectified Linear Unit (ReLU) function
- `Sigmoid`
- `Softmax`. The only layer that can use `Softmax` activation is the final layer.

---

### Constructor

#### Default constructor

All pre-implemented subclasses of `ActivationFunction` have a constructor that takes no arguments. Example: `Relu()`, `Sigmoid()`

---

### Methods

#### compute

*Signature:* `virtual VectorXd compute(const VectorXd& input)`

Applies the activation function element-wise to the input.

**Returns**

* `VectorXd`: `input` after applying this activation function element-wise.

**Parameters**

* `input` (`VectorXd`): Value to calculate.

---

#### compute\_derivative

*Signature:* `virtual VectorXd compute_derivative(const VectorXd& input)`

Applies the derivative of the activation function element-wise to the input.

**Returns**

* `VectorXd`: Activation function's derivative applied element-wise to `input`.

**Parameters**

* `input` (`VectorXd`): Value to calculate.

---

#### name

*Signature:* `virtual std::string name()`

Returns the unique identifier for the activation function.
If not overridden, returns `"none"`.

**Returns**

* `std::string`: Name of the activation function, typically lowercase (e.g. `"relu"`).

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

Creates a new Layer and initializes all fields. The activation function is set to the identity function, `f(x) = x`.

All values are randomly initialized in the range `[-1, 1]`.

**Parameters**

* `input_dimension` (`int`): Number of inputs that the Layer takes in. Must be positive.
* `output_dimension` (`int`): Number of outputs that the Layer gives. Must be positive.
* `name` (`std::string`): Identifier for this Layer. Default: `"layer"`

---

#### With activation function

*Signature:* `Layer(int input_dimension, int output_dimension, std::shared_ptr<ActivationFunction> activation_function, std::string name = "layer")`

Creates a new Layer and loads it with the provided fields. Values are randomly initialized in the range `[-1, 1]`.

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

*Signature:* `VectorXd bias_vector()`

Returns the layer's bias vector, as a `VectorXd`.

**Returns**

* `VectorXd`: Bias vector.

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

*Signature:* `MatrixXd weight_matrix()`

Returns the layer's weight matrix.

**Returns**

* `MatrixXd`: Weight matrix.

---

### Setters

#### set\_activation\_function

*Signature:* `void set_activation_function(std::shared_ptr<ActivationFunction> new_activation_function)`

Sets the layer's activation function to `new_activation_function`.

**Parameters**

* `new_activation_function` (`std::shared_ptr<ActivationFunction>`): Smart pointer to new activation function.

---

#### set\_bias\_vector

*Signature:* `void set_bias_vector(MatrixXd new_biases)`

Sets the layer’s bias vector.

**Parameters**

* `new_biases` (`MatrixXd`): New vector of biases. Must have `output_dimension()` rows and 1 column.

---

#### set\_name

*Signature:* `void set_name(std::string new_name)`

Sets the name of the layer.

**Parameters**

* `new_name` (`std::string`): New name for the layer.

---

#### set\_weight\_matrix

*Signature:* `void set_weight_matrix(MatrixXd new_weights)`

Sets the layer’s weight matrix.

**Parameters**

* `new_weights` (`MatrixXd`): New matrix of weights. Must have `output_dimension()` rows and `input_dimension()` columns.

---

### Methods

#### forward

*Signature:* `VectorXd forward(const MatrixXd& input)`

Performs the linear forward operation for the given input.

Applies weights and adds biases.
**Important note**: This method does *not* apply the layer's activation function.

**Returns**

* `VectorXd`: Resulting vector of length `output_dimension()`.

**Parameters**

* `input` (`MatrixXd`): Input column vector. Must have `input_dimension()` rows and 1 column.

---

### Operator Overloads

#### output stream insertion (`<<`)

*Signature:* `friend std::ostream& operator<<(std::ostream& os, const Layer& layer)`

Exports `layer` to the output stream `os`, returning a new output stream with `layer` inside.

**Parameters**

* `os` (`std::ostream&`): Output stream to write to.
* `layer` (`Layer`): Layer to export.

**Returns**

* `std::ostream&`: Output stream with `layer` added.


---
---


## LossCalculator

Abstract class for calculating loss.

A smart pointer to a `LossCalculator` can be used by a `Network`.

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

*Signature:* `virtual double compute_loss(const MatrixXd& predictions, const MatrixXd& actuals)`

Returns the loss (error) when measured between `predictions` and `actuals`.

**Returns**

* `double`: Calculator's loss of the model predictions.

**Parameters**

* `predictions` (`MatrixXd`): Model's predictions for a given input. Must be a column vector.
* `actuals` (`MatrixXd`): True values for model predictions. Must be a column vector with the same number of rows as `predictions`.

---

#### compute\_loss\_gradient

*Signature:* `virtual VectorXd compute_loss_gradient(const MatrixXd& predictions, const MatrixXd& actuals)`

Returns the gradient of the losses when measured between `predictions` and `actuals`.

**Returns**

* `VectorXd`: Calculator's loss gradient of the model predictions.

**Parameters**

* `predictions` (`MatrixXd`): Model's predictions for a given input. Must be a column vector.
* `actuals` (`MatrixXd`): True values for model predictions. Must be a column vector with the same number of rows as `predictions`.

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

An Optimizer's only abstract method, `step`, is called internally by a `Network`.
Users should never call the `step` method directly.

Pre-implemented concrete subclasses:

* `SGD`, for Stochastic Gradient Descent

---

### Constructor

#### SGD concrete class

*Signature:* `SGD(double learning_rate = 0.01, double momentum_coefficient = 0)`

Creates a new Stochastic Gradient Descent (SGD) optimizer with the assigned learning rate and momentum coefficient.

**Parameters**

* `learning_rate` (`double`): Learning rate to use, dictating speed and precision of convergence. Must be positive. Default: `0.01`.
* `momentum_coefficient` (`double`): Momentum coefficient to use. Cannot be negative. Default: `0`.

---

### Methods

#### name

*Signature:* `virtual std::string name()`

Returns the optimizer’s identifying string.
If not overridden, returns `"optimizer"`.

**Returns**

* `std::string`: Name of the optimizer.

---

#### step

*Signature:*

```
virtual void step(
    vector<Layer>& layers,
    const VectorXd& initial_input,
    const vector<LayerCache>& intermediate_outputs,
    const MatrixXd& predictions,
    const MatrixXd& actuals,
    const std::shared_ptr<LossCalculator> loss_calculator
)
```

Performs an optimization step on the network’s layers using gradients calculated from `predictions` and `actuals`.  
This method is used internally by a Network, and is not intended to be called directly by users.

**Parameters**

* `layers` (`vector<Layer>&`): Vector of layers to optimize.
* `initial_input` (`const VectorXd&`): Input originally provided to the network.
* `intermediate_outputs` (`const vector<LayerCache>&`): Outputs from each layer before and after activation.
* `predictions` (`const MatrixXd&`): Final output of the network for `initial_input`.
* `actuals` (`const MatrixXd&`): Target output corresponding to `initial_input`.
* `loss_calculator` (`const std::shared_ptr<LossCalculator>`): Smart pointer to the loss calculator used for computing gradients.
