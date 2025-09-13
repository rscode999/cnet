
# Class and Method Documentation

## Network

A neural network that can be trained and used for predictions.

The user adds layers, a loss calculator, and an optimizer to the network prior to use.
The input layer is the first layer. The output layer is the last layer. Layers use 0-based indexing.

To use a network, the network must be enabled via `{networkName}.enable()`.

### Constructor

#### Network

*Signature*: `Network()`

Creates an empty network.

The created network is not enabled. It has no layers, loss calculator, or optimizer.

### Getters

#### biases\_at

*Signature*: `VectorXd biases_at(int layer_number)`

Returns a deep copy of the bias vector in layer `layer_number`.

**Returns**

* `VectorXd`: Biases of layer `layer_number`.

**Parameters**

* layer\_number: Layer number to access

**Exceptions**

* `out_of_range`: If `layer_number` is not on the interval `[0, {networkName}.layer_count()-1]`

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

* layer\_number: Layer number (0-based). Must be on the interval `[0, layer_count()-1]`.

---

#### layer\_at (by name)

*Signature*: `Layer layer_at(string layer_name)`

Returns the layer whose name is `layer_name`.

The first matching layer name, i.e. the layer with the lowest index number, is returned.

**Returns**

* `Layer`: Matching layer.

**Parameters**

* layer\_name: Name of layer to find. Must match an existing layer.

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

* `MatrixXd`: Weight matrix at the specified layer

**Parameters**

* layer\_number: Index of layer. Must be on the interval `[0, layer_count()-1]`.

---

### Setters

#### add\_layer (Layer)

*Signature*: `void add_layer(Layer new_layer)`

Adds `new_layer` to the back of the network.

To use this method, the network must be disabled.

**Parameters**

* new\_layer: Layer to add.

**Exceptions**

* `illegal_state`: If the network is enabled.

---

#### add\_layer (dimensions + name)

*Signature*: `void add_layer(int input_dimension, int output_dimension, string name = "layer")`

Adds a new layer with the given dimensions and name. The new layer has no activation function.

To use this method, the network must be disabled.

**Parameters**

* input\_dimension: Number of input nodes. Must be positive.
* output\_dimension: Number of output nodes. Must be positive.
* name: Layer name. Default: "layer"

**Exceptions**

* `illegal_state`: If the network is enabled.

---

#### add\_layer (dimensions + activation + name)

*Signature*: `void add_layer(int input_dimension, int output_dimension, shared_ptr<ActivationFunction> activation_function, string name = "layer")`

Adds a new layer with specified activation function and name.

To use this method, the network must be disabled.

**Parameters**

* input\_dimension: Number of input nodes. Must be positive.
* output\_dimension: Number of output nodes. Must be positive.
* activation\_function: Pointer to activation function.
* name: Layer name. Default: "layer"

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
- The network has at least 1 layer.
- A loss calculator and an optimizer are defined.
- The output dimension of each layer must equal the input dimension of the next layer.
- The only layer with Softmax activation is the final (output) layer.


**Exceptions**

* `illegal_state`: If checks on layer count, optimizer, loss calculator, layer compatibility, or softmax placement fail.

---

#### insert\_layer\_at

*Signature*: `void insert_layer_at(int new_pos, Layer new_layer)`

Inserts `new_layer` at position `new_pos`.

The current layer at `new_pos`, and all layers after that layer, have their positions increased by 1. The new layer will have a position number of `new_pos`.

To use this method, the network must be disabled.

**Parameters**

* new\_pos: Insertion index. Must be on the interval `[0, layer_count()]`.
* new\_layer: Layer to insert.

**Exceptions**

* `illegal_state`: If the network is enabled.

---

#### remove\_layer

*Signature*: `void remove_layer(string layer_name)`

Removes a layer with the name `layer_name`.

The first layer with a matching name, i.e. the one with the lowest index number, is removed.  
If no matching layer names are found, throws `out_of_range`.

To use this method, the network must be disabled.

**Parameters**

* layer\_name: Name of the layer to remove.


**Exceptions**

* `illegal_state`: If the network is enabled.
* `out_of_range`: If no matching layer is found.

---

#### remove\_layer\_at

*Signature*: `void remove_layer_at(int remove_pos)`

Removes the layer at position `remove_pos`.

To use this method, the network must be disabled.

**Parameters**

* remove\_pos: Index of the layer. Must be on the interval `[0, layer_count()-1]`.


**Exceptions**

* `illegal_state`: If the network is enabled.

---

#### rename\_layer\_at

*Signature*: `void rename_layer_at(int rename_pos, string new_name)`

Renames the layer at `rename_pos`.

Unlike most setters, this method can be called when the method is enabled.

**Parameters**

* rename\_pos: Layer index to rename. Must be on the interval `[0, layer_count()-1]`.
* new\_name: New name.

---

#### set\_activation\_function\_at

*Signature*: `void set_activation_function_at(int layer_number, shared_ptr<ActivationFunction> new_activation_function)`

Sets the activation function for a given layer.

To use this method, the network must be disabled.

**Parameters**

* layer\_number: Index of the layer. Must be on the interval `[0, layer_count()-1]`.
* new\_activation\_function: Smart pointer to the function.

**Exceptions**

* `illegal_state`: If the network is enabled.

---

#### set\_biases\_at

*Signature*: `void set_biases_at(int layer_number, MatrixXd new_biases)`

Sets the biases of the specified layer.

To use this method, the network must be disabled.

**Parameters**

* layer\_number: Index of the layer. Must be on the interval `[0, layer_count()-1]`.
* new\_biases: New bias vector (column vector). Must have `output_dimension()` rows and 1 column.

**Exceptions**

* `illegal_state`: If the network is enabled.

---

#### set\_loss\_calculator

*Signature*: `void set_loss_calculator(shared_ptr<LossCalculator> new_calculator)`

Replaces the current loss calculator.

To use this method, the network must be disabled.

**Parameters**

* new\_calculator: Smart pointer to new loss calculator.

**Exceptions**

* `illegal_state`: If the network is enabled.

---

#### set\_optimizer

*Signature*: `void set_optimizer(shared_ptr<Optimizer> new_optimizer)`

Replaces the current optimizer.

To use this method, the network must be disabled.

**Parameters**

* new\_optimizer: Smart pointer to new optimizer.

**Exceptions**

* `illegal_state`: If the network is enabled.

---

#### set\_weights\_at

*Signature*: `void set_weights_at(int layer_number, MatrixXd new_weights)`

Sets the weight matrix of the specified layer.

To use this method, the network must be disabled.

**Parameters**

* layer\_number: Layer index. Must be on the interval `[0, layer_count()-1]`.
* new\_weights: New weight matrix. Must have `{selectedLayer}.output_dimension()` rows and `{selectedLayer}.input_dimension()` columns.

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

* input: Input vector.
* training: Whether the network is in training mode. Default: true.

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

* input: Input vector to make predictions with


---

#### reverse

**Signature*:* `void reverse(const MatrixXd& predictions, const MatrixXd& actuals)`

Updates the weights and biases of this network using `predictions` and `actuals`, using the network's optimizer.

This method requires the network to be enabled. Also, `{networkName}.forward` with `training`=true must have been called since the network was enabled.  
If these conditions are not met, the method throws `illegal_state`.

Requires that the network is enabled.

**Parameters**

* predictions: predicted outputs from the network. Must have `{networkName}.output_dimension()` rows and 1 column
* actuals: expected outputs that the network should have produced. Must have `{networkName}.output_dimension()` rows and 1 column

**Exceptions**

* `illegal_state` if the network is disabled

---

### Operator Overrides

#### add-assign (`+=`)

*Signature*: `void operator+=(Layer new_layer)`

Adds `new_layer` as the final layer of the network.

If the network is enabled, this method throws `illegal_state`.

Equivalent to `{networkName}.add_layer(new_layer)`.

**Parameters**

* new\_layer: layer to add to the back of the network

**Exceptions**

* `illegal_state` if the network is enabled

---

#### output stream insertion (`<<`)

*Signature*: `friend std::ostream& operator<<(std::ostream& os, const Network& network)`

Exports `network` to the output stream `os`, returning a reference to `os` with `network` added.

The exported network, as a string, contains its enabled/disabled status, loss calculator, optimizer, and all layers.

**Returns**

* `std::ostream`: New output stream containing the network's information inside

**Parameters**

* os: output stream to export to
* network: network to export