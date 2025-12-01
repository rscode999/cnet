# Optimizer Documentation

[Back to central documentation](documentation.md)

Documentation for each optimizer class, inside the `CNet` namespace.

To use an Optimizer, make a `std::shared_ptr` to a concrete Optimizer subclass, then pass the smart pointer to the Network.  
Example with SGD optimizer:
```
std::shared_ptr<SGD> sgd_optim_ptr = std::make_shared<SGD>();
network.set_optimizer(sgd_optim_ptr);
```
Note: Only a concrete subclass of Optimizer may be created. The Optimizer is an abstract class, so directly instantiating `Optimizer` causes an error.

Using an external smart pointer, an Optimizer's hyperparameters can be changed or retrieved.  
Setting hyperparameters can also be done through the Network, with the Network's method `set_optimizer_hyperparameters`.

<br>
<details>
  <summary>Notice to Implementers</summary>
  
The file "optimizer.cpp" contains the method `make_optimizer`. If creating a new optimizer, add the optimizer's name to `make_optimizer`. If not, the optimizer can't be stored and loaded to external files.

</details>
<br>

## Table of Contents

- [Shared Virtual Methods](#shared-virtual-methods)

Pre-implemented concrete subclasses:

- [Stochastic Gradient Descent (SGD)](#sgd)
    - [Constructor](#constructor)
    - [Getters](#getters)
    - [Setters](#setters)

---
---
---


## Shared Virtual Methods

The Optimizer is an abstract class. A user cannot directly instantiate an Optimizer. Only its concrete subclasses may be created.

All Optimizers share the virtual methods below. They can be called on any Optimizer instance.

<details>
<summary>Implementation Details</summary>

Optimizers have a private method, `step`, with the following signature:
```
virtual void step(
    std::vector<Layer>& layers, 
    const Eigen::VectorXd& initial_input,
    const std::vector<LayerCache>& intermediate_outputs,
    const Eigen::VectorXd& predictions,
    const Eigen::VectorXd& actuals,
    const std::shared_ptr<LossCalculator> loss_calculator
)
```
Parameter descriptions:
* `layers` std::vector of layers to optimize
* `initial_input` input value of the network
* `intermediate_outputs` outputs of each layer before and after the layer's activation function is applied (the `LayerCache` struct, defined in the file "layer.cpp", contains two Eigen::VectorXd's: `pre_activation` and `post_activation`)
* `predictions` the output of the network for `initial_input`
* `actuals` what the network should predict for `initial_input`
* `loss_calculator` smart pointer to loss calculator object

This method mutates its `layers` parameter.

When using its `reverse` method, a Network internally calls its optimizer's `step` method.

</details>
<br>

---

#### hyperparameters

*Signature:* `virtual std::vector<double> hyperparameters() const`

Returns a vector containing the optimizer hyperparameters, in the order required by the optimizer's `set_hyperparameters` method.

All hyperparameters are of type `double`, even if some hyperparameters must be integers.

**Returns:**

* `std::vector<double>`: Vector containing the optimizer hyperparameters.

---

#### name

*Signature*: `virtual std::string name() const`

Returns the optimizer's identifying string.

The string returned is typically the optimizer's class name converted to lowercase, with underscores in place of spaces.  
Example: `SGD`'s name is `"sgd"`.

<details>
<summary>Implementation Details</summary>

If not overridden, this method returns the string "optimizer".

</details>
<br>

**Returns**

* `std::string`: The optimizer's identifying string

---


#### set_hyperparameters

*Signature*: `virtual void set_hyperparameters(const vector<double>& hyperparameters)`

Sets the optimizer's hyperparameters. The meaning of each index in `hyperparameters` varies depending on the optimizer.

Example of usage for SGD: Index 0 is the new learning rate. Index 1 is the new momentum coefficient. Index 2 is the new batch size.

Note that the preconditions on `hyperparameters`'s length, and for each of its indices, vary, depending on the optimizer.


**Parameters**

* `hyperparameters` (`const std::vector<double>&`): New hyperparameters for the given optimizer. Preconditions depend on the specific optimizer.

---

#### to_string

*Signature*: `virtual std::string to_string() const`

Returns a string containing detailed information about the optimizer's state.

The information includes the optimizer's name, as well as the values of its hyperparameters.

<details>
<summary>Implementation Details</summary>

If not overridden, this method returns "optimizer".

</details>
<br>

**Returns**

* `std::string`: Information about the optimizer.


---
---


## SGD

Uses Stochastic Gradient Descent (SGD) optimization on a Network.  
The SGD optimizer has an adjustable learning rate and momentum coefficient.


On construction, SGD optimizers have an additional parameter, `batch_size`, allowing for batch training.  
The optimizer updates weights and biases on every `batch_size`-th input.  
No updates occur on inputs other than every `batch_size`-th input.

---

### Constructor

#### Default Constructor

*Signature*: `SGD(double learning_rate = 0.01, double momentum_coefficient = 0, int batch_size = 1)`

Constructs a new SGD optimizer using the given hyperparameters.

**Parameters**

* `learning_rate` (`double`): Learning rate, which determines the step size and speed of optimization. Must be positive. Default: 0.01
* `momentum_coefficient` (`double`): Momentum coefficient to accelerate convergence or dampen oscillations. Cannot be negative. Default: 0
* `batch_size` (`int`): Number of samples to average over during training. Must be positive. Default: 1

---


### Getters

#### batch\_size

*Signature*: `int batch_size() const`

Returns the batch size used by the optimizer.

**Returns**

* `int`: The SGD optimizer's current batch size.

---

#### hyperparameters

*Signature*: `std::vector<double> hyperparameters() const override`

Returns a std::vector of 3 hyperparameters: learning rate (index 0), momentum coefficient (index 1), and batch size (index 2).

**Returns:**

* `std::vector<double>`: vector containing optimizer hyperparameters

---

#### learning\_rate

*Signature*: `double learning_rate() const`

Returns the learning rate used by the optimizer.

**Returns**

* `double`: The SGD optimizer's current learning rate.

---

#### momentum\_coefficient

*Signature*: `double momentum_coefficient() const`

Returns the momentum coefficient used by the optimizer.

**Returns**

* `double`: The SGD optimizer's current momentum coefficient.

---

#### name

*Signature*: `std::string name() const override`

Returns `"sgd"`, the identifying string of the SGD optimizer.

**Returns**

* `std::string`: The string `"sgd"`.

---

#### to\_string

*Signature*: `std::string to_string() const override`

Returns a string containing detailed information about the optimizer's state.

The information includes the optimizer's name (`"sgd"`), learning rate, momentum coefficient, and batch size.

**Returns**

* `std::string`: String containing detailed information about the optimizer.

---

### Setters

#### set\_batch\_size

*Signature*: `void set_batch_size(int new_batch_size)`

Sets the optimizer's batch size to `new_batch_size`.

When this method is called, the optimizer's internal training state is reset.  
Biases, weights, and momentum data are reset. The number of samples trained in the current batch is set to 0.

**Parameters**

* `new_batch_size` (`int`): new number of training examples to average over during training. Must be positive.


---

#### set\_hyperparameters

*Signature*: `void set_hyperparameters(const std::vector<double>& hyperparameters) override`

Sets the optimizer's hyperparameters using the values from `hyperparameters`.

For SGD optimizers, `hyperparameters` must be of length 3.   
Index 0 contains the new *learning rate* to set. It must be positive.  
Index 1 contains the new *momentum coefficient*. It must be non-negative.  
Index 2 contains the new *batch size*. It must be positive, when rounded down to the nearest integer.

If the batch size is changed, the optimizer's internal training data (i.e. momentum, intermediate layer outputs, number of samples trained so far) is reset.

Required contents of `hyperparameters`: {new learning rate, new momentum coefficient, new batch size}

**Parameters**

* `hyperparameters` (`const std::vector<double>&`): Vector of new hyperparameters. Must be of length 3, where index 0 is positive, index 1 is non-negative, and index 2 (rounded down) is positive
---

#### set\_learning\_rate

*Signature*: `void set_learning_rate(double new_learning_rate)`

Sets the optimizer's learning rate to `new_learning_rate`.

**Parameters**

* `new_learning_rate` (`double`): New learning rate to use. Must be positive.


---

#### set\_momentum\_coefficient

*Signature*: `void set_momentum_coefficient(double new_momentum_coefficient)`

Sets the optimizer's momentum coefficient to `new_momentum_coefficient`.

**Parameters**

* `new_momentum_coefficient` (`double`): New momentum coefficient to use. Cannot be negative.


---
---
---

[Back to table of contents](#table-of-contents)