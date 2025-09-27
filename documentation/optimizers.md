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

## Table of Contents

- [Shared Virtual Methods](#shared-virtual-methods)

Pre-implemented concrete subclasses:

- [Stochastic Gradient Descent (SGD)](#sgd)
    - [Constructor](#constructor)
    - [Getters](#getters)
    - [Setters](#setters)
- [Batch SGD (Subclass of SGD)](#batchsgd)
    - [Constructor](#constructor-1)
    - [Getters](#getters-1)
    - [Setters](#setters-1)

---
---
---


## Shared Virtual Methods

The Optimizer is an abstract class. A user cannot directly instantiate an Optimizer. Only its concrete subclasses may be created.

All Optimizers share the virtual methods below. They can be called on any Optimizer instance.

---

#### name

*Signature*: `virtual std::string name()`

Returns the optimizer's identifying string. If not overridden, returns `"optimizer"`.

The string returned is typically the optimizer's class name converted to lowercase, with underscores in place of spaces.

**Returns**

* `std::string`: The optimizer's identifying string

---


#### set_hyperparameters

*Signature*: `virtual void set_hyperparameters(const vector<double>& hyperparameters)`

Sets the optimizer's hyperparameters. The meaning of each index in `hyperparameters` varies depending on the optimizer.

Example of usage for SGD: Index 0 is the new learning rate. Index 1 is the new momentum coefficient.

Note that the preconditions on `hyperparameters`'s length, and for each of its indices, vary, depending on the optimizer.

**Parameters**

* `hyperparameters` (`const std::vector<double>&`): New hyperparameters for the given optimizer. Preconditions depend on the specific optimizer.

---

#### to_string

*Signature*: `virtual std::string to_string()`

Returns a string containing detailed information about the optimizer's state.

The information includes the optimizer's name, as well as the values of its hyperparameters.

**Returns**

* `std::string`: Information about the optimizer.

---
---

## SGD

Uses Stochastic Gradient Descent (SGD) optimization on a Network.  
The SGD optimizer has an adjustable learning rate and momentum coefficient.

---

### Constructor

#### Default Constructor

*Signature*: `SGD(double learning_rate = 0.01, double momentum_coefficient = 0)`

Constructs a new SGD optimizer using the specified hyperparameters.

**Parameters**

* `learning_rate` (`double`): Learning rate, which determines the step size and speed of optimization. Must be positive. Default: 0.01
* `momentum_coefficient` (`double`): Momentum coefficient to accelerate convergence and dampen oscillations. Cannot be negative. Default: 0

---

### Getters

#### learning\_rate

*Signature*: `double learning_rate()`

Returns the learning rate used by the optimizer.

**Returns**

* `double`: The SGD optimizer's current learning rate.

---

#### momentum\_coefficient

*Signature*: `double momentum_coefficient()`

Returns the momentum coefficient used by the optimizer.

**Returns**

* `double`: The SGD optimizer's current momentum coefficient.

---

#### name

*Signature*: `virtual std::string name() override`

Returns `"sgd"`, the identifying string of the optimizer.

**Returns**

* `std::string`: The string `"sgd"`.

---

#### to_string

*Signature*: `virtual std::string to_string()`

Returns a string containing detailed information about the optimizer's state.

The information includes the optimizer's name `"sgd"`, its learning rate, and its momentum coefficient.

**Returns**

* `std::string`: String containing detailed information about the optimizer.

---

### Setters

#### set\_hyperparameters

*Signature*: `virtual void set_hyperparameters(const std::vector<double>& hyperparameters) override`

Sets the optimizer's hyperparameters using the values from `hyperparameters`.

For SGD optimizers, `hyperparameters` must be of length 2.   
Index 0 contains the new *learning rate* to set. It must be positive.  
Index 1 contains the new *momentum coefficient*. It must be non-negative.

Required contents of `hyperparameters`: {new learning rate, new momentum coefficient}

**Parameters**

* `hyperparameters` (`const std::vector<double>&`): Vector of new hyperparameters. Must be of length 2, where index 0 is positive and index 1 is non-negative
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

## BatchSGD

Subclass of the `SGD` optimizer. Capable of training with multiple examples at a time.

A `BatchSGD` optimizer has a preset batch size, `batch_size`, given at the optimizer's construction. The optimizer updates weights and biases on every `batch_size`-th input.  
No updates occur on inputs other than every `batch_size`-th input.

---

### Constructor

#### Default Constructor

*Signature*: `BatchSGD(double learning_rate = 0.01, double momentum_coefficient = 0, int batch_size = 1)`

Constructs a new Batch SGD optimizer using the given hyperparameters.

**Parameters**

* `learning_rate` (`double`): Learning rate, which determines the step size and speed of optimization. Must be positive. Default: 0.01
* `momentum_coefficient` (`double`): Momentum coefficient to accelerate convergence and dampen oscillations. Cannot be negative. Default: 0
* `batch_size` (`int`): Number of samples to average over during training. Must be positive. Default: 1

---


### Getters

#### learning\_rate

*Signature*: `double learning_rate()`

Returns the learning rate used by the optimizer.

**Returns**

* `double`: The SGD optimizer's current learning rate.

---

#### momentum\_coefficient

*Signature*: `double momentum_coefficient()`

Returns the momentum coefficient used by the optimizer.

**Returns**

* `double`: The SGD optimizer's current momentum coefficient.

---

#### name

*Signature*: `std::string name() override`

Returns `"batch_sgd"`, the identifying string of the Batch SGD optimizer.

**Returns**

* `std::string`: The string `"batch_sgd"`.

---

#### to_string

*Signature*: `virtual std::string to_string()`

Returns a string containing detailed information about the optimizer's state.

The information includes the optimizer's name (`"batch_sgd"`), learning rate, momentum coefficient, and batch size.

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

For Batch SGD optimizers, `hyperparameters` must be of length 3.   
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