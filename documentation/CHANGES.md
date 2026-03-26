# Changes

## 1.4.0
**26 March 2026**

### Overall

- Eigen Lite is now available! Eigen 3 is no longer required. Existing code is modified to use Eigen Lite.
- Tested with Eigen 5
- Parameters and return values of type `int` are now of type `int32_t`

### Network

- Added `clear_training_state` method, which removes training data without having to enable the network

### Optimizer

- The deprecated methods `batch_size` and `set_batch_size` are removed


## 1.3.0
**30 January 2026**

### Network

- Multithreaded batch training with many input vectors is now available. Batch training with one input vector at a time is no longer implemented

### Optimizer

- Batch training with many input vectors is now available. Batch training with one input vector at a time is no longer implemented
- `batch_size` and `set_batch_size` have been deprecated


## 1.2.0
**1 December 2025**

### Overall

- Added ability to load and save network configurations to files


## 1.1.0
**5 October 2025**

### Overall

- Uses `Eigen/Core` instead of `Eigen/Dense`

### Optimizer

- Implemented batch training (one vector at a time)


## 1.0.0
**21 September 2025**

First full release