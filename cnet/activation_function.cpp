#include <cmath>
#include <string>

#include <Eigen/Dense>

using namespace std;

using namespace Eigen;

/**
 * Holds an activation function, along with all information required to do backpropagation with it.
 * 
 * Abstract class.
 */
class ActivationFunction {
public:

    /**
     * Returns the output of the given activation function
     * @param input value to calculate 
     * @return `input` after applying this activation function element-wise
     */
    virtual VectorXd compute(const VectorXd& input) = 0;

    /**
     * Returns the derivative output of the given activation function
     * @param input value to calculate 
     * @return the activation function's derivative applied element-wise to `input`
     */
    virtual VectorXd compute_derivative(const VectorXd& input) = 0;

    /**
     * @return unique identifying string for the activation function.
     * If not overridden, returns `"none"`.
     * 
     * Typically the activation function's name in all lowercase.
     */
    virtual string name()  {
        return "none";
    };

    /**
     * @return whether the activation function should be applied on pre-activation inputs.
     * 
     * Used when the ActivationFunction calculates initial differences on the final bias vector.
     * 
     * If not overridden, returns `false`.
     */
    virtual bool using_pre_activation() {
        return false;
    }

    /**
     * Properly destroys an ActivationFunction
     */
    virtual ~ActivationFunction() = default;
};



/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////



/**
 * A placeholder activation function. Does nothing to its inputs.
 * 
 * Its "activation function" is the identity function f(x)=x.
 * The derivative of the activation is f'(x)=1.
 */
class IdentityActivation : public ActivationFunction {
public:

    /**
     * Creates a new IdentityActivation object
     */
    IdentityActivation() {
    }

    /**
     * Does nothing to the input.
     * @param input input value
     * @return the input value as itself
     */
    VectorXd compute(const VectorXd& input) override {
        VectorXd output = input;
        return output;
    }

    /**
     * Returns a VectorXd whose indices are the constant value 1, regardless of the input value.
     * @param input input value
     * @return VectorXd of 1.0
     */
    VectorXd compute_derivative(const VectorXd& input) override {
        return VectorXd::Constant(input.size(), 1.0);
    }

};



/**
 * The Rectified Linear Unit (ReLU) activation function.
 *
 * The ReLU function returns 0 if the input is negative, otherwise returns the input as itself.
 * Its derivative is 0 for negative input, 1 otherwise.
 */
class Relu : public ActivationFunction {
public:

    /**
     * Creates a new Relu activation function object
     */
    Relu() {
    }

    /**
     * Returns the Rectified Linear Unit function applied to each value in the input
     * @param input The input values
     * @return ReLU applied to the input element-wise
     */
    VectorXd compute(const VectorXd& input) override {
        return input.unaryExpr([](double x) {
                return (x<0) ? 0 : x;
            }
        );
    }

    /**
     * Computes the derivative of the ReLU function (the unit step function) for each element of the input.
     * 
     * `compute_derivative(0)` is defined to be 1.
     * 
     * @param input The input values
     * @return for each element, 1 if `input` >= 0, else 0.
     */
    VectorXd compute_derivative(const VectorXd& input) override {
        return input.unaryExpr([](double x) {
                return (x<0) ? 0.0 : 1.0;
            }
        );
    }

    /**
     * @return `"relu"`, the function's unique identifier.
     */
    string name() override {
        return "relu";
    }

    /**
     * @return the Boolean value `true`.
     * In backpropagation, ReLU is best applied before activations are used.
     */
    bool using_pre_activation() override {
        return true;
    }
};



/**
 * Sigmoid activation function
 */
class Sigmoid : public ActivationFunction {
public:
    /**
     * Creates a new Sigmoid activation function object
     */
    Sigmoid() {
    }

    /**
     * Returns a VectorXd with the sigmoid function applied element-wise
     * 
     * @param input inputs to compute
     * @return VectorXd with the sigmoid function applied
     */
    VectorXd compute(const VectorXd& input) override {
        return input.unaryExpr([](double x) {
            return 1.0 / (1.0 + exp(-x));
        });
    }

    /**
     * Applies the element-wise sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
     * @param input inputs to compute
     * @return VectorXd with the sigmoid function's derivative applied
     */
    VectorXd compute_derivative(const VectorXd& input) override {
        VectorXd sig = compute(input);
        return sig.array() * (1.0 - sig.array());
    }

    std::string name() override {
        return "sigmoid";
    }

    /**
     * @return `true`.
     * This implementation of the Sigmoid derivative should be taken before activation functions are applied.
     */
    bool using_pre_activation() override {
        return true;
    }
};



/**
 * Calculates Softmax activation.
 * 
 * The only layer allowed to use Softmax is the output layer.
 */
class Softmax : public ActivationFunction {

private:

public:
    /**
     * Creates a new Softmax object
     */
    Softmax() {
    }


    /**
     * Softmax over the entire vector
     */
    VectorXd compute(const VectorXd& input) override {
        VectorXd shifted = input.array() - input.maxCoeff();  // for numerical stability
        VectorXd exps = shifted.array().exp();
        double sum = exps.sum();
        return exps / sum;
    }

    /**
     * Should not be used.
     */
    VectorXd compute_derivative(const VectorXd& input) override {
        assert((false && "Should not compute softmax derivative element-wise"));
        throw exception();
    }

    /**
     * @return `"softmax"`, the identifier for a Softmax activation function
     */
    string name() override {
        return "softmax";
    }

};