#include <cmath>
#include <string>

using namespace std;


/**
 * Abstract class. Holds an activation function, its derivative, and a unique identifier.
 */
class ActivationFunction {
public:

    /**
     * Returns the output of the given activation function
     * @param input value to calculate 
     * @return `input` after applying this activation function
     */
    virtual double compute(double input) = 0;

    /**
     * Returns the derivative output of the given activation function
     * @param input value to calculate 
     * @return the activation function's derivative applied to `input`
     */
    virtual double compute_derivative(double input) = 0;

    /**
     * @return unique identifying string for the actuvation function.
     * If not overridden, returns `"no_identifier"`.
     */
    virtual string identifier()  {
        return "no_identifier";
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




/**
 * A placeholder activation function. Does nothing to its inputs.
 * 
 * Its "activation function" is the identity function f(x)=x.
 * The derivative of the activation is f(x)=1.
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
    double compute(double input) override {
        return input;
    }

    /**
     * Returns the constant value 1.
     * @param input input value
     * @return 1.0
     */
    double compute_derivative(double input) override {
        return 1.0;
    }

    /**
     * @return the string `"identity"`
     */
    string identifier() override {
        return "identity";
    }
};



/**
 * The Rectified Linear Unit (ReLU) activation function.
 *
 * The ReLU function returns the input if it is positive, otherwise returns 0.
 * Its derivative is 1 for positive input, 0 otherwise.
 */
class Relu : public ActivationFunction {
public:

    /**
     * Creates a new Relu activation function object
     */
    Relu() {
    }

    /**
     * Returns `ReLU(input)`
     * @param input The input value
     * @return ReLU applied to the input
     */
    double compute(double input) override {
        return input > 0 ? input : 0.0;
    }

    /**
     * @brief Computes the unit step function, the derivative of the ReLU function, for the input.
     * @param input The input value
     * @return 1 if `input` > 0, else 0.
     */
    double compute_derivative(double input) override {
        return input > 0 ? 1.0 : 0.0;
    }

    /**
     * @return `"relu"`, the function's unique identifier.
     */
    string identifier() override {
        return "relu";
    }

    /**
     * @return the Boolean value `true`.
     * ReLU is best applied before activations are used.
     */
    bool using_pre_activation() override {
        return true;
    }
};



/**
 * The sigmoid activation function.
 *
 * The sigmoid function returns a value between 0 and 1. It is defined as:
 * sigmoid(x) = 1 / (1 + exp(-x))
 * The derivative is: sigmoid(x) * (1 - sigmoid(x))
 */
class Sigmoid : public ActivationFunction {
public:

    /**
     * Creates a new Sigmoid activation function object
     */
    Sigmoid() {
    }


    /**
     * Returns the result of the sigmoid function on the input.
     * @param input The input value.
     * @return `sigmoid(input)` as a double
     */
    double compute(double input) override {
        return 1.0 / (1.0 + exp(-1 * input));
    }

    /**
     * Returns the derivative of the sigmoid function applied to the input.
     * @param input The input value.
     * @return Sigmoid's derivative: sigmoid(x) * (1 - sigmoid(x)).
     */
    double compute_derivative(double input) override {
        double sigmoid = compute(input);
        return sigmoid * (1.0 - sigmoid);
    }

    /**
     * @return the string `"sigmoid"`
     */
    string identifier() override {
        return "sigmoid";
    }
};