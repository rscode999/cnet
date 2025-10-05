#include <cmath>
#include <string>

#include <Eigen/Core>



namespace CNet {



    
/**
 * Holds an activation function, along with all information required to do backpropagation with it.
 * Abstract class.
 * 
 * All concrete implementations have no internal state.
 */
class ActivationFunction {
public:

    /**
     * Returns the output of the given activation function applied to each element of `input`.
     * @param input value to calculate 
     * @return copy of `input` after applying this activation function element-wise
     */
    virtual Eigen::VectorXd compute(const Eigen::VectorXd& input) = 0;

    /**
     * Returns the derivative output of the given activation function
     * @param input value to calculate 
     * @return the activation function's derivative applied element-wise to `input`
     */
    virtual Eigen::VectorXd compute_derivative(const Eigen::VectorXd& input) = 0;

    /**
     * @return unique identifying string for the activation function.
     * If not overridden, returns `"none"`.
     * 
     * Typically the activation function's name in all lowercase.
     */
    virtual std::string name()  {
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
    Eigen::VectorXd compute(const Eigen::VectorXd& input) override {
        Eigen::VectorXd output = input;
        return output;
    }

    /**
     * Returns a Eigen::VectorXd whose indices are the constant value 1, regardless of the input value.
     * @param input input value
     * @return Eigen::VectorXd of 1.0
     */
    Eigen::VectorXd compute_derivative(const Eigen::VectorXd& input) override {
        return Eigen::VectorXd::Constant(input.size(), 1.0);
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
    Eigen::VectorXd compute(const Eigen::VectorXd& input) override {
        return input.unaryExpr([](double x) {
                return (x<0) ? 0.0 : x;
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
    Eigen::VectorXd compute_derivative(const Eigen::VectorXd& input) override {
        return input.unaryExpr([](double x) {
                return (x<0) ? 0.0 : 1.0;
            }
        );
    }

    /**
     * @return `"relu"`, the function's unique identifier.
     */
    std::string name() override {
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
     * Returns a Eigen::VectorXd with the sigmoid function applied element-wise
     * 
     * @param input inputs to compute
     * @return Eigen::VectorXd with the sigmoid function applied
     */
    Eigen::VectorXd compute(const Eigen::VectorXd& input) override {
        return input.unaryExpr([](double x) {
            return 1.0 / (1.0 + exp(-x));
        });
    }

    /**
     * Applies the element-wise sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
     * @param input inputs to compute
     * @return Eigen::VectorXd with the sigmoid function's derivative applied
     */
    Eigen::VectorXd compute_derivative(const Eigen::VectorXd& input) override {
        Eigen::VectorXd sig = compute(input);
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
public:
    /**
     * Creates a new Softmax object
     */
    Softmax() {
    }


    /**
     * @return Softmax computed element-wise over the entire vector
     */
    Eigen::VectorXd compute(const Eigen::VectorXd& input) override {
        Eigen::VectorXd shifted = input.array() - input.maxCoeff();  // for numerical stability
        Eigen::VectorXd exps = shifted.array().exp();
        double sum = exps.sum();
        return exps / sum;
    }

    /**
     * Should not be used.
     */
    Eigen::VectorXd compute_derivative(const Eigen::VectorXd& input) override {
        assert((false && "Should not compute softmax derivative element-wise"));
        throw std::exception();
    }

    /**
     * @return `"softmax"`, the identifier for a Softmax activation function
     */
    std::string name() override {
        return "softmax";
    }

};




}