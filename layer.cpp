#include <functional>
#include <string>
#include <type_traits>
#include <iostream>

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

/**
 * @return `input` as itself
 */
double identity(double input) {
    return input;
}

/**
 * @return the constant value 1, regardless of the input's value
 * 
 * This is the derivative of the identity function, f(x)=x.
 */
double one(double input) {
    return 1;
}


/**
 * @return the Rectified Linear Unit (ReLU) function applied to `input`
 */
double relu(double input) {
    return input>=0 ? input : 0;
}

/**
 * @return the unit step function applied to `input`. 0 if `input` is negative, 1 otherwise.
 * 
 * This is the derivative of the Rectified Linear Unit (ReLU) function, with `unit_step(0)` defined to equal 1.
 */
double unit_step(double input) {
    return input>=0 ? 1 : 0;
}


/**
 * @return the sigmoid function applied to `input`
 */
double sigmoid(double input) {  
    return 1 / (1 + pow(2.71828182845, input * -1));
}


/**
 * @return the derivative of the sigmoid function applied to `input`
 */
double sigmoid_derivative(double input) {  
    return sigmoid(input) * (1 - sigmoid(input));
}




/**
 * Utility struct for storing pre-activation and post-activation results
 * at each step of the forward process
 */
struct LayerCache {
    Eigen::VectorXd pre_activation;
    Eigen::VectorXd post_activation;
};




/**
 * Represents a layer in a network. Can be an input, hidden layer, or output.
 */
class Layer {

    private:
    
    /**
     * Holds the per-neuron weights for this layer- does not contain biases.
     */
    MatrixXd weights;

    /**
     * Holds the biases. Must be a column vector (i.e. has 1 column).
     */
    MatrixXd biases;


    /**
     * Activation function, to be applied element-wise to the output
     */
    function<double(double)> activation_fcn;

    /**
     * Activation function's derivative, to be applied element-wise during backpropagation
     */
    function<double(double)> activation_fcn_derivative;

    
    /**
     * Identifier for this layer
     */
    string name;


    public:

    /**
     * Creates a new Layer and initializes all fields. The activation function is set to the identity function, f(x)=x.
     * 
     * All indices in the layer's weight matrix and bias vector are initialized to random values on the interval [-1, 1].
     * 
     * @param input_dimension number of inputs that the Layer takes in. Must be positive.
     * @param output_dimension number of outputs that the Layer gives. Must be positive.
     * @param name (default: "layer")- identifier for this Layer
     */
    Layer(int input_dimension, int output_dimension, string layer_name = "layer") {
        assert((input_dimension>0 && "Input vector's dimension must be positive"));
        assert((output_dimension>0 && "Output vector's dimension must be positive"));

        //Initialize weights and biases to random values on [-1, 1]
        weights = MatrixXd::Random(output_dimension, input_dimension);
        biases  = MatrixXd::Random(output_dimension, 1);
        
        //set functions to default
        activation_fcn = identity;
        activation_fcn_derivative = one;

        //initialize name and type

        name = layer_name;
    }


    /**
     * Creates a new Layer and loads it with the provided fields.
     * 
     * The caller is responsible for ensuring that `activation_function_derivative` is truly the derivative of `activation_function`.
     * 
     * @param input_dimension number of inputs that the Layer takes in. Must be positive.
     * @param output_dimension number of outputs that the Layer gives. Must be positive.
     * @param activation_function activation function to be applied element-wise to the output
     * @param activation_function_derivative derivative of `activation_function`, for backpropagation
     * @param name (default: "layer")- identifier for this Layer
     */
    Layer(int input_dimension, int output_dimension,
        function<double(double)> activation_function, function<double(double)> activation_function_derivative,
        string layer_name = "layer") {

        assert((input_dimension>0 && "Input vector's dimension must be positive"));
        assert((output_dimension>0 && "Output vector's dimension must be positive"));

        //Initialize weights and biases to random values on [-1, 1]
        weights = MatrixXd::Random(output_dimension, input_dimension);
        biases  = MatrixXd::Random(output_dimension, 1);
        
        //initialize activation function
        activation_fcn = activation_function;
        activation_fcn_derivative = activation_function_derivative;

        //initialize name
        name = layer_name;
    }


    ///////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////
    //GETTERS
    
    /**
     * @return the layer's activation function
     */
    function<double(double)> activation_function() {
        return activation_fcn;
    }

    /**
     * @return the derivative of the layer's activation function
     */
    function<double(double)> activation_function_derivative() {
        return activation_fcn_derivative;
    }

    /**
     * @return the layer's bias vector, as an Eigen::VectorXd.
     */
    VectorXd bias_vector() {
        //Need to explicitly reshape the column matrix into a vector
        return biases.reshaped();
    }
    
    /**
     * @return the number of elements in the layer's input
     */
    int input_dimension() {
        return weights.cols();
    }

    /**
     * @return the number of elements in the layer's output
     */
    int output_dimension() {
        return weights.rows();
    }

    /**
     * @return the layer's name
     */
    string layer_name() {
        return name;
    }

    /**
     * @return the layer's weight matrix, as an Eigen::MatrixXd
     */
    MatrixXd weight_matrix() {
        return weights;
    }


    ///////////////////////////////////////////////////////////////////////////////////
    //SETTERS

    /**
     * Sets the layer's bias vector to `new_biases`, a column vector.
     * 
     * This method also accepts variables of type Eigen::VectorXd.
     * 
     * @param new_biases new vector of biases for the layer to use. Must have `{layerName}.output_dimension()` rows and 1 column
     */
    void set_bias_vector(MatrixXd new_biases) {
        assert((new_biases.cols() == 1 && "New biases must be a column vector"));
        assert((new_biases.rows() == output_dimension() && "New bias vector must be of the same dimension as the output"));
        biases = new_biases;
    }

    /**
     * Sets the layer's weight matrix to `new_weights`.
     * @param new_weights new matrix of weights for the layer to use. Must have `{layerName}.output_dimension()` rows and `{layerName}.input_dimension()` columns
     */
    void set_weight_matrix(MatrixXd new_weights) {
        assert((new_weights.rows() == output_dimension() && "New weight matrix must have a row count equal to the output dimension"));
        assert((new_weights.cols() == input_dimension() && "New weight matrix must have a column count equal to the input dimension"));
        weights = new_weights;
    }




    ///////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////
    //METHODS

    /**
     * Applies the layer's activation function to each value of `input`
     */
    VectorXd apply_activation(const MatrixXd& input) {
        assert((input.cols() == 1 && "The activation's input must be a column vector"));
        assert((input.rows() == output_dimension() && "Activation's input vector must have dimension equal to the layer's output dimension"));

        VectorXd output(input.rows());
        for(int i=0; i<input.rows(); i++) {
            output(i) = activation_fcn(input(i));
        }
        return output;
    }


    
    /**
     * Returns the result of the layer's forward operation on `input`.
     * 
     * Multiplies `input` by the layer's weight matrix and adds the result to the bias vector.
     * Does not apply the activation function.
     * 
     * Transforms a `{layerName}.input_dimension()` column vector into a `{layerName}.output_dimension()` vector.
     * 
     * Also works on variables of type Eigen::VectorXd.
     * 
     * @param input vector to apply the forward operation to. Must have `{layerName}.input_dimension()` rows and 1 column
     * @return vector, of length `{layerName}.output_dimension()`, after `input`'s forward process
     */
    VectorXd forward(const MatrixXd& input) {
        assert((input.cols() == 1 && "The forward process's input must be a column vector"));
        assert((input.rows() == input_dimension() && "Forward process's input vector must have dimension equal to the weight matrix's number of columns"));

        return (weights * input) + biases;
    }
};
