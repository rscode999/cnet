#include "activation_function.cpp"

#include <memory>

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;


/**
 * Utility struct for storing pre-activation and post-activation results
 * at each step of the forward process
 */
struct LayerCache {
    Eigen::VectorXd pre_activation;
    Eigen::VectorXd post_activation;
};
//This should probably be put in network.cpp



/**
 * Represents a layer in a network. 
 * 
 * Can be an input, hidden layer, or output.
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
     * Smart pointer to activation function object
     */
    shared_ptr<ActivationFunction> activation_fcn;

    
    /**
     * Identifier for this layer
     */
    string layer_name;


public:

    /**
     * Creates a new Layer and initializes all fields. The activation function is set to the identity function, f(x)=x.
     * 
     * All indices in the layer's weight matrix and bias vector are initialized to random values on the interval [-1, 1].
     * 
     * @param input_dimension number of inputs that the Layer takes in. Must be positive.
     * @param output_dimension number of outputs that the Layer gives. Must be positive.
     * @param name identifier for this Layer. Default: `"layer"`
     */
    Layer(int input_dimension, int output_dimension, string name = "layer") {
        assert((input_dimension>0 && "Input vector's dimension must be positive"));
        assert((output_dimension>0 && "Output vector's dimension must be positive"));

        //Initialize weights and biases to random values on [-1, 1]
        weights = MatrixXd::Random(output_dimension, input_dimension);
        biases  = MatrixXd::Random(output_dimension, 1);
        
        //set functions to default
        activation_fcn = make_shared<IdentityActivation>();

        //initialize name
        this->layer_name = name;
    }


    /**
     * Creates a new Layer and loads it with the provided fields.
     * 
     * @param input_dimension number of inputs that the Layer takes in. Must be positive.
     * @param output_dimension number of outputs that the Layer gives. Must be positive.
     * @param activation_function smart pointer to activation function object to use
     * @param name identifier for this Layer. Default: `"layer"`
     */
    Layer(int input_dimension, int output_dimension,
        shared_ptr<ActivationFunction> activation_function,
        string name = "layer") {

        assert((input_dimension>0 && "Input vector's dimension must be positive"));
        assert((output_dimension>0 && "Output vector's dimension must be positive"));

        //Initialize weights and biases to random values on [-1, 1]
        weights = MatrixXd::Random(output_dimension, input_dimension) * 0.5;
        biases  = MatrixXd::Random(output_dimension, 1) * 0.5;
        
        //initialize activation function
        activation_fcn = activation_function;

        //initialize name
        this->layer_name = name;
    }


    ///////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////
    //GETTERS
    
    /**
     * @return smart pointer to the layer's activation function
     */
    const shared_ptr<ActivationFunction> activation_function() const {
        return activation_fcn;
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
    string name() {
        return layer_name;
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
            output(i) = activation_fcn->compute(input(i));
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


    /**
     * Returns an output stream containing `layer` added to the output stream `os`.
     * @param os output stream to export to
     * @param layer layer to export
     * @return new output stream containing the layer's information inside
     */
    friend std::ostream& operator<<(std::ostream& os, const Layer& layer);
};


std::ostream& operator<<(std::ostream& os, const Layer& layer) {
    os << "Layer \"" << layer.layer_name << "\" (" << layer.weights.cols() << ", " << layer.weights.rows() 
        << "), activation function: " << layer.activation_fcn->identifier();
    return os;
}