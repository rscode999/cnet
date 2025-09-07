#include "optimizer.cpp"

/**
 * Thrown to indicate that the model is not in the proper state to call a method.
 */
class illegal_state : exception {

private:
    /**
     * Error message explaining what is wrong
     */
    string msg;

public:
    /**
     * Creates a new exception with an empty error message
     */
    illegal_state() {
        msg = "";
    }

    /**
     * Creates a new exception with `error_message` as the error message
     * @param error_message the error message to display upon throw
     */
    illegal_state(std::string error_message) {
        msg = error_message;
    }

    /**
     * Displays the error message
     */
    const char* what() const noexcept override {
        return msg.c_str();
    }
};





/**
 * A neural network that can be trained and used for predictions
 */
class Network {

public:


/**
 * The input vector used for predictions.
 * 
 * Used in backpropagation.
 */
VectorXd initial_input;

/**
 * Outputs from before and after the activation function is applied at each layer.
 * 
 * Used in backpropagation.
 */
vector<LayerCache> intermediate_outputs;


/**
 * The network's (linear) layers.
 * Index 0 is the input. The final index is the output.
 */
vector<Layer> layers;


/**
 * Smart pointer to object that calculates losses
 */
shared_ptr<LossCalculator> loss_calculator;

/**
 * Smart pointer to object that improves the model's weights.
 */
shared_ptr<Optimizer> optimizer;




/**
 * Whether the network is ready for training and evaluation.
 * 
 * A network is operation-ready if the network has at least 1 layer,
 * a loss calculator and optimizer are defined,
 * and the input/output dimensions of each layer are compatible.
 */
bool training_mode;



public:

/**
 * Creates a new network with no defined layers or optimizer
 */
Network() {
    training_mode = false;
}


////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
//GETTERS

/**
 * Returns a deep copy of the bias vector in layer `layer_number`.
 * 
 * Layers use 0-based indexing. The first layer is at layer number 0.
 * 
 * @param layer_number layer number to access
 * @return biases of layer `layer_number`
 */
VectorXd biases_at(int layer_number) {
    if(layer_number<0 || layer_number>=layers.size()) {
        throw out_of_range("Layer number must be on the interval [0, " + to_string(layers.size()-1) + "]");
    }

    return layers[layer_number].bias_vector();
}



/**
 * @return the number of inputs of this network
 * @throws `illegal_state` if the network has less than 1 layer
 */
int input_dimension() {
    if(layers.size()<1) {
        throw illegal_state("The network must have at least 1 layer");
    }
    return layers[0].input_dimension();
}



/**
 * @return the number of layers in the network
 */
int n_layers() {
    return layers.size();
}



/**
 * @return the number of outputs of this network
 * @throws `illegal_state` if the network has less than 1 layer
 */
int output_dimension() {
    if(layers.size()<1) {
        throw illegal_state("The network must have at least 1 layer");
    }
    return layers[layers.size() - 1].output_dimension();
}



/**
 * Returns a deep copy of the weight matrix in layer `layer_number`.
 * 
 * Layers use 0-based indexing. The first layer is at layer number 0.
 * 
 * @param layer_number layer number to access
 * @return weights of layer `layer_number`
 */
MatrixXd weights_at(int layer_number) {
    if(layer_number<0 || layer_number>=layers.size()) {
        throw out_of_range("Layer number must be on the interval [0, " + to_string(layers.size()-1) + "]");
    }

    return layers[layer_number].weight_matrix();
}


/////////////////////////////////////////////////////////////////////////////////////////////
//SETTERS

/**
 * Sets the bias vector at layer `layer_number` to `new_biases`.
 * 
 * The input layer's number is 0. The first hidden layer's number is 1.
 * 
 * This method also accepts `new_biases` of type Eigen::VectorXd.
 * 
 * @param layer_number which layer's biases to set (0-based indexing). Must be between 0 and {networkName}.n_layers()-1, inclusive on both ends
 * @param new_biases new bias vector to set. Must have {selected layer}.output_dimension() rows and 1 column
 * @throws `illegal_state` if the network is in training mode
 */
void set_biases_at(int layer_number, MatrixXd new_biases) {
    assert((layer_number>=0 && layer_number<layers.size() && "Layer number must be between 0 and (number of layers)-1"));
    assert((new_biases.rows() == layers[layer_number].output_dimension() && "New bias vector's number of rows must equal the selected layer's output dimension"));
    assert((new_biases.cols() == 1 && "New biases must be a column vector"));

    if(training_mode) {
        throw illegal_state("Cannot update biases in training mode");
    }

    layers[layer_number].set_bias_vector(new_biases);
}



/**
 * Sets the weights at layer `layer_number` to `new_weights`.
 * 
 * The input layer's number is 0. The first hidden layer's number is 1.
 * 
 * @param layer_number which layer's weights to set (0-based indexing). Must be between 0 and {networkName}.n_layers()-1, inclusive on both ends
 * @param new_weights new weight matrix for layer `layer_number`.
 * Number of rows must equal {selected layer}.output_dimension(), number of columns must equal {selected layer}.input_dimension()
 * @throws `illegal_state` if the network is in training mode
 */
void set_weights_at(int layer_number, MatrixXd new_weights) {
    assert((layer_number>=0 && layer_number<layers.size() && "Layer number must be between 0 and (number of layers)-1"));
    assert((new_weights.rows() == layers[layer_number].output_dimension() && "New weight matrix's row count must equal the layer's output dimension"));
    assert((new_weights.cols() == layers[layer_number].input_dimension() && "New weight matrix's column count must equal the layer's input dimension"));
   
    if(training_mode) {
        throw illegal_state("Cannot update weights in training mode");
    }

    layers[layer_number].set_weight_matrix(new_weights);
}



////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
//METHODS


/**
 * Adds `new_layer` as the final layer of the network.
 * 
 * If the network is in training mode, this method throws `illegal_state`.
 * 
 * @param new_layer layer to add
 * @throws illegal_state if the network is in training mode
 */
void add_layer(Layer new_layer) {
    //NO REFERENCES! The user can change the layers if so.

    if(training_mode) {
        throw illegal_state("Network must not be ready for training to add a layer");
    }
    layers.push_back(new_layer);
}



/**
 * Replaces the curent loss calculator with `new_calculator`.
 * If no loss calculator is defined yet, the defined loss calculator becomes `new_calculator`.
 * 
 * If the network is in training mode, this method throws `illegal_state`.
 * 
 * @param new_calculator smart pointer to loss calculator object
 * @throws `illegal_state` if the network is in training mode
 */
void add_loss_calculator(shared_ptr<LossCalculator> new_calculator) {
    if(training_mode) {
        throw illegal_state("Network must not be ready for training to update the loss calculator");
    }
    loss_calculator = new_calculator;
}



/**
 * Replaces the curent optimizer with `new_optimizer`.
 * If no optimizer is defined yet, the defined optimizer becomes `new_optimizer`.
 * 
 * If the network is in training mode, this method throws `illegal_state`.
 * 
 * @param new_optimizer smart pointer to desired optimizer object
 * @throws `illegal_state` if the network is in training mode
 */
void add_optimizer(shared_ptr<Optimizer> new_optimizer) {
    if(training_mode) {
        throw illegal_state("Network must not be ready for training to update the optimizer");
    }
    optimizer = new_optimizer;
}



/**
 * Sets the network to training mode.
 * 
 * Before changing the mode, the network performs a check.
 * The network must have at least 1 layer, a loss calculator, and an optimizer.
 * The output dimension of each layer must equal the input dimension of the next layer.
 * 
 * If the check fails, the method throws `illegal_state`.
 * 
 * @throws `illegal_state` if the network state check fails
 */
void enable_training() {

    //Ensure there are at least 1 layer
    if(layers.size() < 1) {
        throw illegal_state("The network needs at least 1 layer to begin training");
    }

    //Ensure there is a loss calculator
    if(!loss_calculator) {
        throw illegal_state("The network must have a loss calculator to begin training");
    }

    //Ensure there is an optimizer
    if(!optimizer) {
        throw illegal_state("The network must have an optimizer to begin training");
    }

    //Check inputs and outputs of each layer
    for(int i=0; i<layers.size()-1; i++) {
        if(layers[i].output_dimension() != layers[i+1].input_dimension()) {
            throw illegal_state("Output dimension of layer " + to_string(i) + " (dimension=" + to_string(layers[i].output_dimension()) +
            ") must equal the input dimension of layer " + to_string(i+1) + " (dimension=" + to_string(layers[i+1].input_dimension()) + ")");
        }
    }

    //Initialize input vector and intermediate outputs
    initial_input = VectorXd(input_dimension());
    intermediate_outputs.clear();
    training_mode = true;
}



/**
 * Returns the forward operation for the given input, i.e. the network's predictions for `input`.
 * 
 * If the network is not in training mode, this method throws `illegal_state`.
 * 
 * @param input input to the network. Must have `{networkName}.input_dimension()` rows and 1 column
 * @return the network's output, as a VectorXd of dimension `{networkName}.output_dimension()`
 * @throws `illegal_state` if the network is not in training mode
 */
VectorXd forward(const MatrixXd& input) {
    assert((input.cols() == 1 && "Input to forward operation must be a column vector"));
    assert((input.rows() == input_dimension() && "Input to forward operation must have same dimension as the network's input"));
    
    //Training mode check
    if(!training_mode) {
        throw illegal_state("Forward operation requires the network to be in training mode");
    }

    intermediate_outputs.clear();
    intermediate_outputs.reserve(layers.size());

    //Pass input through all layers' forward operations
    VectorXd current_layer_output = input;
    for(int i=0; i<layers.size(); i++) {
        VectorXd pre_activation = layers[i].forward(current_layer_output);
        current_layer_output = layers[i].apply_activation(pre_activation);
        intermediate_outputs.push_back({pre_activation, current_layer_output});
    }

    return current_layer_output;
}



/**
 * Not yet implemented
 */
void reverse(const MatrixXd& predictions, const MatrixXd& actuals) {
    //Later: Put assertion that length of the intermediate outputs must equal the length of the layers

    optimizer->step(layers, initial_input, intermediate_outputs,
        predictions, actuals, loss_calculator);
}

};
