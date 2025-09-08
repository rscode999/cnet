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

private:


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
bool train_mode;



public:

/**
 * Creates an empty network.
 * 
 * The created network has no layers, loss calculator, or optimizer.
 */
Network() {
    train_mode = false;
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
 * @throws `out_of_range` if `layer_number` is not on the interval [0, `{network}.n_layers()` - 1]
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
 * @return deep copy of the layer at `layer_number` (0-based indexing). Must be between 0 and `{network}.n_layers()`-1
 */
Layer layer_at(int layer_number) {
    assert((layer_number>=0 && layer_number<n_layers() && "Layer number must be between 0 and (number of layers)-1, inclusive on both ends"));
    return layers[layer_number];
}



/**
 * Returns the layer whose name is `layer_name`.
 * 
 * The first matching layer name is returned.
 * 
 * If no layer with `layer_name` is found, throws std::out_of_range.
 * 
 * @param layer_name name of layer to find
 * @return layer with the lowest index whose name is `layer_name`
 * @throws `out_of_range` if no matching layer name is found
 */
Layer layer_at(string layer_name) {
    for(Layer l : layers) {
        if(l.name() == layer_name) {
            return l;
        }
    }

    throw out_of_range("Could not find any matching layers with the given name");
}



/**
 * @return the number of layers in the network
 */
int n_layers() const {
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
 * @return whether the network is in training mode
 */
bool training_mode() {
    return train_mode;
}



/**
 * Returns a deep copy of the weight matrix in layer `layer_number`.
 * 
 * Layers use 0-based indexing. The first layer is at layer number 0.
 * 
 * If `layer_number` is not between 0 and `{network}.n_layers()`-1, inclusive on both sides,
 * the method throws std::out_of_range.
 * 
 * @param layer_number layer number to access
 * @return weights of layer `layer_number`
 * @throws `out_of_range` if `layer_number` is not a valid index number
 */
MatrixXd weights_at(int layer_number) {
    if(layer_number<0 || layer_number>=layers.size()) {
        throw out_of_range("Layer number must be on the interval [0, " + to_string(layers.size()-1) + "]");
    }

    return layers[layer_number].weight_matrix();
}


/////////////////////////////////////////////////////////////////////////////////////////////
//SETTERS (DOES NOT INCLUDE WHOLE LAYER OPERATIONS)

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

    if(train_mode) {
        throw illegal_state("Cannot manually set bias vectors in training mode");
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
   
    if(train_mode) {
        throw illegal_state("Cannot manually set weight matrices in training mode");
    }

    layers[layer_number].set_weight_matrix(new_weights);
}



////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
//METHODS


/**
 * Adds `new_layer` to the back of the network.
 * 
 * If the network is in training mode, this method throws `illegal_state`.
 * 
 * @param new_layer layer to add
 * @throws `illegal_state` if the network is in training mode
 */
void add_layer(Layer new_layer) {
    //NO REFERENCES! The user can change the layers if so.

    if(train_mode) {
        throw illegal_state("Network must not be in training mode to add a layer");
    }
    layers.push_back(new_layer);
}



/**
 * Adds a new layer to the back of the network.
 * The new layer has `input_dimension` inputs, `output_dimension` outputs, and a name of `name`.
 * 
 * The layer does not have an activation function.
 * 
 * @param input_dimension dimension of the layer's input
 * @param output_dimension dimension of the layer's output
 * @param name name of the layer. Default: `"layer"`
 */
void add_layer(int input_dimension, int output_dimension, string name="layer") {
    Layer new_layer = Layer(input_dimension, output_dimension, name);
    layers.push_back(new_layer);
}



/**
 * Adds a new layer to the back of the network.
 * The new layer has `input_dimension` inputs, `output_dimension` outputs, 
 * an activation function of `activation_function`. and a name of `name`.
 * 
 * @param input_dimension dimension of the layer's input
 * @param output_dimension dimension of the layer's output
 * @param activation_function smart pointer to activation function of the new layer
 * @param name name of the layer. Default: `"layer"`
 */
void add_layer(int input_dimension, int output_dimension, shared_ptr<ActivationFunction> activation_function, string name="layer") {
    Layer new_layer = Layer(input_dimension, output_dimension, activation_function, name);
    layers.push_back(new_layer);
}



/**
 * Replaces the current loss calculator with `new_calculator`.
 * If no loss calculator is defined yet, the defined loss calculator becomes `new_calculator`.
 * 
 * If the network is in training mode, this method throws `illegal_state`.
 * 
 * @param new_calculator smart pointer to loss calculator object
 * @throws `illegal_state` if the network is in training mode
 */
void add_loss_calculator(shared_ptr<LossCalculator> new_calculator) {
    if(train_mode) {
        throw illegal_state("Network must not be ready for training to update the loss calculator");
    }

    //Free any existing loss calculator
    if(loss_calculator) {
        loss_calculator.reset();
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
    if(train_mode) {
        throw illegal_state("Network must not be ready for training to update the optimizer");
    }

    //Free any existing loss calculator
    if(optimizer) {
        optimizer.reset();
    }
    optimizer = new_optimizer;
}



/**
 * Disables Training Mode for this network, allowing the network to be edited.
 * 
 * Also removes unnecessary memory usage accumulated during training.
 */
void disable_training() {
    intermediate_outputs.clear();
    train_mode = false;
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
    train_mode = true;
}



/**
 * Returns the forward operation for the given input, i.e. the network's predictions for `input`.
 * 
 * If `training` is true, the network internally tracks layer outputs for training.
 * 
 * If the network is not in training mode, this method throws `illegal_state`.
 * 
 * @param input input to the network. Must have `{networkName}.input_dimension()` rows and 1 column
 * @param training true if  training the network, false if getting results for evaluation only. Default: `true`
 * @return the network's output, as a VectorXd of dimension `{networkName}.output_dimension()`
 * @throws `illegal_state` if the network is not in training mode
 */
VectorXd forward(const MatrixXd& input, bool training = true) {
    assert((input.cols() == 1 && "Input to forward operation must be a column vector"));
    assert((input.rows() == input_dimension() && "Input to forward operation must have same dimension as the network's input"));
    
    //Training mode check
    if(!train_mode) {
        throw illegal_state("Forward operation requires the network to be in training mode");
    }

    if(training) {
        intermediate_outputs.clear();
        intermediate_outputs.reserve(layers.size());
        initial_input = input;
    }

    //Pass input through all layers' forward operations
    VectorXd current_layer_output = input;
    for(int i=0; i<layers.size(); i++) {
        VectorXd pre_activation = layers[i].forward(current_layer_output);
        current_layer_output = layers[i].apply_activation(pre_activation);

        if(training) {
            intermediate_outputs.push_back({pre_activation, current_layer_output});
        }
    }

    return current_layer_output;
}



/**
 * Inserts `new_layer` at position `new_pos` in the network.
 * All layers at or after `new_pos` are moved one position backwards.
 * 
 * Inserting at position `{network}.n_layers()` will put the new layer at the end of the network.
 * 
 * Execution time scales linearly with the network's number of layers.
 * 
 * Network layers use 0-based indexing. The first layer is at index 0.
 * 
 * @param new_pos new position number to insert the layer. Must be between 0 and `{network}.n_layers()`, inclusive on both sides
 * @param new_layer layer to insert at position `new_pos`
 * @throws `illegal_state` if this method is called while in training mode
 */
void insert_layer(int new_pos, Layer new_layer) {
    assert((new_pos>=0 && new_pos<=n_layers() && "New layer insertion position must be between 0 and (number of layers)"));
    if(train_mode) {
        throw illegal_state("Cannot insert a new layer while in training mode");
    }

    layers.insert(layers.begin() + new_pos, new_layer);
}



/**
 * Removes the layer at position `remove_pos`.
 * 
 * @param remove_pos layer number to remove. Must be between 0 and `{network}.n_layers()`-1
 */
void remove_layer(int remove_pos) {
    assert((remove_pos>=0 && remove_pos<n_layers() && "Remove position must be between 0 and (number of layers)-1"));
    layers.erase(layers.begin() + remove_pos);
}



/**
 * Removes the layer whose name is `removal_name`.
 * 
 * The first layer in the network whose name matches (i.e. the layer with the lowest index)
 * will be removed.
 * 
 * If there are no matches, throws std::out_of_range.
 * 
 * @param layer_name layer name to remove
 * @throws `out_of_range` if no layer's name matches `removal_name`
 */
void remove_layer(string layer_name) {
    for(int i=0; i<layers.size(); i++) {
        if(layers[i].name() == layer_name) {
            layers.erase(layers.begin() + i);
            return;
        }
    }
    throw out_of_range("No matches found");
}



/**
 * Updates the weights and biases of this network using `predictions` and `actuals`.
 * 
 * The network's optimizer uses the loss calculator for the updating.
 * 
 * @param predictions what the network predicts for a given input
 * @param actuals expected output for the network's prediction
 */
void reverse(const MatrixXd& predictions, const MatrixXd& actuals) {
    assert((predictions.cols() == 1 && "Reverse process predictions must be a column vector"));
    assert((actuals.cols() == 1 && "Reverse process actuals must be a column vector"));
    assert((predictions.rows() == output_dimension() && "Reverse process predictions length must equal network's output dimension"));
    assert((actuals.rows() == output_dimension() && "Reverse process actuals length must equal network's output dimension"));

    //Use the network's optimizer
    optimizer->step(layers, initial_input, intermediate_outputs,
        predictions, actuals, loss_calculator);
}



////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
//OPERATOR OVERRIDES

/**
 * Adds `new_layer` as the final layer of the network.
 * 
 * If the network is in training mode, this method throws `illegal_state`.
 * 
 * Equivalent to `{network}.add_layer(new_layer)`.
 * 
 * @param new_layer layer to add
 * @throws illegal_state if the network is in training mode
 */
void operator+(Layer new_layer) {
    if(train_mode) {
        throw illegal_state("(+ operator) Cannot add layer while in training mode");
    }
    add_layer(new_layer);
}



/**
 * Returns an output stream containing `network` added to the output stream `os`.
 * 
 * The output stream will contain all layers converted to strings, separated by newlines.
 * @param os output stream to export to
 * @param network network to export
 * @return new output stream containing the network's information inside
 */
friend std::ostream& operator<<(std::ostream& os, const Network& network);

////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
//DESTRUCTOR

/**
 * Properly destroys a Network.
 */
~Network() {
    loss_calculator.reset();
    optimizer.reset();
}

};



std::ostream& operator<<(std::ostream& os, const Network& network) {
    
    os << "Network, " << network.n_layers() << " layers:\n";
    for(Layer l : network.layers) {
        os << l << "\n";
    }
    return os;
}