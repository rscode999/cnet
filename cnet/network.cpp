#include "optimizer.cpp"
#include <stdexcept>

/**
 * Thrown to indicate that the network is not in the proper state to call a method.
 */
class illegal_state : public runtime_error {

public:

    /**
     * Creates a new exception with `error_message` as the error message
     * @param error_message the error message to display upon throw
     */
    illegal_state(std::string error_message) : runtime_error(error_message) {
    }

};





/**
 * A neural network that can be trained and used for predictions.
 * 
 * The user adds layers, a loss calculator, and an optimizer to the network prior to use.
 * The input layer is the first layer. The output layer is the last layer. Layers use 0-based indexing.
 * 
 * To use a network, the network must be enabled, by calling the `{networkName}.enable()` method.
 * `enable` checks if the network has valid settings, i.e. layer inputs and outputs are valid.
 * 
 * Components of a network can be changed at any time, provided that it is not enabled.
 * Disable a network with `{networkName}.disable()`.
 * 
 * Important Note: Softmax activations are not allowed, except in the network's final layer.
 */
class Network {

private:


/**
 * Whether the network is enabled (ready for training and evaluation).
 * 
 * A network is operation-ready if the network has at least 1 layer,
 * a loss calculator and optimizer are defined,
 * and the input/output dimensions of each layer are compatible.
 * Also, no layer may have Softmax activation, except for the final layer.
 */
bool enabled;


/**
 * Stores the input vector given to the network
 * 
 * Used in backpropagation.
 */
VectorXd initial_input;

/**
 * Stores outputs from before and after the activation function is applied at each layer.
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
 * Smart pointer to object that improves the model's weights
 */
shared_ptr<Optimizer> optimizer;




public:

/**
 * Creates an empty network.
 * 
 * The created network is not enabled. It has no layers, loss calculator, or optimizer.
 */
Network() {
    enabled = false;
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
 * @throws `out_of_range` if `layer_number` is not on the interval [0, `{network}.layer_count()` - 1]
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
 * @return whether the network is enabled
 */
bool is_enabled() {
    return enabled;
}



/**
 * @return deep copy of the layer at `layer_number` (0-based indexing). Must be between 0 and `{network}.layer_count()`-1
 */
Layer layer_at(int layer_number) {
    assert((layer_number>=0 && layer_number<layer_count() && "Layer number must be between 0 and (number of layers)-1, inclusive on both ends"));
    return layers[layer_number];
}



/**
 * Returns the layer whose name is `layer_name`.
 * 
 * The first matching layer name, i.e. the layer closest to the input layer, is returned.
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
int layer_count() const {
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
 * @param layer_number layer number to access. Must be between 0 and `{networkName}.layer_count()`-1, inclusive on both sides
 * @return weights of layer `layer_number`
 * @throws `out_of_range` if `layer_number` is not a valid index number
 */
MatrixXd weights_at(int layer_number) {
    assert((layer_number>=0 && layer_number<layers.size() && "Layer number must be between 0 and (number of layers)-1"));
    return layers[layer_number].weight_matrix();
}



/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
//SETTERS (MAY NOT BE CALLED IF THE NETWORK IS ENABLED)

/**
 * Adds `new_layer` to the back of the network.
 * 
 * If the network is enabled, this method throws `illegal_state`.
 * 
 * @param new_layer layer to add
 * @throws `illegal_state` if the network is enabled
 */
void add_layer(Layer new_layer) {
    //NO REFERENCES! The user can change the layers if so.

    if(enabled) {
        throw illegal_state("Network must not be enabled to add a layer");
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
 * @throws `illegal_state` if the network is enabled
 */
void add_layer(int input_dimension, int output_dimension, string name="layer") {
    if(enabled) {
        throw illegal_state("Network must not be enabled to add a layer");
    }
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
 * @throws `illegal_state` if the network is enabled
 */
void add_layer(int input_dimension, int output_dimension, shared_ptr<ActivationFunction> activation_function, string name="layer") {
    if(enabled) {
        throw illegal_state("Network must not be enabled to add a layer");
    }
    Layer new_layer = Layer(input_dimension, output_dimension, activation_function, name);
    layers.push_back(new_layer);
}



/**
 * Disables the network, allowing the network to be edited.
 */
void disable() {
    enabled = false;
}



/**
 * Enables the network, allowing training and predictions.
 * 
 * Before being enabled, the network performs a check.
 * 
 * Check: The network must have at least 1 layer, a loss calculator, and an optimizer.
 * The output dimension of each layer must equal the input dimension of the next layer.
 * The only layer that can have Softmax activation is the final (output) layer.
 * 
 * If the check fails, the method throws `illegal_state`.
 * 
 * If the check passes, to prepare for network architecture changes, all training state information is reset.
 * 
 * @throws `illegal_state` (with descriptive error message) if the network state check fails
 */
void enable() {

    //Ensure there are at least 1 layer
    if(layers.size() < 1) {
        throw illegal_state("Enable check failed- The network needs at least 1 layer to begin training");
    }

    //Ensure there is a loss calculator
    if(!loss_calculator) {
        throw illegal_state("Enable check failed- The network must have a loss calculator to begin training");
    }

    //Ensure there is an optimizer
    if(!optimizer) {
        throw illegal_state("Enable check failed- The network must have an optimizer to begin training");
    }

    //Check inputs and outputs of each layer. Also check layers for Softmax activation
    for(int i=0; i<layers.size()-1; i++) {
        //Dimension compatibility
        if(layers[i].output_dimension() != layers[i+1].input_dimension()) {
            throw illegal_state("Enable check failed- Output dimension of layer " + to_string(i) + " (dimension=" + to_string(layers[i].output_dimension()) +
            ") must equal the input dimension of layer " + to_string(i+1) + " (dimension=" + to_string(layers[i+1].input_dimension()) + ")");
        }

        //Softmax activation in non-output layers (the output is the last layer)
        if(layers[i].activation_function()->name() == "softmax") {
            throw illegal_state("Enable check failed- Layer " + to_string(i) + " (\"" + layers[i].name() + "\") is not an output layer, so it cannot have Softmax activation");
        }
    }

    //Check passed: Initialize input vector and reset intermediate outputs
    initial_input = VectorXd(layers[0].input_dimension());
    intermediate_outputs.clear();
    optimizer->clear_state();
    enabled = true;
}



/**
 * Inserts `new_layer` at position `new_pos` in the network.
 * All layers at or after `new_pos` are moved one position backwards.
 * 
 * Inserting at position `{network}.layer_count()` will put the new layer at the end of the network.
 * 
 * Execution time scales linearly with the network's number of layers.
 * 
 * Network layers use 0-based indexing. The first layer is at index 0.
 * 
 * @param new_pos new position number to insert the layer. Must be between 0 and `{network}.layer_count()`, inclusive on both sides
 * @param new_layer layer to insert at position `new_pos`
 * @throws `illegal_state` if this method is called while enabled
 */
void insert_layer_at(int new_pos, Layer new_layer) {
    assert((new_pos>=0 && new_pos<=layer_count() && "New layer insertion position must be between 0 and (number of layers)"));
    if(enabled) {
        throw illegal_state("Cannot insert a new layer while enabled");
    }

    layers.insert(layers.begin() + new_pos, new_layer);
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
 * @throws `illegal_state` if the network is enabled
 * @throws `out_of_range` if no layer's name matches `removal_name`
 */
void remove_layer(string removal_name) {
    if(enabled) {
        throw illegal_state("Cannot remove layers by name while the network is enabled");
    }

    for(int i=0; i<layers.size(); i++) {
        if(layers[i].name() == removal_name) {
            layers.erase(layers.begin() + i);
            return;
        }
    }
    throw out_of_range("No matches found");
}



/**
 * Removes the layer at position `remove_pos`.
 * 
 * @param remove_pos layer number to remove. Must be between 0 and `{network}.layer_count()`-1
 * @throws `illegal_state` if this method is called while the network is enabled
 */
void remove_layer_at(int remove_pos) {
    assert((remove_pos>=0 && remove_pos<layer_count() && "Remove position must be between 0 and (number of layers)-1"));

    if(enabled) {
        throw illegal_state("Cannot remove a layer at a position if the network is enabled");
    }
    layers.erase(layers.begin() + remove_pos);
}



/**
 * Renames the layer at position `rename_pos` to `new_name`.
 * 
 * Unlike most setters, this method can be called, even if the network is enabled.
 * 
 * @param rename_pos layer number to rename (0-based indexing). Must be between 0 and `{network}.layer_count()`-1
 * @param new_name what to remane the layer at `rename_pos` to
 */
void rename_layer_at(int rename_pos, string new_name) {
    assert((rename_pos>=0 && rename_pos<layer_count() && "Rename position must be between 0 and (number of layers)-1"));
    layers[rename_pos].set_name(new_name);
}



/**
 * Sets the activation function at layer `layer_number` to `new_activation_function`.
 * 
 * The input layer's number is 0. The first hidden layer's number is 1.
 * 
 * To remove a layer's activation function, set the layer's activation function to a `shared_ptr<IdentityActivation>`.
 * The IdentityActivation, f(x)=x, is a placeholder that does nothing.
 * 
 * @param layer_number which layer's biases to set (0-based indexing). Must be between 0 and {networkName}.layer_count()-1, inclusive on both ends
 * @param new_activation_function smart pointer to activation function to use 
 * @throws `illegal_state` if the network is enabled
 */
void set_activation_function_at(int layer_number, shared_ptr<ActivationFunction> new_activation_function) {
    assert((layer_number>=0 && layer_number<layers.size() && "To change activation functions, layer number must be between 0 and (number of layers)-1"));

    if(enabled) {
        throw illegal_state("Cannot change activation functions while the network is enabled");
    }

    layers[layer_number].set_activation_function(new_activation_function);
}



/**
 * Sets the bias vector at layer `layer_number` to `new_biases`.
 * 
 * The input layer's number is 0. The first hidden layer's number is 1.
 * 
 * @param layer_number which layer's biases to set (0-based indexing). Must be between 0 and {networkName}.layer_count()-1, inclusive on both ends
 * @param new_biases new bias vector to set. Must have {selected layer}.output_dimension() rows
 * @throws `illegal_state` if the network is enabled
 */
void set_biases_at(int layer_number, VectorXd new_biases) {
    assert((layer_number>=0 && layer_number<layers.size() && "For changing bias vectors, layer number must be between 0 and (number of layers)-1"));
    assert((new_biases.rows() == layers[layer_number].output_dimension() && "New bias vector's number of rows must equal the selected layer's output dimension"));
    assert((new_biases.cols() == 1 && "New biases must be a column vector"));

    if(enabled) {
        throw illegal_state("Cannot manually set bias vectors while the network is enabled");
    }

    layers[layer_number].set_bias_vector(new_biases);
}



/**
 * Replaces the current loss calculator with `new_calculator`.
 * If no loss calculator is defined yet, the defined loss calculator becomes `new_calculator`.
 * 
 * If the network is enabled, this method throws `illegal_state`.
 * 
 * @param new_calculator smart pointer to loss calculator object
 * @throws `illegal_state` if the network is enabled
 */
void set_loss_calculator(shared_ptr<LossCalculator> new_calculator) {
    if(enabled) {
        throw illegal_state("Network must not be enabled to update the loss calculator");
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
 * If the network is enabled, this method throws `illegal_state`.
 * 
 * @param new_optimizer smart pointer to desired optimizer object
 * @throws `illegal_state` if the network is enabled
 */
void set_optimizer(shared_ptr<Optimizer> new_optimizer) {
    if(enabled) {
        throw illegal_state("Network must not be enabled to update the optimizer");
    }

    //Free any existing optimizer
    if(optimizer) {
        optimizer.reset();
    }
    optimizer = new_optimizer;
}



/**
 * Sets the weights at layer `layer_number` to `new_weights`.
 * 
 * The input layer's number is 0. The first hidden layer's number is 1.
 * 
 * @param layer_number which layer's weights to set (0-based indexing). Must be between 0 and {networkName}.layer_count()-1, inclusive on both ends
 * @param new_weights new weight matrix for layer `layer_number`.
 * Number of rows must equal {selected layer}.output_dimension(), number of columns must equal {selected layer}.input_dimension()
 * @throws `illegal_state` if the network is enabled
 */
void set_weights_at(int layer_number, MatrixXd new_weights) {
    assert((layer_number>=0 && layer_number<layers.size() && "Layer number must be between 0 and (number of layers)-1"));
    assert((new_weights.rows() == layers[layer_number].output_dimension() && "New weight matrix's row count must equal the layer's output dimension"));
    assert((new_weights.cols() == layers[layer_number].input_dimension() && "New weight matrix's column count must equal the layer's input dimension"));
   
    if(enabled) {
        throw illegal_state("Cannot manually set weight matrices when the network is enabled");
    }

    layers[layer_number].set_weight_matrix(new_weights);
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//METHODS (MAY BE CALLED ONLY IF THE NETWORK IS ENABLED)


/**
 * Returns the result of the feed-forward operation on the given input, i.e. the network's predictions for `input`.
 * 
 * If `training` is true, the network internally records layer outputs for backpropagation.
 * After using the method with `training`=true, the `reverse` method can be called.
 * 
 * Requires that the network is enabled. If not, the method throws `illegal_state`.
 * 
 * @param input input to the network. Must have `{networkName}.input_dimension()` rows
 * @param training true if  training the network, false if getting results for evaluation only. Default: `true`
 * @return the network's output, as a VectorXd of dimension `{networkName}.output_dimension()`
 * @throws `illegal_state` if the network is not enabled
 */
VectorXd forward(const VectorXd& input, bool training = true) {
    assert((input.cols() == 1 && "Input to forward operation must be a column vector"));
    assert((input.rows() == input_dimension() && "Input to forward operation must have same dimension as the network's input"));
    
    //Enable check
    if(!enabled) {
        throw illegal_state("Forward operation requires the network to be enabled");
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
        current_layer_output = layers[i].activation_function()->compute(pre_activation);

        if(training) {
            intermediate_outputs.push_back({pre_activation, current_layer_output});
        }
    }

    return current_layer_output;
}



/**
 * Returns the network's predictions for `input`.
 * 
 * When this method is used, the network *does not* internally record intermediate layer outputs for backpropagation.
 * 
 * Equivalent to `{networkName}.forward(input, false)`.
 * 
 * Requires that the network is enabled.
 * 
 * @param input input to the network. Must have `{networkName}.input_dimension()` rows
 * @return the network's output, as a VectorXd of dimension `{networkName}.output_dimension()`
 * @throws `illegal_state` if the network is not enabled
 */
VectorXd predict(const VectorXd& input) {
    assert((input.cols() == 1 && "Input to prediction must be a column vector"));
    assert((input.rows() == input_dimension() && "Input to prediction must have same dimension as the network's input"));

    //Enable check
    if(!enabled) {
        throw illegal_state("Predict operation requires the network to be enabled");
    }

    return forward(input, false);
}



/**
 * Updates the weights and biases of this network using `predictions` and `actuals`, using the network's optimizer.
 * 
 * This method requires the network to be enabled. Also, `{networkName}.forward` with `training`=true must have been called since the network was enabled.
 * If these conditions are not met, the method throws `illegal_state`.
 * 
 * @param predictions what the network predicts for a given input
 * @param actuals expected output for the network's prediction
 * @throws `illegal_state` if the network is not enabled, or a feed-forward training operation was not done
 */
void reverse(const VectorXd& predictions, const VectorXd& actuals) {
    assert((predictions.cols() == 1 && "Reverse process predictions must be a column vector"));
    assert((actuals.cols() == 1 && "Reverse process actuals must be a column vector"));
    assert((predictions.rows() == output_dimension() && "Reverse process predictions length must equal network's output dimension"));
    assert((actuals.rows() == output_dimension() && "Reverse process actuals length must equal network's output dimension"));

    //Enable check
    if(!enabled) {
        throw illegal_state("Reverse operation requires the network to be enabled");
    }
    
    //Ensure that there are intermediate outputs
    if(intermediate_outputs.size() <= 0) {
        throw illegal_state("Reverse operation requires that the `forward` method (with training = true) was previously used");
    }
    
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
 * If the network is enabled, this method throws `illegal_state`.
 * 
 * Equivalent to `{networkName}.add_layer(new_layer)`.
 * 
 * @param new_layer layer to add
 * @throws `illegal_state` if the network is enabled
 */
void operator+=(Layer new_layer) {
    if(enabled) {
        throw illegal_state("(+ operator) Cannot add layer while the network is enabled");
    }
    add_layer(new_layer);
}



/**
 * Exports `network` to the output stream `os`, returning a reference to `os` with `network` added.
 * 
 * The output stream will contain all layers converted to strings, separated by newlines.
 * 
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
    
    //enabled or disabled, layer count
    os << "Network (" << (network.enabled ? "enabled" : "disabled") << "), ";
    os << network.layer_count() << " layers, ";

    //loss calculator
    if(network.loss_calculator) {
        os << network.loss_calculator->name() << " loss, ";
    }
    else {
        os << "no defined loss, ";
    }

    //optimizer
    if(network.optimizer) {
        os << network.optimizer->name() << " optimizer";
    }
    else {
        os << "no defined optimizer";
    }

    //layers
    if(network.layers.size() > 0) {
        os << ":\n";
        for(Layer l : network.layers) {
            os << l << "\n";
        }
    }
    return os;
}