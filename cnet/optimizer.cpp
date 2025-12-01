#include "layer.cpp"
#include "loss_calculator.cpp"

#include <list>
#include <vector>



namespace CNet {




/**
 * Abstract class for network optimizers.
 * 
 * The Optimizer's `step` method performs an optimization pass over a Network's layers.
 * `step` is a private method that can be called only from within a Network, and is inaccessible to users.
 * 
 * An Optimizer's hyperparameters can be adjusted with optimizer-specific setter methods,
 * or use the `set_hyperparameters` method.
 * 
 * Normally used in a Network when inside a smart pointer, specifically a `std::shared_ptr`.
 */
class Optimizer {

friend class Network;

public:

    /**
     * @return std::vector containing the optimizer hyperparameters, in the order required by `set_hyperparameters`
     */
    virtual std::vector<double> hyperparameters() const = 0;

    /**
     * @return the optimizer's identifying std::string.
     */
    virtual std::string name() const {
        return "optimizer";
    }

    /**
     * Sets the optimizer's hyperparameters.
     * The purpose of each index in `hyperparameters` depends on the optimizer.
     * 
     * Example: For SGD optimizers, index 0 is the new learning rate, index 1 is for the new momentum coefficient,
     * and index 2 is for the new batch size.
     * 
     * @param hyperparameters vector of new hyperparameters to set (exact parameters depends on the optimizer)
     */
    virtual void set_hyperparameters(const std::vector<double>& hyperparameters) = 0;

    /**
     * @return detailed information about the optimizer, including hyperparameters
     */
    virtual std::string to_string() const {
        return "optimizer";
    }

private:

    /**
     * Resets the optimizer to its pre-training state.
     * 
     * The reset allows the Optimizer to handle network architecture changes.
     * 
     * Called internally by a Network when the Network is disabled.
     */
    virtual void clear_state() = 0;

    /**
     * Updates `layers` in-place using the optimizer, using calculated gradients.
     * 
     * This is a private method. Not intended to be called by a user.
     * 
     * Mutates `layers` and this Optimizer object.
     * 
     * @param layers std::vector of layers to optimize
     * @param initial_input input value of the network
     * @param intermediate_outputs outputs of each layer before and after the layer's activation function is applied
     * @param predictions the output of the network for `initial_input`
     * @param actuals what the network should predict for `initial_input`
     * @param loss_calculator smart pointer to loss calculator object
     */
    virtual void step(std::vector<Layer>& layers, const Eigen::VectorXd& initial_input, const std::vector<LayerCache>& intermediate_outputs,
        const Eigen::VectorXd& predictions, const Eigen::VectorXd& actuals, const std::shared_ptr<LossCalculator> loss_calculator) = 0;
    

public:
    /**
     * Properly destroys an Optimizer.
     */
    virtual ~Optimizer() = default;
};





/**
 * A Stochastic Gradient Descent (SGD) optimizer, capable of training in batches.
 *
 * The optimizer has an adjustable learning rate and momentum coefficient.
 *   
 * The optimizer updates weights and biases on every `batch_size`-th input.
 * No updates occur on inputs other than ever `batch_size`-th input.
 */
class SGD : public Optimizer {

friend class Network;


private:

    /**
     * Private struct to store weight and bias velocities for momentum SGD
     */
    struct MomentumCache {
        Eigen::MatrixXd weight_velocity;
        Eigen::VectorXd bias_velocity;
        
        MomentumCache(int weight_rows, int weight_cols, int bias_size) {
            weight_velocity = Eigen::MatrixXd::Zero(weight_rows, weight_cols);
            bias_velocity = Eigen::VectorXd::Zero(bias_size);
        }
    };


    /**
     * Number of samples to train per batch. Must be positive.
     * 
     * The name avoids a conflict with the method `batch_size`.
     */
    int batch_size_;

    /**
     * Learning rate, dictating the optimization step size. Must be positive.
     */
    double learn_rate;

    /**
     * Amount of momentum to use. Must be non-negative.
     */
    double momentum_coeff;

    /**
     * Number of samples trained so far. 
     * 
     * When this quantity reaches `batch_size`-1, the given layers are updated.
     * 
     * Must be between 0 and `batch_size`-1.
     */
    int n_samples_trained;

    /**
     * Holds per-layer biases accumulated over training.
     * 
     * Each index holds the total bias for the given layer.
     */
    std::list<Eigen::VectorXd> total_biases;

    /**
     * Holds per-layer weights accumulated over training.
     * 
     * Each index holds the total bias for the given layer.
     */
    std::list<Eigen::MatrixXd> total_weights;


    /**
     * Holds previous matrices and vectors used in backpropagation. For momentum.
     */
    std::vector<MomentumCache> momentum_cache;


    /**
     * Returns the product of a softmax output's Jacobian matrix 
     * with a gradient vector.
     * 
     * Does not explicitly calculate the softmax Jacobian.
     * 
     * Used when a layer uses softmax activation, but cross-entropy loss is not used.
     * 
     * @param softmax_out the softmax layer's output
     * @param loss_grad the gradient of the loss with respect to the softmax output
     * @return Jacobian from `softmax_output` * `loss_grad`
     */
    Eigen::VectorXd softmax_jacobian_vector_product(const Eigen::VectorXd& softmax_out, const Eigen::VectorXd& loss_grad) {
        double dot = softmax_out.dot(loss_grad);
        return softmax_out.array() * (loss_grad.array() - dot);
    }

public:


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //CONSTRUCTOR

    /**
     * Creates a new SGD optimizer, loading it with the given hyperparameters `learning_rate`, `momentum_coefficient`, and `batch_size`.
     * 
     * @param learning_rate learning rate, for determining speed of convergence. Must be positive. Default 0.01
     * @param momentum_coefficient for determining amount of momentum to use. Cannot be negative. Default 0
     * @param batch_size number of datapoints to use in one batch. Must be positive. Default 1
     */
    SGD(double learning_rate = 0.01, double momentum_coefficient = 0, int batch_size = 1) {
        assert((learning_rate>0 && "Learning rate must be positive"));
        assert((momentum_coefficient>=0 && "Momentum coefficient cannot be negative"));
        assert((batch_size>0 && "Batch size must be positive"));
        
        learn_rate = learning_rate;
        momentum_coeff = momentum_coefficient;
        batch_size_ = batch_size;
        n_samples_trained = 0;
    }



    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //GETTERS

    /**
     * @return optimizer's batch size
     */
    int batch_size() const {
        return batch_size_;
    }

    /**
     * @return vector of 3 hyperparameters: learning rate (index 0), momentum coefficient (index 1), batch size (index 2)
     */
    std::vector<double> hyperparameters() const override {
        return {learn_rate, momentum_coeff, (double)batch_size_};
    }


    /**
     * @return optimizer's learning rate
     */
    double learning_rate() const {
        return learn_rate;
    }


    /**
     * @return optimizer's momentum coefficient
     */
    double momentum_coefficient() const {
        return momentum_coeff;
    }

    
    /**
     * @return `"sgd"`, the optimizer's identifying string
     */
    std::string name() const override {
        return "sgd";
    }


    /**
     * @return string containing the optimizer's name, learning rate, momentum coefficient, and batch size
     */
    std::string to_string() const override {
        return "sgd, learning rate=" + std::to_string(learn_rate) + ", momentum coefficient=" + std::to_string(momentum_coeff) 
            + ", batch size=" + std::to_string(batch_size_);
    }


    ////////////////////////////////////////////////////////////////////////////////////
    //SETTERS


    /**
     * Sets the SGD's batch size to `new_batch_size`.
     * 
     * The optimizer's current training data will be reset.
     * 
     * @param new_batch_size new batch size. Must be positive.
     */
    void set_batch_size(int new_batch_size) {
        assert((new_batch_size>0 && "SGD new batch size must be positive"));

        clear_state();
        batch_size_ = new_batch_size;
    }


    /**
     * Sets the SGD optimizer's hyperparameters to `hyperparameters`.
     * Index 0 contains the new learning rate. Index 1 contains the new momentum coefficient. Index 2 contains the new batch size.
     * 
     * If the batch size is changed, the optimizer's training data will be reset.
     * 
     * @param hyperparameters vector of new hyperparameters. Must be of length 3, where index 0 is positive, index 1 is non-negative, and index 2 (rounded down to the nearest integer) is positive
     */
    void set_hyperparameters(const std::vector<double>& hyperparameters) override {
        assert((hyperparameters.size() == 3 && "SGD optimizer hyperparameter list must be of length 3"));
        assert((hyperparameters[0]>0 && "SGD hyperparameter index 0 (new learning rate) must be positive"));
        assert((hyperparameters[1]>=0 && "SGD hyperparameter index 1 (new momentum coefficient) must be positive"));
        assert(((int)hyperparameters[2]>0 && "SGD hyperparameter index 2 (new batch size), as an integer, must be positive"));

        if(batch_size_ != (int)hyperparameters[2]) {
            clear_state();
        }

        learn_rate = hyperparameters[0];
        momentum_coeff = hyperparameters[1];
        batch_size_ = (int)hyperparameters[2];
    }


    /**
     * Sets the optimizer's learning rate to `new_learning_rate`.
     * @param new_learning_rate new rate. Must be positive.
     */
    void set_learning_rate(double new_learning_rate) {
        assert(new_learning_rate > 0 && "SGD new learning rate must be positive");
        learn_rate = new_learning_rate;
    }


    /**
     * Sets the optimizer's momentum coefficient to `new_momentum_coefficient`.
     * @param new_learning_rate new momentum coefficient. Cannot be negative.
     */
    void set_momentum_coefficient(double new_momentum_coefficient) {
        assert(new_momentum_coefficient >= 0 && "SGD new momentum coefficient must be non-negative");
        momentum_coeff = new_momentum_coefficient;
    }

private:

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //METHODS (PRIVATE)

    /**
     * Resets the optimizer to its pre-training state.
     * 
     * Resets biases, weights, momentum data, and number of samples trained.
     */
    void clear_state() override {
        total_biases.clear();
        total_weights.clear();
        n_samples_trained = 0;
        momentum_cache.clear();
    }


    /**
     * Updates `layers` using SGD optimization.
     * 
     * Mutates `layers`.
     * 
     * @param layers std::vector of layers to optimize
     * @param initial_input value that was first given to the network
     * @param intermediate_outputs outputs of each layer before and after the layer's activation function is applied
     * @param predictions the output of the network for `initial_input`
     * @param actuals what the network should predict for `initial_input`
     * @param loss_calculator smart pointer to loss calculator object
     */
    void step(std::vector<Layer>& layers, const Eigen::VectorXd& initial_input, const std::vector<LayerCache>& intermediate_outputs,
        const Eigen::VectorXd& predictions, const Eigen::VectorXd& actuals, const std::shared_ptr<LossCalculator> loss_calculator) override {
            
        //The entire network is not passed in. This allows one-way friend access
        
        assert((predictions.cols() == 1 && "Predicted values must be a column vector"));
        assert((predictions.rows() == layers.back().output_dimension() && "Predicted value vector must have dimension equal to the network's output dimension"));
        assert((actuals.cols() == 1 && "Actual values must be a column vector"));
        assert((actuals.rows() == layers.back().output_dimension() && "Actual value vector must have dimension equal to the network's output dimension"));

        //Idiot check
        assert(intermediate_outputs.size() == layers.size());

        //Initialize momentums to all 0's (if not already initialized)
        if(momentum_cache.size() == 0) {
            for(Layer& layer : layers) {
                momentum_cache.emplace_back(layer.output_dimension(), layer.input_dimension(), layer.output_dimension());
            }
        }


        Eigen::VectorXd delta;
        auto final_activation = layers.back().activation_function();
        bool final_activation_using_softmax = final_activation->name() == "softmax";
        bool using_cross_entropy_loss = loss_calculator->name() == "cross_entropy";

        // Step 1: Compute dL/dy
        Eigen::VectorXd loss_grad = loss_calculator->compute_loss_gradient(predictions, actuals);
 
        // Step 2: Handle softmax Jacobian if needed
        bool activation_derivative_applied = false; //Ensures that, if softmax Jacobian is applied, it isn't applied again
        if (final_activation_using_softmax && !using_cross_entropy_loss) {
            // This is softmax + non-cross-entropy (e.g., MSE)
            delta = softmax_jacobian_vector_product(predictions, loss_grad);
            activation_derivative_applied = true;
        } 
        else {
            // For cross-entropy + softmax or any other case
            delta = loss_grad;
        }

        // Step 3: Apply activation derivative if applicable
        if (!(final_activation_using_softmax && using_cross_entropy_loss)
            && !activation_derivative_applied) { //Only do this step if the softmax Jacobian is not already applied
                
            if (final_activation->using_pre_activation()) {
                delta = delta.cwiseProduct(final_activation->compute_derivative(intermediate_outputs.back().pre_activation));
            } 
            else {
                delta = delta.cwiseProduct(final_activation->compute_derivative(intermediate_outputs.back().post_activation));
            }
        }
                
   
        //update remaining layers
        auto biases_reverse_iterator = total_biases.rbegin();
        auto weights_reverse_iterator = total_weights.rbegin();

        for(int l = layers.size()-1; l >= 0; l--) {
            //Get original post-activation of the previous layer
            Eigen::VectorXd previous_post_activation = (l > 0) 
                ? intermediate_outputs[l-1].post_activation
                : initial_input;

            //propagate delta
            Eigen::VectorXd new_delta;
            if(l > 0) {
                Eigen::MatrixXd current_weights = layers[l].weight_matrix();

                Eigen::VectorXd current_intermediate_output = layers[l-1].activation_function()->using_pre_activation()
                    ? intermediate_outputs[l-1].pre_activation
                    : intermediate_outputs[l-1].post_activation;
                
                //apply previous layer's activation function derivative to each intermediate output element
                current_intermediate_output = layers[l-1].activation_function()->compute_derivative(current_intermediate_output);

                /*
                update new delta with product of current weights and previous intermediate output 
                (with activation deriv. applied to each element)
                */
                new_delta = (current_weights.transpose() * delta).cwiseProduct(current_intermediate_output);
            }
                
            //weight gradient = outer product of delta and previous layer's post activation outputs
            Eigen::MatrixXd weight_grad = delta * previous_post_activation.transpose();
            //bias gradient = delta as itself
            Eigen::VectorXd bias_grad   = delta;

            //Put the gradients in the storage, if none are in there yet
            if(n_samples_trained == 0) {
                total_weights.push_front(weight_grad);
                total_biases.push_front(bias_grad);
            }
            //If there are gradients, add to the existing gradients
            else {
                *biases_reverse_iterator = *biases_reverse_iterator + bias_grad;
                *weights_reverse_iterator = *weights_reverse_iterator + weight_grad;
            }
            
            //update delta
            if(l > 0) {
                delta = new_delta;
            }

            //move the iterators backward
            biases_reverse_iterator++;
            weights_reverse_iterator++;
        }


        //update layer weights and biases, if enough training examples are done already
        if(n_samples_trained == batch_size_-1) {
            //Counts current layer
            int current_layer = layers.size()-1;
            //Iterates through weights
            auto weight_iterator = total_weights.rbegin();
            //Iterates through biases
            auto bias_iterator = total_biases.rbegin();

            while(weight_iterator != total_weights.rend()) {
                //divide average weights and biases by batch size
                Eigen::MatrixXd average_weights = *weight_iterator * (1.0 / batch_size_);
                Eigen::VectorXd average_biases = *bias_iterator * (1.0 / batch_size_);

                //get current momentums for weights and biases
                Eigen::MatrixXd& weight_velocity = momentum_cache[current_layer].weight_velocity;
                Eigen::VectorXd& bias_velocity   = momentum_cache[current_layer].bias_velocity;

                //update weight and bias velocities
                weight_velocity = momentum_coeff * weight_velocity - learn_rate * average_weights;
                bias_velocity   = momentum_coeff * bias_velocity - learn_rate * average_biases;
                //THESE ARE NOT THE GRADIENTS!
                
                //update the layers
                layers[current_layer].set_weight_matrix(layers[current_layer].weight_matrix() + weight_velocity);
                layers[current_layer].set_bias_vector((layers[current_layer].bias_vector() + bias_velocity).eval());
                
                weight_iterator++;
                bias_iterator++;
                current_layer--;
            }
        }

        //Update number of samples trained
        n_samples_trained++;
        if(n_samples_trained >= batch_size_) {
            total_biases.clear();
            total_weights.clear();
            n_samples_trained = 0;
        }
    }

};



/**
 * Returns a std::shared_ptr to the optimizer whose name is `name` (i.e. "sgd") and specified by the hyperparameters given in `hyperparameters`.
 * 
 * @param name name of desired optimizer
 * @param hyperparameters list of hyperparameters for the new optimizer. Must be accepted by the optimizer's `set_hyperparameters` method
 * @return optimizer with matching name
 * @throws `runtime_error` if no matching optimizer name is found
 */
std::shared_ptr<Optimizer> make_optimizer(const std::string& name, const std::vector<double>& hyperparameters) {
    using namespace std;

    if(name == "sgd") {
        assert(hyperparameters.size() == 3 && "SGD creation requires 3 hyperparameters");
        assert((int)hyperparameters[2] > 0 && "Batch size of hyperparameters for SGD must be positive");
        shared_ptr<SGD> out = make_shared<SGD>(hyperparameters[0], hyperparameters[1], (int)hyperparameters[2]);
        return out;
    }
    throw runtime_error("optimizer name not recognized");
}




}