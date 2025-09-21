#include "layer.cpp"
#include "loss_calculator.cpp"

#include <vector>



namespace CNet {




/**
 * Abstract class for network optimizers. 
 * 
 * Its `clear_state` and `step` methods are private and cannot be directly used.
 */
class Optimizer {

friend class Network;

public:

    /**
     * @return the optimizer's identifying std::string. If not overridden, returns `"optimizer"`.
     */
    virtual std::string name() {
        return "optimizer";
    }

    /**
     * Sets the optimizer's hyperparameters.
     * The purpose of each index in `hyperparameters` depends on the optimizer.
     * 
     * Example: For SGD optimizers, index 0 is the new learning rate, and index 1 is for the new momentum coefficient.
     * 
     * @param hyperparameters vector of new hyperparameters to set (exact parameters depends on the optimizer)
     */
    virtual void set_hyperparameters(const std::vector<double>& hyperparameters) = 0;

    /**
     * @return detailed information about the optimizer, including hyperparameters
     */
    virtual std::string to_string() {
        return "optimizer";
    }

private:

    /**
     * Resets the optimizer's internal state, allowing it to handle network architecture changes.
     * 
     * Called internally by a Network when the Network is disabled.
     */
    virtual void clear_state() = 0;

    /**
     * Updates `layers` in-place using the optimizer's method, using calculated gradients.
     * 
     * This is a private method. Not intended to be called by a user.
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
    virtual void step(std::vector<Layer>& layers, const Eigen::VectorXd& initial_input, const std::vector<LayerCache>& intermediate_outputs,
        const Eigen::VectorXd& predictions, const Eigen::VectorXd& actuals, const std::shared_ptr<LossCalculator> loss_calculator) = 0;
    

public:
    /**
     * Properly destroys an Optimizer.
     */
    virtual ~Optimizer() = default;
};




/**
 * A Stochastic Gradient Descent (SGD) optimizer
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
     * Learning rate, dictating the optimization step size. Must be positive.
     */
    double learn_rate;

    /**
     * Amount of momentum to use
     */
    double momentum_coeff;

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

    /**
     * Creates a new SGD optimizer, loading it with the given hyperparameters `learning_rate` and `momentum_coefficient`.
     * 
     * @param learning_rate learning rate, for determining speed of convergence.  Default 0.01. Must be positive
     * @param momentum_coefficient for determining amount of momentum to use. Default 0. Cannot be negative
     */
    SGD(double learning_rate = 0.01, double momentum_coefficient = 0) {
        assert((learning_rate>0 && "Learning rate must be positive"));
        assert((momentum_coefficient>=0 && "Momentum coefficient cannot be negative"));
        learn_rate = learning_rate;
        momentum_coeff = momentum_coefficient;
    }   

    
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //GETTERS

    
    /**
     * @return the SGD optimizer's learning rate
     */
    double learning_rate() {
        return learn_rate;
    }


    /**
     * @return the SGD optimizer's momentum coefficient
     */
    double momentum_coefficient() {
        return momentum_coeff;
    }


    /**
     * @return `"sgd"`, the optimizer's identifying string
     */
    std::string name() override {
        return "sgd";
    }


    /**
     * @return string containing the optimizer's name, its learning rate, and its momentum coefficient
     */
    std::string to_string() override {
        return "sgd, learning rate=" + std::to_string(learn_rate) + ", momentum coefficient=" + std::to_string(momentum_coeff);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //SETTERS

    /**
     * Sets the SGD optimizer's hyperparameters to `hyperparameters`.
     * Index 0 contains the new learning rate. Index 1 contains the new momentum coefficient.
     * 
     * Index 0 must be positive. Index 1 must be non-negative.
     * 
     * @param hyperparameters std::vector of new hyperparameters. Must be of length 2, where index 0 is positive and index 1 is non-negative
     */
    void set_hyperparameters(const std::vector<double>& hyperparameters) override {
        assert((hyperparameters.size() == 2 && "Hyperparameter list for SGD optimizer must be of length 2"));
        assert((hyperparameters[0]>0 && "Index 0 of new SGD hyperparameters (learning rate) must be positive"));
        assert((hyperparameters[1]>=0 && "Index 1 of new SGD hyperparameters (momentum coefficient) must be non-negative"));

        learn_rate = hyperparameters[0];
        momentum_coeff = hyperparameters[1];
    }


    /**
     * Sets the optimizer's learning rate to `new_learning_rate`.
     * @param new_learning_rate learning rate to use. Must be positive
     */
    void set_learning_rate(double new_learning_rate) {
        assert((new_learning_rate>0 && "New learning rate must be positive"));
        learn_rate = new_learning_rate;
    }

    
    /**
     * Sets the optimizer's momentum coefficient to `new_momentum_coefficient`.
     * @param new_momentum_coefficient momentum coefficient to use. Cannot be negative
     */
    void set_momentum_coefficient(double new_momentum_coefficient) {
         assert((new_momentum_coefficient>=0 && "New momentum coefficient cannot be negative"));
         momentum_coeff = new_momentum_coefficient;
    }


private:
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //METHODS (PRIVATE)

    /**
     * Removes the SGD optimizer's weight and bias velocities, preparing it to handle a different network architecture.
     */
    void clear_state() override {
        momentum_cache.clear();
    }


    /**
     * Updates `layers` using SGD optimization
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
                do backpropagation
                update new delta with product of current weights and previous intermediate output 
                (with activation deriv. applied to each element)
                */
                new_delta = (current_weights.transpose() * delta).cwiseProduct(current_intermediate_output);
            }
                
            //weight gradient = outer product of delta and previous layer's post activation outputs
            Eigen::MatrixXd weight_grad = delta * previous_post_activation.transpose();
            //bias gradient = delta as itself
            Eigen::VectorXd bias_grad   = delta;

            //get current momentums for weights and biases
            Eigen::MatrixXd& weight_velocity = momentum_cache[l].weight_velocity;
            Eigen::VectorXd& bias_velocity   = momentum_cache[l].bias_velocity;

            //update weight
            weight_velocity = momentum_coeff * weight_velocity + learn_rate * weight_grad;
            bias_velocity   = momentum_coeff * bias_velocity + learn_rate * bias_grad;

            //update layer weights
            layers[l].set_weight_matrix(layers[l].weight_matrix() - weight_velocity);
            layers[l].set_bias_vector((layers[l].bias_vector() - bias_velocity).eval());
            
            //update delta
            if(l > 0) {
                delta = new_delta;
            }
        }
    }

};




}