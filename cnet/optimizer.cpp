#include "layer.cpp"
#include "loss_calculator.cpp"

#include <vector>



/**
 * Abstract class for network optimizers. 
 * 
 * Its `clear_state` and `step` methods are private and cannot be directly used.
 */
class Optimizer {

friend class Network;

public:

    /**
     * @return the optimizer's identifying string. If not overridden, returns `"optimizer"`.
     */
    virtual string name() {
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
     * Updates `layers` using the optimizer's method.
     * 
     * Used internally by a Network. Not intended to be called by a user.
     * 
     * @param layers vector of layers to optimize
     * @param initial_input value that was first given to the network
     * @param intermediate_outputs outputs of each layer before and after the layer's activation function is applied
     * @param predictions the output of the network for `initial_input`
     * @param actuals what the network should predict for `initial_input`
     * @param loss_calculator smart pointer to loss calculator object
     */
    virtual void step(vector<Layer>& layers, const VectorXd& initial_input, const vector<LayerCache>& intermediate_outputs,
        const VectorXd& predictions, const VectorXd& actuals, const shared_ptr<LossCalculator> loss_calculator) = 0;
    

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
        MatrixXd weight_velocity;
        VectorXd bias_velocity;
        
        MomentumCache(int weight_rows, int weight_cols, int bias_size) {
            weight_velocity = MatrixXd::Zero(weight_rows, weight_cols);
            bias_velocity = VectorXd::Zero(bias_size);
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
    vector<MomentumCache> momentum_cache;


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
    VectorXd softmax_jacobian_vector_product(const VectorXd& softmax_out, const VectorXd& loss_grad) {
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


    /**
     * @return `"sgd"`, the optimizer's identifying string
     */
    string name() override {
        return "sgd";
    }



private:

    /**
     * Removes the SGD optimizer's weight and bias velocities, preparing it to handle a different network architecture.
     */
    void clear_state() override {
        momentum_cache.clear();
    }


    /**
     * Updates `layers` using SGD optimization
     * 
     * @param layers vector of layers to optimize
     * @param initial_input value that was first given to the network
     * @param intermediate_outputs outputs of each layer before and after the layer's activation function is applied
     * @param predictions the output of the network for `initial_input`
     * @param actuals what the network should predict for `initial_input`
     * @param loss_calculator smart pointer to loss calculator object
     */
    void step(vector<Layer>& layers, const VectorXd& initial_input, const vector<LayerCache>& intermediate_outputs,
        const VectorXd& predictions, const VectorXd& actuals, const shared_ptr<LossCalculator> loss_calculator) override {
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


        VectorXd delta;
        auto final_activation = layers.back().activation_function();
        bool final_activation_using_softmax = final_activation->name() == "softmax";
        bool using_cross_entropy_loss = loss_calculator->name() == "cross_entropy";

        // Step 1: Compute dL/dy
        VectorXd loss_grad = loss_calculator->compute_loss_gradient(predictions, actuals);

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
            VectorXd previous_post_activation = (l > 0) 
                ? intermediate_outputs[l-1].post_activation
                : initial_input;

            //propagate delta
            VectorXd new_delta;
            if(l > 0) {
                MatrixXd current_weights = layers[l].weight_matrix();

                VectorXd current_intermediate_output = layers[l-1].activation_function()->using_pre_activation()
                    ? intermediate_outputs[l-1].pre_activation
                    : intermediate_outputs[l-1].post_activation;
                
                //apply previous layer's activation function derivative to each intermediate output element
                current_intermediate_output = layers[l-1].activation_function()->compute_derivative(current_intermediate_output);

                /*
                do backpropagation
                update new delta with product of current weights and previous intermediate output 
                (with activation deriv. applied to each element)
                */
                new_delta = (current_weights.transpose() * delta).cwiseProduct(
                    current_intermediate_output
                );
            }
            
            //weight gradient = outer product of delta and previous layer's post activation outputs
            MatrixXd weight_grad = delta * previous_post_activation.transpose();
            //bias gradient = delta as itself
            VectorXd bias_grad   = delta;

            //get current momentums for weights and biases
            MatrixXd& weight_velocity = momentum_cache[l].weight_velocity;
            VectorXd& bias_velocity   = momentum_cache[l].bias_velocity;

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
