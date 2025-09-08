#include "layer.cpp"
#include "loss_calculator.cpp"

#include <vector>



/**
 * Interface (fully abstract class) for network optimizers. Cannot be directly used.
 */
class Optimizer {
public:

    /**
     * @return the optimizer's identifying string. If not overridden, returns `"optimizer"`.
     */
    virtual string identifier() {
        return "optimizer";
    }

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
        const MatrixXd& predictions, const MatrixXd& actuals, const shared_ptr<LossCalculator> loss_calculator) = 0;
    
    /**
     * Properly destroys an Optimizer.
     */
    virtual ~Optimizer() = default;
};



/**
 * A Stochastic Gradient Descent (SGD) optimizer
 */
class SGD : public Optimizer {

private:
    /**
     * Struct to store weight and bias velocities for momentum SGD
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
     * Holds matrices and vectors 
     */
    vector<MomentumCache> momentum_cache;

public:

    /**
     * Creates a new SGD optimizer
     * 
     * @param learning_rate learning rate, for determining speed of convergence.  Default 0.01. Must be positive
     * @param momentum_coefficient for determining amount of momentum to use. Default 0.9. Cannot be negative
     */
    SGD(double learning_rate = 0.01, double momentum_coefficient = 0.9) {
        assert((learning_rate>0 && "Learning rate must be positive"));
        assert((momentum_coefficient>=0 && "Momentum coefficient cannot be negative"));
        learn_rate = learning_rate;
        momentum_coeff = momentum_coefficient;
    }   


    /**
     * @return `"sgd"`, the optimizer's identifying string
     */
    string identifier() override {
        return "sgd";
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
        const MatrixXd& predictions, const MatrixXd& actuals, const shared_ptr<LossCalculator> loss_calculator) override {
        
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

        VectorXd delta = loss_calculator->compute_loss_gradient(predictions, actuals);
        //or otherwise use the loss function's derivative
        
        //Get final layer's activation function
        shared_ptr<ActivationFunction> final_activation_function = layers.back().activation_function();


        //Component-wise multiply to the element-wise differentiated final bias vector
        
        //Pre-activation 
        if(final_activation_function->using_pre_activation()) {
            delta = delta.cwiseProduct(
                intermediate_outputs.back().pre_activation.unaryExpr(
                    //Call the final layer's activation function's derivative on each element
                    [&final_activation_function](double x) {
                        return final_activation_function->compute_derivative(x);
                    }
                )
            );
        }
        //Post-activation
        else {
            delta = delta.cwiseProduct(
                intermediate_outputs.back().post_activation.unaryExpr(
                    //Call the final layer's activation function's derivative on each element
                    [final_activation_function](double x) {
                        return final_activation_function->compute_derivative(x);
                    }
                )
            );
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
                current_intermediate_output = current_intermediate_output.unaryExpr(
                    [activation = layers[l-1].activation_function()](double x) {
                        return activation->compute_derivative(x);
                    }
                );

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
