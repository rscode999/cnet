#include "layer.cpp"
#include "loss_calculator.cpp"

#include <vector>



/**
 * Interface (fully abstract class) for network optimizers. Cannot be directly used.
 */
class Optimizer {
public:
    /**
     * Not yet implemented.
     * 
     * Not intended to be called by a user.
     */
    virtual void step(vector<Layer>& layers, const VectorXd& initial_input, const vector<LayerCache>& intermediate_outputs,
        const MatrixXd& predictions, const MatrixXd& actuals, const shared_ptr<LossCalculator> loss_calculator) = 0;

    virtual ~Optimizer() = default;
};



/**
 * A Stochastic Gradient Descent (SGD) optimizer
 */
class SGD : public Optimizer {

private:

    /**
     * Learning rate, dictating the speed of convergence step size. Must be positive.
     */
    double learn_rate;

public:

    /**
     * Creates a new SGD optimizer
     */
    SGD(double learning_rate = 0.9) {
        assert((learning_rate>0 && "Learning rate must be positive"));
        learn_rate = learning_rate;
    }   


    /**
     * Not yet implemented!
     */
    void step(vector<Layer>& layers, const VectorXd& initial_input, const vector<LayerCache>& intermediate_outputs,
        const MatrixXd& predictions, const MatrixXd& actuals, const shared_ptr<LossCalculator> loss_calculator) override {
        
        assert((predictions.cols() == 1 && "Predicted values must be a column vector"));
        assert((predictions.rows() == layers.back().output_dimension() && "Predicted value vector must have dimension equal to the network's output dimension"));
        assert((actuals.cols() == 1 && "Actual values must be a column vector"));
        assert((actuals.rows() == layers.back().output_dimension() && "Actual value vector must have dimension equal to the network's output dimension"));

        VectorXd delta = loss_calculator->compute_loss_gradient(predictions, actuals);
        //or otherwise use the loss function's derivative
        
        //Get final layer's activation function
        shared_ptr<ActivationFunction> final_activation_function = layers.back().activation_function();


        //Component-wise multiply to the element-wise differentiated final bias vector
        
 
        //Do this if using a pre-activation function
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
        //If using a post-activation function
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
        
   

        for(int l = layers.size()-1; l >= 0; l--) {
            VectorXd previous_post_activation = (l > 0) 
                ? intermediate_outputs[l-1].post_activation
                : initial_input;

            //propagate delta
            VectorXd new_delta;
            if(l > 0) {
                MatrixXd current_weights = layers[l].weight_matrix();

                VectorXd activation_derivative_input = layers[l-1].activation_function()->using_pre_activation()
                    ? intermediate_outputs[l-1].pre_activation
                    : intermediate_outputs[l-1].post_activation;

                new_delta = (current_weights.transpose() * delta).cwiseProduct(
                    activation_derivative_input.unaryExpr(
                        [activation = layers[l-1].activation_function()](double x) {
                            return activation->compute_derivative(x);
                        }
                    )
                );
            }

            //get gradients
            MatrixXd weight_changes = delta * previous_post_activation.transpose();
            VectorXd bias_changes = delta;

            //update weights and biases
            layers[l].set_weight_matrix( 
                layers[l].weight_matrix() - (learn_rate * weight_changes)
            );
            layers[l].set_bias_vector(
                (layers[l].bias_vector() - (learn_rate * bias_changes)).eval()
            );

            if(l > 0) {
                delta = new_delta;
            }
        }
    }
};
