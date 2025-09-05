#include "layer.cpp"
#include "loss_calculator.cpp"

#include <memory>
#include <vector>


/**
 * Returns the MSE derivative taken element-wise between `prediction` and `actual`.
 * 
 * Both inputs must be column vectors with the same dimension.
 */
VectorXd mean_squared_error_deriv(const MatrixXd& prediction, const MatrixXd& actual) {
    assert((prediction.cols() == 1 && "Prediction must be a column vector"));
    assert((actual.cols() == 1 && "Actual must be a column vector"));
    assert((prediction.rows() == actual.rows() && "Prediction and actual must have same number of rows"));

    VectorXd output(prediction.rows());
    for(int i=0; i<prediction.rows(); i++) {
        output(i) = 1.0/prediction.rows() * (prediction(i) - actual(i));
    }
    return output;
}




/**
 * Interface (fully abstract class) for network optimizers. Cannot be directly used.
 */
class Optimizer {
public:
    /**
     * Not yet implemented.
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

        const int N_NETWORK_LAYERS = layers.size();
        
        assert((predictions.cols() == 1 && "Predicted values must be a column vector"));
        assert((predictions.rows() == layers[N_NETWORK_LAYERS-1].output_dimension() && "Predicted value vector must have dimension equal to the network's output dimension"));
        assert((actuals.cols() == 1 && "Actual values must be a column vector"));
        assert((actuals.rows() == layers[N_NETWORK_LAYERS-1].output_dimension() && "Actual value vector must have dimension equal to the network's output dimension"));

        VectorXd delta = mean_squared_error_deriv(intermediate_outputs.back().post_activation, actuals);
        //or otherwise use the loss function's derivative

        //Component-wise multiply to the element-wise differentiated final bias vector
        if(true) { //Do this if not using both MSE loss and sigmoid activation
            if(true) {
                delta = delta.cwiseProduct(intermediate_outputs.back().post_activation.unaryExpr( layers.back().activation_function_derivative() ));
            }
            else { //Do this if using relu activation (or a pre-activation function)
                delta = delta.cwiseProduct(intermediate_outputs.back().pre_activation.unaryExpr( layers.back().activation_function_derivative() ));
            }
        }
        //This is for MSE + sigmoid
        else { 
            delta = intermediate_outputs.back().post_activation - actuals;
        }

        for(int l = N_NETWORK_LAYERS-1; l >= 0; l--) {
            VectorXd previous_post_activation = (l > 0) 
                ? intermediate_outputs[l-1].post_activation
                : initial_input;
            
            //get gradients
            MatrixXd weight_changes = delta * previous_post_activation.transpose();
            VectorXd bias_changes = delta;

            

            //propagate delta
            if(l > 0) {
                MatrixXd current_weights = layers[l].weight_matrix();
                delta = (current_weights.transpose() * delta)
                .cwiseProduct(intermediate_outputs[l-1].post_activation.unaryExpr( layers[l-1].activation_function_derivative() ));
            }

            //update weights and biases
            layers[l].set_weight_matrix( 
                layers[l].weight_matrix() - (learn_rate * weight_changes)
            );
            layers[l].set_bias_vector(
                layers[l].bias_vector() - (learn_rate * bias_changes)
            );
        }
    }
};
