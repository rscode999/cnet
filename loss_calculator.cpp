#include <string>

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class LossCalculator {
public:

    /**
     * Returns the loss (error) when measured between `predictions` and `actuals`.
     * 
     * `predictions` and `actuals` must be column vectors with the same length.
     * 
     * @param predictions model's predictions for a given input
     * @param actuals true values for model predictions
     * @return calculator's loss of the model predictions
     */
    virtual double compute_loss(const MatrixXd& predictions, const MatrixXd& actuals) = 0;

    /**
     * Returns the gradient of the losses when measured between `predictions` and `actuals`.
     * 
     * `predictions` and `actuals` must be column vectors with the same length.
     * 
     * @param predictions model's predictions for a given input
     * @param actuals true values for model predictions
     * @return calculator's loss gradient of the model predictions
     */
    virtual VectorXd compute_loss_gradient(const MatrixXd& predictions, const MatrixXd& actuals) = 0;

    /**
     * @return the identifying string of the loss calculator
     */
    virtual string identifier() = 0;
};



/**
 * You think I have any idea what this does???
 */
class BinaryCrossEntropy : public LossCalculator {
public:
    double compute_loss(const MatrixXd& predictions, const MatrixXd& actuals) override {
        assert(predictions.rows() == actuals.rows());
        double loss = 0.0;
        for (int i = 0; i < predictions.rows(); ++i) {
            double y_hat = predictions(i, 0);
            double y     = actuals(i, 0);
            y_hat = std::min(std::max(y_hat, 1e-7), 1.0 - 1e-7); // Clamp for stability
            loss += -y * log(y_hat) - (1 - y) * log(1 - y_hat);
        }
        return loss / predictions.rows();
    }

    VectorXd compute_loss_gradient(const MatrixXd& predictions, const MatrixXd& actuals) override {
        assert(predictions.rows() == actuals.rows());
        VectorXd grad = predictions.col(0) - actuals.col(0); // Simple difference
        return grad;
    }

    string identifier() override {
        return "binary_cross_entropy";
    }
};



/**
 * Object that calculates Mean Squared Error (MSE) loss
 */
class MSE : public LossCalculator {

public:

    /**
     * Creates a new MSE loss calculator
     */
    MSE() {
    }



    /**
     * Returns a VectorXd containing the mean-squared error (MSE) losses, 
     * when measured between `predictions` and `actuals`.
     * 
     * `predictions` and `actuals` must be column vectors with the same length.
     */
    double compute_loss(const MatrixXd& predictions, const MatrixXd& actuals) override {
        assert((predictions.cols() == 1 && "Predictions must be a column vector"));
        assert((actuals.cols() == 1 && "Actuals must be a column vector"));
        assert((predictions.rows() == actuals.rows() && "Predictions and actuals must have the same dimension"));

        // Compute the MSE
        VectorXd output = actuals - predictions;
        return output.squaredNorm() / output.size();
    }


    /**
     * Returns the gradient of MSE for `predictions` and `actuals`
     */
    VectorXd compute_loss_gradient(const MatrixXd& predictions, const MatrixXd& actuals) override {
        assert((predictions.cols() == 1 && "Predictions must be a column vector"));
        assert((actuals.cols() == 1 && "Actuals must be a column vector"));
        assert((predictions.rows() == actuals.rows() && "Predictions and actuals must have the same dimension for gradient calculation"));

        return(2.0 / predictions.rows())* (predictions - actuals);
    }


    /**
     * @return `"mse"`, the identifying string for a MSE loss calculator
     */
    string identifier() override {
        return "mse";
    }
};