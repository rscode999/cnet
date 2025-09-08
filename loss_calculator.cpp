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
 * Calculates Mean Squared Error (MSE) loss
 */
class MeanSquaredError : public LossCalculator {

public:

    /**
     * Creates a new MSE loss calculator
     */
    MeanSquaredError() {
    }



    /**
     * Returns a VectorXd containing the mean-squared error (MSE) losses, 
     * when measured between `predictions` and `actuals`.
     * 
     * `predictions` and `actuals` must be column vectors with the same length.
     * 
     * @param predictions predictions for a given input
     * @param actuals true values for model predictions
     * @return MSE calculator's loss of the model predictions
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
     * 
     * @param predictions predictions for a given input
     * @param actuals true values for model predictions
     * @return gradient of MSE calculator's loss of the model predictions
     */
    VectorXd compute_loss_gradient(const MatrixXd& predictions, const MatrixXd& actuals) override {
        assert((predictions.cols() == 1 && "Predictions must be a column vector"));
        assert((actuals.cols() == 1 && "Actuals must be a column vector"));
        assert((predictions.rows() == actuals.rows() && "Predictions and actuals must have the same dimension for gradient calculation"));

        return(2.0 / predictions.rows()) * (predictions - actuals);
    }


    /**
     * @return `"mse"`, the identifying string for a MSE loss calculator
     */
    string identifier() override {
        return "mse";
    }
};