#include <Eigen/Dense>

#include <string>

using namespace Eigen;

class LossCalculator {

    /**
     * Returns the losses when measured between `predictions` and `actuals`.
     * 
     * `predictions` and `actuals` must be column vectors with the same length.
     */
    virtual double compute_loss(const MatrixXd& predictions, const MatrixXd& actuals) = 0;

    /**
     * Returns the gradient of the losses when measured between `predictions` and `actuals`.
     * 
     * `predictions` and `actuals` must be column vectors with the same length.
     */
    virtual VectorXd compute_loss_gradient(const MatrixXd& predictions, const MatrixXd& actuals) = 0;

    /**
     * Returns the identifying string of the loss calculator.
     */
    virtual string identifier() = 0;
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

        return (2.0 / actuals.size()) * (predictions - actuals);
    }


    /**
     * @return `"mse"`, the identifying string for a MSE loss calculator
     */
    string identifier() override {
        return "mse";
    }
};