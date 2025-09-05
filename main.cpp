#include "network.cpp"

#include <iostream>

using namespace std;



/**
 * Loads the model with precomputed weights and biases for a 2-input XOR classification problem,
 * then checks the forward process's output against the reference model's results.
 * 
 * Model architecture: 2d -> 3d -> 1d.
 */
void test_xor_2layer() {
    //The weights and biases came from a Pytorch model

    MatrixXd l0_weights(3, 2);
    l0_weights << -0.33610883, -0.07779725,
    -1.5301809, -1.5219239,
    -3.9380543, -3.8303926;

    MatrixXd l1_weights(1, 3);
    l1_weights << 0.16919717,  3.2087102,  -3.4358373;

    VectorXd l0_biases(3);
    l0_biases << -1.9276419,   1.7493086,   0.39942038;

    VectorXd l1_biases(1);
    l1_biases << -0.6983648;

    Layer l0 = Layer(2, 3, sigmoid, sigmoid_derivative, "l0");
    Layer l1 = Layer(3, 1, "l1");
    l0.set_weight_matrix(l0_weights);
    l1.set_weight_matrix(l1_weights);
    l0.set_bias_vector(l0_biases);
    l1.set_bias_vector(l1_biases);

    shared_ptr<SGD> sgd = make_shared<SGD>();
    shared_ptr<MSE> mse = make_shared<MSE>();
    Network net;
    net.add_loss_calculator(mse);
    net.add_optimizer(sgd);
    net.add_layer(l0);
    net.add_layer(l1);
    
    net.enable_training();
    
    VectorXd in(2);
    in << 0,0;
    cout << "Results for 0,0:  " << net.forward(in) << endl;
    cout << "Expected for 0,0: -5.9605e-08\n" << endl;

    in << 0,1;
    cout << "Results for 0,1:  " << net.forward(in) << endl;
    cout << "Expected for 0,1: 1\n" << endl;

    in << 1,0;
    cout << "Results for 1,0:  " << net.forward(in) << endl;
    cout << "Expected for 1,0: 1\n" << endl;

    in << 1,1;
    cout << "Results for 1,1:  " << net.forward(in) << endl;
    cout << "Expected for 1,1: 3.5763e-07" << endl;

    sgd.reset();
}


/**
 * Loads the model with precomputed weights and biases for a 2-input XOR classification problem,
 * then checks the forward process's output against the reference model's results.
 * 
 * Architecture: 2d -> 1d.
 * 
 * Note that this model is poorly trained.
 */
void test_xor_1layer() {
    MatrixXd weights(1, 2);
    weights << 0.2617063,  0.27747557;

    VectorXd biases(1);
    biases << -0.19150648;

    Layer layer = Layer(2, 1, sigmoid, sigmoid_derivative, "layer");
    layer.set_weight_matrix(weights);
    layer.set_bias_vector(biases);

    Network net = Network();
    shared_ptr<SGD> optimizer = make_shared<SGD>();
    shared_ptr<MSE> loss_calc = make_shared<MSE>();
    net.add_layer(layer);
    net.add_loss_calculator(loss_calc);
    net.add_optimizer(optimizer);

    net.enable_training();

    VectorXd in(2);

    in << 0,0;
    cout << "Results for 0,0:  " << net.forward(in) << endl;
    cout << "Expected for 0,0: 0.4523\n" << endl;

    in << 0,1;
    cout << "Results for 0,1:  " << net.forward(in) << endl;
    cout << "Expected for 0,1: 0.5215\n" << endl;

    in << 1,0;
    cout << "Results for 1,0:  " << net.forward(in) << endl;
    cout << "Expected for 1,0: 0.5175\n" << endl;

    in << 1,1;
    cout << "Results for 1,1:  " << net.forward(in) << endl;
    cout << "Expected for 1,1: 0.5861" << endl;

    optimizer.reset();
}




int main() {
    test_xor_2layer();
}