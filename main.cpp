#include "cnet/network.cpp"

#include <iostream>

using namespace std;

int main() {
    Network net = Network();
    shared_ptr<Relu> relu_activ = make_shared<Relu>();
    net.add_layer(Layer(3, 5, relu_activ, "input"));
    net.add_layer(Layer(5, 10, relu_activ, "hidden"));
    net.add_layer(Layer(10, 1, relu_activ, "output"));
    relu_activ.reset();

    shared_ptr<MeanSquaredError> loss_calc = make_shared<MeanSquaredError>();
    net.set_loss_calculator(loss_calc);

    shared_ptr<SGD> optim = make_shared<SGD>();
    net.set_optimizer(optim);

    net.enable();

}