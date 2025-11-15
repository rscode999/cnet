#include "cnet/storage.cpp"


int main() {
    using namespace CNet;
    using namespace std;

    Network net;
    shared_ptr<Relu> relu = make_shared<Relu>();
    net.add_layer(1, 3, relu, "my relu layer");
    net.add_layer(3, 3);
    net.set_loss_calculator(make_shared<MeanSquaredError>());
    net.set_optimizer(make_shared<SGD>(0.01, 0.9, 5));

    store_network_config("test.txt", net);
    Network net2 = load_network_config("test.txt");
    cout << net2 << endl;
}   