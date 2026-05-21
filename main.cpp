#include "cnet/data_loader.cpp"
#include <iostream>
#include <Eigen/Core>


Eigen::VectorXd decimal_to_binary(int n_bits, int decimal_number) {
    assert((n_bits>0 && "Number of bits for dec->bin conversion must be positive"));
    assert((decimal_number>=0 && decimal_number<(int)pow(2, n_bits) && "Decimal number must be between 0 and 2^n_bits-1"));

    Eigen::VectorXd output(n_bits);
    int current = decimal_number;
    for(int i=0; i<output.size(); i++) {
        output(i) = current % 2;
        current = current / 2;
    }

    return output;
}


int main() {
    using namespace CNet;
    using namespace std;
    using namespace Eigen;

    vector<VectorXd> inputs;
    for (int i = 0; i < 16; i++) {
        inputs.push_back(decimal_to_binary(5, i));
    }

    DataLoader loader = DataLoader(inputs);
    for (int i = 17; i < 32; i++) {
        loader.add(decimal_to_binary(5, i));
    }

    loader.shuffle();
}