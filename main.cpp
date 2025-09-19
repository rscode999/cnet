#include "cnet/network.cpp"

#include <iostream>

using namespace std;

const int N_INPUTS = 10;

/**
 * Returns a vector of VectorXd's, of length 2^`n_bits`, where each VectorXd represents its index number in binary
 * 
 * The most significant digit of each VectorXd is the VectorXd's highest index number.
 * 
 * The output contains every possible combination of `n_bits` 0's and 1's exactly once.
 * 
 * @param n_bits number of bits to use in the conversion 
 * @return vector of VectorXd's, serving as input to the classification problem
 */
vector<VectorXd> decimal_to_binary(int n_bits) {
    assert((n_bits>0 && "Number of inputs must be positive"));

    vector<VectorXd> output;

    //Create each binary number specified
    for(int decimal_number = 0; decimal_number < pow(2, n_bits); decimal_number++) {

        VectorXd binary_number(n_bits);
        int current = decimal_number;

        //get current bit, divide off the bit, repeat
        for(int i=0; i<binary_number.size(); i++) {
            binary_number(i) = current % 2;
            current = current / 2;
        }

        //add binary number to the output
        output.push_back(binary_number);
    }

    return output;
}


/**
 * Returns a vector of VectorXd's, of length 2^`n_bits`.
 * Each VectorXd is of length 1, containing a 1 if the VectorXd's index number has an even number of 1's in its binary representation, and 0 otherwise.
 * 
 * @param n_bits number of bits to use in the conversion 
 * @return vector of VectorXd's, serving as output to the classification problem
 */
vector<VectorXd> decimal_to_even_count(int n_bits) {
    assert((n_bits>0 && "Number of inputs must be positive"));

    vector<VectorXd> output;

    //Create each binary number specified
    for(int decimal_number = 0; decimal_number < pow(2, n_bits); decimal_number++) {

        VectorXd binary_number(1);
        int current = decimal_number;
        int ones_count = 0;

        //get current bit, divide off the bit, repeat
        for(int i=0; i<n_bits; i++) {
            ones_count += current % 2;
            current = current / 2;
        }

        binary_number(0) = (ones_count % 2 == 0) ? 1 : 0;

        //add binary number to the output
        output.push_back(binary_number);
    }

    return output;
}




int main() {
    
    /*
    TASK: Even Number Tracker

    You are given `N_INPUTS` inputs, each of which can either be 0 or 1.

    You have 1 output.
    This output should be 1 if, out of the `N_INPUTS` inputs, an even number of them are 1.
    Otherwise, the output should be 0.

    Example if N_INPUTS = 4:
    A possible input is [1, 0, 0, 0]. The output should be 0, because there is an odd number of 1's in the input.
    Another input is [0, 1, 0, 1]. The output should be 1, because there is an even number of 1's in the input.

    You are given `inputs`, to put in the network, and `expected_outputs`.
    For every index `i` in `expected_outputs`, expected_outputs[i] is what should be the network output if `inputs[i]` is given as an input.

    Train a neural network to the input data.
    Then print the network's output for each value in `inputs`.

    The headers <iostream>, <memory>, and <vector> are provided. The Eigen and std namespaces are used.
    To make a shared smart pointer, use `shared_ptr {name} = make_shared<typename>(type's arguments...)`

    For 5 inputs, this setup gave over 90% accuracy:
        Layer 1: 5 -> 20, Relu activation
        Layer 2: 20 -> 10, Sigmoid activation
        Layer 3: 10 -> 1, no activation

        Cross Entropy loss
        SGD optimization, learning rate 0.0005, momentum coefficient 0.9
        5,000 epochs of training
    */

    const int N_INPUTS = 5;

    vector<VectorXd> inputs = decimal_to_binary(N_INPUTS);
    vector<VectorXd> expected_outputs = decimal_to_even_count(N_INPUTS);

    //Create, train, and evaluate your network here.
}