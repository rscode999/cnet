#include "cnet/core.cpp"

#include <iostream>


/**
 * Returns a std::vector of Eigen::VectorXd's, of length 2^`n_bits`, where each Eigen::VectorXd represents its index number in binary
 * 
 * The most significant digit of each Eigen::VectorXd is the Eigen::VectorXd's highest index number.
 * 
 * The output contains every possible combination of `n_bits` 0's and 1's exactly once.
 * 
 * @param n_bits number of bits to use in the conversion 
 * @return std::vector of Eigen::VectorXd's, serving as input to the classification problem
 */
std::vector<Eigen::VectorXd> decimal_to_binary(int n_bits) {
    assert((n_bits>0 && "Number of inputs must be positive"));

    std::vector<Eigen::VectorXd> output;

    //Create each binary number specified
    for(int decimal_number = 0; decimal_number < pow(2, n_bits); decimal_number++) {

        Eigen::VectorXd binary_number(n_bits);
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
 * Returns a std::vector of Eigen::VectorXd's, of length 2^`n_bits`.
 * Each Eigen::VectorXd is of length 1, containing a 1 if the Eigen::VectorXd's index number has an even number of 1's in its binary representation, and 0 otherwise.
 * 
 * @param n_bits number of bits to use in the conversion 
 * @return std::vector of Eigen::VectorXd's, serving as output to the classification problem
 */
std::vector<Eigen::VectorXd> decimal_to_even_count(int n_bits) {
    assert((n_bits>0 && "Number of inputs must be positive"));

    std::vector<Eigen::VectorXd> output;

    //Create each binary number specified
    for(int decimal_number = 0; decimal_number < pow(2, n_bits); decimal_number++) {

        Eigen::VectorXd binary_number(1);
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
    using namespace std;
    using namespace Eigen;
    using namespace CNet;
}