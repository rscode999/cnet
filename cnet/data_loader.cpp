#pragma once

#include <cassert>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

namespace CNet {




/**
 * Utility class that stores and feeds data to a Network.
 * 
 * Stores input vectors, as type `Eigen::VectorXd`, of a specified nonzero length.
 * Shuffles and loads a specified number of inputs.
 */
class DataLoader {
private:
    /**
     * List of network inputs
     */
    std::vector<Eigen::VectorXd> vectors;

    /**
     * Length of the input vectors. Must be positive.
     */
    int32_t input_dimension;

    /**
     * Next index to load in the inputs.
     * 
     * Resets to 0 if it equals or exceeds `inputs.size()`, 
     */
    int32_t next_input;

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////

public:
    friend void shuffle(DataLoader& loader1, DataLoader& loader2, unsigned int seed);

    /**
     * Creates an empty DataLoader of input dimension `input_vector_dimension`.
     * @param input_vector_dimension dimension of the DataLoader's input vectors. Must be positive.
     */
    DataLoader(int32_t input_vector_dimension) {
        assert(input_vector_dimension > 0 && "Data loader's input dimension must be positive");
        input_dimension = input_vector_dimension;
        next_input = 0;
    }

    /**
     * Creates a new DataLoader containing `initial_inputs`.
     * @param initial_inputs inputs to load. All elements must have the same nonzero length.
     */
    DataLoader(std::vector<Eigen::VectorXd> initial_inputs) {
        next_input = 0;
        vectors = {};

        for(int32_t i = 0; i < (int32_t)initial_inputs.size(); i++) {
            assert(initial_inputs[i].size() == initial_inputs[0].size() && "All inputs to a DataLoader must have the same size");
            vectors.push_back(initial_inputs[i]);
        }
        input_dimension = initial_inputs[0].size();
    }


    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    //GETTERS

    /**
     * @return the DataLoader's data, as a `std::vector<Eigen::VectorXd>`
     */
    std::vector<Eigen::VectorXd> data() const {
        return vectors;
    }



    /**
     * @return next input vector in the DataLoader
     * @throws `std::out_of_range` if the DataLoader is empty
     */
    Eigen::VectorXd next() {
        //Ensure there is a next input
        if(vectors.size() <= 0) {
            throw std::out_of_range("Cannot load next value from an empty DataLoader");
        }

        Eigen::VectorXd output = vectors[next_input];
        next_input = (next_input + 1) % (int32_t)vectors.size();
        return output;
    }



    /**
     * Returns a `std::vector<Eigen::VectorXd>` containing the next `load_amount` inputs in the DataLoader.
     * 
     * @param load_amount number of inputs to load
     * @return the next `load_amount` input vectors
     * @param `std::out_of_range` if `load_amount` is greater than the DataLoader's amount of vectors
     */
    std::vector<Eigen::VectorXd> next(int32_t load_amount) {
        //Ensure there is a next input
        if(load_amount > (int32_t)vectors.size()) {
            throw std::out_of_range("Too few vectors in the DataLoader");
        }

        std::vector<Eigen::VectorXd> output;

        for(int32_t i = 0; i < load_amount; i++) {
            output.push_back(vectors[next_input]);
            next_input = (next_input + 1) % (int32_t)vectors.size();
        }
        
        return output;
    }



    /**
     * @return index of the next input vector to load
     */
    int32_t next_input_index() const {
        return next_input;
    }



    /**
     * @return number of vectors managed by this DataLoader
     */
    int32_t size() const {
        return (int32_t)vectors.size();
    }



    /**
     * @return required input dimension of vectors accepted by this DataLoader
     */
    int32_t vector_input_dimension() const {
        return this->input_dimension;
    }

    

    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    //SETTERS

    /**
     * Adds `new_input` to the DataLoader.
     * @param new_input input to add. Must have dimension `vector_input_dimension()`
     */
    void add(const Eigen::VectorXd& new_input) {
        assert(new_input.size() == vector_input_dimension() && "DataLoader add operation requires vectors to have dimension vector_input_dimension()");
        vectors.push_back(new_input);
    }



    /**
     * Removes all input vectors from the DataLoader.
     */
    void clear() {
        vectors.clear();
        next_input = 0;
    }



    /**
     * Replaces element `index` of the DataLoader with `new_input`.
     * @param index element number to replace (0-based indexing). Must be on the interval [0, `size()`-1]
     * @param new_input value to replace. Must have dimension `vector_input_dimension()`
     */
    void set(int32_t index, const Eigen::VectorXd& new_input) {
        assert(0 <= index && index < size() && "DataLoader replace-at index must be on the interval [0, size() - 1]");
        assert(new_input.size() == vector_input_dimension() && "DataLoader replace-at operation requires vectors to have dimension vector_input_dimension()");
        vectors[index] = new_input;
    }


    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    //METHODS

    /**
     * Randomly permutes the DataLoader's data.
     * @param seed seed for the random number generator; random if set to 0. Default 0.
     */
    void shuffle(unsigned int seed = 0) {

        // Random number generator
        static std::random_device rd;
        static std::mt19937 gen((seed == 0) ? rd() : seed);
        std::uniform_real_distribution<double> dist(0.0, (double)size());

        for(int32_t i = (int32_t)size() - 1; i > 0; i--) {
            //random index from 0 to i
            const int32_t random_index = ((int32_t)dist(gen)) % i;

            //switch indices
            Eigen::VectorXd temp = vectors[i];
            vectors[i] = vectors[random_index];
            vectors[random_index] = temp;
        } 
    }

 
    /////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////
    //OPERATOR OVERLOADS

    /**
     * Returns element `index` of the DataLoader, as a non-const reference.
     * @param index element number to retrieve. Must be on the interval [0, `size()`-1]
     * @return the `index`-th input of the DataLoader
     */
    Eigen::VectorXd& operator[](int32_t index) {
        assert(0 <= index && index < (int32_t)vectors.size() && "Array-access operator requires index to be on [0, size() - 1]");
        return vectors[index];
    }

    /**
     * Returns element `index` of the DataLoader, as a const reference.
     * @param index element number to retrieve. Must be on the interval [0, `size()`-1]
     * @return the `index`-th input of the DataLoader
     */
    const Eigen::VectorXd& operator[](int32_t index) const {
        assert(0 <= index && index < (int32_t)vectors.size() && "Array-access operator requires index to be on [0, size() - 1]");
        return vectors[index];
    }
};



/**
 * Randomly permutes the data of `loader1` and `loader2`.
 * 
 * Corresponding elements in both loaders remain aligned after shuffling.
 * 
 * Mutates both DataLoader inputs.
 * 
 * @param loader1 first DataLoader to shuffle
 * @param loader2 second DataLoader to shuffle. Must have the same amount of vectors as `loader1`
 * @param seed seed for the random number generator; random if set to 0. Default 0.
 */
void shuffle(DataLoader& loader1, DataLoader& loader2, unsigned int seed = 0) {
    assert(loader1.size() == loader2.size() && "Shuffle requires both DataLoaders to have the same size");

    // Random number generator
    static std::random_device rd;
    static std::mt19937 gen = std::mt19937( (seed == 0) ? rd() : seed);
    std::uniform_real_distribution<double> dist(0.0, (double)loader1.size());

    for(int32_t i = (int32_t)loader1.size() - 1; i > 0; i--) {
        //random index from 0 to i
        const int32_t random_index = ((int32_t)dist(gen)) % i;

        //switch indices
        Eigen::VectorXd temp = loader1.vectors[i];
        loader1.vectors[i] = loader1.vectors[random_index];
        loader1.vectors[random_index] = temp;

        temp = loader2.vectors[i];
        loader2.vectors[i] = loader2.vectors[random_index];
        loader2.vectors[random_index] = temp;
    } 
}




}