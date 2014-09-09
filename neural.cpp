#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <armadillo>
using namespace std;

typedef struct Network {
    vector<arma::mat> weights;
    vector<arma::vec> biases;
    
    Network(int layers, vector< int > layer_sizes, int input_size) {
        weights = vector<arma::mat>(layers);
        weights[0] = arma::mat(layer_sizes[0], input_size, arma::fill::randn);

        for (int i = 1; i < layers; i++) {
            weights[i] = arma::mat(layer_sizes[i], layer_sizes[i-1], arma::fill::randn);
        }
        
        biases = vector<arma::vec>(layers);
        for (int i = 0; i < layers; i++) {
            biases[i] = arma::vec(layer_sizes[i], arma::fill::randn);
        }
    }
} Network;

inline arma::vec sigmoid(arma::vec input) {
    return 1.0 / (1.0 + arma::exp(-input));
}

vector<arma::vec> feedforward(Network network, arma::vec inputs) {
    vector<arma::vec> outputs(network.weights.size());

    arma::vec input = inputs;

    for (int i = 0; i < network.weights.size(); i++) {
        outputs[i] = sigmoid(network.biases[i] + network.weights[i] * input);
        input = outputs[i];
    }

    return outputs;
}

vector<arma::vec> calculate_deltas(Network network, vector<arma::vec> outputs, arma::vec labels) {
    vector<arma::vec> deltas(network.weights.size());
    
    //Calculate the deltas for the output nodes
    int lastIndex = network.weights.size() - 1;
    arma::vec lastOutputs = outputs[lastIndex];

    arma::vec lo1 = lastOutputs % (1 - lastOutputs); //output * (1 - output)
    arma::vec lo2 = lastOutputs % lo1; //output * output * (1 - output)

    deltas[lastIndex] = lo2 - lo1 % labels;
    
    //Backpropagate the deltas to the previous layers
    for (int i = lastIndex - 1; i >= 0; i--) {
        deltas[i] = (network.weights[i+1].t() * deltas[i+1]) % outputs[i] % (1 - outputs[i]);
    }
    
    return deltas;
}

vector<arma::mat> calculate_weight_deltas(Network network, arma::vec inputs, vector<arma::vec> outputs, vector<arma::vec> deltas, double learning_rate) {
    //Initialize the weight deltas array
    vector<arma::mat> weight_deltas(network.weights.size());
    arma::vec input = inputs;

    for (int i = 0; i < network.weights.size(); i++) {
        weight_deltas[i] = -learning_rate * deltas[i] * input.t();
        input = outputs[i];
    }
    
    return weight_deltas;
}

float calculate_current_error(vector<arma::vec> outputs, arma::vec labels) {
    return arma::norm(labels - outputs.back(), 2);
}

//Perform a gradient descent iteration for one input and one label
void gradient_descent_iteration(Network &network, arma::vec inputs, arma::vec labels, double learning_rate) {
    vector<arma::vec> outputs = feedforward(network, inputs);
    vector<arma::vec> deltas = calculate_deltas(network, outputs, labels);
    vector<arma::mat> weight_deltas = calculate_weight_deltas(network, inputs, outputs, deltas, learning_rate);
    
    printf("Error: %f\n", calculate_current_error(outputs, labels));
    
    for (int i = 0; i < network.weights.size(); i++) {
        network.weights[i] += weight_deltas[i];
    }
    
    for (int i = 0; i < network.biases.size(); i++) {
        network.biases[i] -= learning_rate * deltas[i];
    }
}

//Perform a learning iteration on multiple inputs (in a round-robin fashion)
void multiple_input_iteration(Network &network, vector<arma::vec> inputs, vector<arma::vec> labels, double learning_rate) {
    for (int i = 0; i < inputs.size(); i++) {
        gradient_descent_iteration(network, inputs[i], labels[i], learning_rate);
    }
}

//Create and train a test network that adds two numbers together
int main() {
    Network n(2, {2048, 1}, 1);
    vector<arma::vec> inputs = {arma::vec({0.3}), arma::vec({0.4}), arma::vec({-0.4}), arma::vec({-0.2}), arma::vec({0.9})};
    vector<arma::vec> labels = {arma::vec({0.09}), arma::vec({0.16}), arma::vec({0.16}), arma::vec({0.04}), arma::vec({0.81})};
    
    for (int i = 0; i < 10000; i++) {    
        multiple_input_iteration(n, inputs, labels, 1.0);
    }
    
    printf("%f\n", feedforward(n, arma::vec({0.4}))[0][0]);
    
    return 0;
}
