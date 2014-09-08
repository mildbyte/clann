#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
using namespace std;

typedef struct Network {
    //weights[l][j][k]: weight from neuron k on layer l-1 to neuron j on layer l
    //biases[l][j]: bias of neuron j on layer l
    vector< vector< vector< float > > > weights;
    vector< vector< float > > biases;
    
    Network(int layers, vector< int > layer_sizes, int input_size) {
        weights = vector< vector< vector< float > > >(layers);
        weights[0] = vector< vector< float > >(layer_sizes[0]);
        for (int j = 0; j < layer_sizes[0]; j++) {
            weights[0][j] = vector< float >(input_size);
            for (int k = 0; k < input_size; k++) {
                weights[0][j][k] = rand() / (float)RAND_MAX;
            }
        }
        
        for (int i = 1; i < layers; i++) {
            weights[i] = vector< vector< float > >(layer_sizes[i]);
            for (int j = 0; j < layer_sizes[i]; j++) {
                weights[i][j] = vector< float >(layer_sizes[i-1]);
                for (int k = 0; k < layer_sizes[i-1]; k++) {
                    weights[i][j][k] = rand() / (float)RAND_MAX;
                }
            }
        }
        
        biases = vector< vector< float > >(layers);
        for (int i = 0; i < layers; i++) {
            biases[i] = vector< float >(layer_sizes[i]);
        }
    }
} Network;

inline float sigmoid(float input) {
    return 1.0 / (1.0 + exp(-input));
}

inline float inverse_sigmoid(float input) {
    return -log(1.0/input - 1.0);
}

vector< vector< float > > feedforward(Network network, vector< float > inputs) {
    vector< vector< float > > outputs(network.weights.size());
    for (int i = 0; i < network.weights.size(); i++) {
        outputs[i] = vector< float >(network.weights[i].size());
    }

    //Calculate the outputs for the 0th layer
    //Iterate over every neuron on the layer
    for (int j = 0; j < network.weights[0].size(); j++) {
        outputs[0][j] = network.biases[0][j];
        //Iterate over all inputs to this neuron
        for (int k = 0; k < network.weights[0][j].size(); k++) {
            outputs[0][j] += inputs[k] * network.weights[0][j][k];
        }
        outputs[0][j] = sigmoid(outputs[0][j]);
    }
    
    //Iterate over all layers i
    for (int i = 1; i < network.weights.size(); i++) {
        for (int j = 0; j < network.weights[i].size(); j++) {
            outputs[i][j] = network.biases[i][j];
            for (int k = 0; k < network.weights[i][j].size(); k++) {
                outputs[i][j] += outputs[i-1][k] * network.weights[i][j][k];
            }
            outputs[i][j] = sigmoid(outputs[i][j]);
        }
    }
    
    return outputs;
}

vector< vector< float > > calculate_deltas(Network network, vector< vector< float > > outputs, vector< float > labels) {
    vector< vector< float > > deltas(network.weights.size());
    for (int i = 0; i < network.weights.size(); i++) {
        deltas[i] = vector< float >(network.weights[i].size());
    }
    
    //Calculate the deltas for the output nodes
    int lastIndex = network.weights.size() - 1;
    for (int i = 0; i < network.weights[lastIndex].size(); i++) {
        float output = outputs[lastIndex][i];
        deltas[lastIndex][i] = (output - labels[i]) * output * (1 - output);
    }
    
    //Backpropagate the deltas to the previous layers
    for (int i = lastIndex - 1; i >= 0; i--) {        
        for (int j = 0; j < deltas[i].size(); j++) {
            float output = outputs[i][j];
            float delta_sum = 0;
            for (int k = 0; k < deltas[i+1].size(); k++) {
                delta_sum += deltas[i+1][k] * network.weights[i+1][k][j];
            }
            deltas[i][j] = delta_sum * output * (1 - output);
        }
    }
    
    return deltas;
}

vector< vector< vector< float > > > calculate_weight_deltas(Network network, vector< float > inputs, vector< vector< float > > deltas, float learning_rate) {
    //Initialize the weight deltas array
    vector< vector< vector< float > > > weight_deltas(network.weights.size());
    for (int i = 0; i < network.weights.size(); i++) {
        weight_deltas[i] = vector< vector< float > >(network.weights[i].size());
        for (int j = 0; j < network.weights[i].size(); j++) {
            weight_deltas[i][j] = vector< float >(network.weights[i][j].size());
        }
    }
    
    //Calculate the deltas for the input layer
    for (int j = 0; j < network.weights[0].size(); j++) {
        for (int k = 0; k < network.weights[0][j].size(); k++) {
            weight_deltas[0][j][k] = -learning_rate * deltas[0][j] * inputs[k];   
        }
    }
    
    //Caltulate the deltas for the rest
    for (int i = 1; i < network.weights.size(); i++) {
        for (int j = 0; j < network.weights[i].size(); j++) {
            for (int k = 0; k < network.weights[i][j].size(); k++) {
                weight_deltas[i][j][k] = -learning_rate * deltas[i][j] * inputs[k];   
            }
        }
    }
    
    return weight_deltas;
}

float calculate_current_error(vector< vector< float > > outputs, vector< float > labels) {
    float result = 0;
    
    for (int i = 0; i < labels.size(); i++) {
        result += pow(outputs[outputs.size()-1][i] - labels[i], 2);
    }
    
    return result * 0.5;
}

//Perform a gradient descent iteration for one input and one label
void gradient_descent_iteration(Network &network, vector< float > inputs, vector< float > labels, float learning_rate) {
    vector< vector< float > > outputs = feedforward(network, inputs);
    vector< vector< float > > deltas = calculate_deltas(network, outputs, labels);
    vector< vector< vector< float > > > weight_deltas = calculate_weight_deltas(network, inputs, deltas, learning_rate);
    
    //printf("Error: %f\n", calculate_current_error(outputs, labels));
    
    for (int i = 0; i < network.weights.size(); i++) {
        for (int j = 0; j < network.weights[i].size(); j++) {
            for (int k = 0; k < network.weights[i][j].size(); k++) {
                network.weights[i][j][k] += weight_deltas[i][j][k];   
            }
        }
    }
    
    for (int i = 0; i < network.biases.size(); i++) {
        for (int j = 0; j < network.biases[i].size(); j++) {
            network.biases[i][j] -= learning_rate * deltas[i][j];
        }
    }
}

//Perform a learning iteration on multiple inputs (in a round-robin fashion)
void multiple_input_iteration(Network &network, vector< vector< float > > inputs, vector< vector< float > > labels, float learning_rate) {
    for (int i = 0; i < inputs.size(); i++) {
        gradient_descent_iteration(network, inputs[i], labels[i], learning_rate);
    }
}

//Create and train a test network that adds two numbers together
int main() {
    Network n(3, {128, 128, 1}, 1);
//    n.weights = {{{1.7, 1.5}}, {{1.5}}};
    vector< vector< float > > inputs = {{0.3}, {0.4}, {-0.4}, {-0.2}, {0.9}};
    vector< vector< float > > labels = {{0.09}, {0.16}, {0.16}, {0.04}, {0.81}};
    
    for (int i = 0; i < 10000; i++) {    
        multiple_input_iteration(n, inputs, labels, 1.0);
    }
    
    printf("%f\n", feedforward(n, {0.4})[0][0]);
    
    return 0;
}
