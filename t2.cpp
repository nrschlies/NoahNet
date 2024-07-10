#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <pthread.h>
#include <memory>  // Include memory header for smart pointers
#include <fstream>

// Add debug prints to trace the program's execution
#define DEBUG_PRINT(x) std::cout << x << std::endl;

// Positional Encoding
std::vector<std::vector<float>> positional_encoding(int seq_len, int d_model) {
    DEBUG_PRINT("Entering positional_encoding");
    std::vector<std::vector<float>> pos_enc(seq_len, std::vector<float>(d_model));
    for (int pos = 0; pos < seq_len; ++pos) {
        for (int i = 0; i < d_model; ++i) {
            if (i % 2 == 0) {
                pos_enc[pos][i] = sin(pos / pow(10000, (2 * i) / float(d_model)));
            } else {
                pos_enc[pos][i] = cos(pos / pow(10000, (2 * i) / float(d_model)));
            }
        }
    }
    DEBUG_PRINT("Exiting positional_encoding");
    return pos_enc;
}

// Scaled Dot-Product Attention
Eigen::MatrixXd scaled_dot_product_attention(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& K, const Eigen::MatrixXd& V, double d_k) {
    DEBUG_PRINT("Entering scaled_dot_product_attention");
    DEBUG_PRINT("Q dimensions: " << Q.rows() << "x" << Q.cols());
    DEBUG_PRINT("K dimensions: " << K.rows() << "x" << K.cols());
    DEBUG_PRINT("V dimensions: " << V.rows() << "x" << V.cols());

    Eigen::MatrixXd scores = Q * K.transpose() / sqrt(d_k);
    DEBUG_PRINT("Scores dimensions: " << scores.rows() << "x" << scores.cols());

    Eigen::MatrixXd exp_scores = scores.array().exp();
    DEBUG_PRINT("Exp scores dimensions: " << exp_scores.rows() << "x" << exp_scores.cols());

    Eigen::VectorXd sum_exp_scores = exp_scores.rowwise().sum();
    DEBUG_PRINT("Sum exp scores dimensions: " << sum_exp_scores.rows() << "x" << sum_exp_scores.cols());

    Eigen::MatrixXd softmax_scores = exp_scores.array().colwise() / sum_exp_scores.array();
    DEBUG_PRINT("Softmax scores dimensions: " << softmax_scores.rows() << "x" << softmax_scores.cols());

    Eigen::MatrixXd output = softmax_scores * V;
    DEBUG_PRINT("Output dimensions: " << output.rows() << "x" << output.cols());

    DEBUG_PRINT("Exiting scaled_dot_product_attention");
    return output;
}

// Feed-Forward Neural Network
Eigen::MatrixXd feed_forward(const Eigen::MatrixXd& input, const Eigen::MatrixXd& W1, const Eigen::MatrixXd& W2, const Eigen::VectorXd& b1, const Eigen::VectorXd& b2) {
    DEBUG_PRINT("Entering feed_forward");
    assert(input.cols() == W1.rows() && "Input columns must match W1 rows");
    assert(W1.cols() == W2.rows() && "W1 columns must match W2 rows");
    assert(W2.cols() == b2.size() && "W2 columns must match b2 size");

    Eigen::MatrixXd hidden = (input * W1).rowwise() + b1.transpose();
    hidden = hidden.array().max(0); // ReLU activation
    Eigen::MatrixXd output = (hidden * W2).rowwise() + b2.transpose();
    DEBUG_PRINT("Exiting feed_forward");
    return output;
}

// Layer Normalization
Eigen::MatrixXd layer_norm(const Eigen::MatrixXd& input, const Eigen::VectorXd& gamma, const Eigen::VectorXd& beta) {
    DEBUG_PRINT("Entering layer_norm");
    DEBUG_PRINT("Input dimensions: " << input.rows() << "x" << input.cols());
    DEBUG_PRINT("Gamma dimensions: " << gamma.size());
    DEBUG_PRINT("Beta dimensions: " << beta.size());

    assert(input.cols() == gamma.size() && "Input columns must match gamma size");
    assert(input.cols() == beta.size() && "Input columns must match beta size");

    Eigen::VectorXd mean = input.colwise().mean();
    DEBUG_PRINT("Mean dimensions: " << mean.size());

    Eigen::VectorXd variance = ((input.rowwise() - mean.transpose()).array().square().colwise().mean()).matrix();
    DEBUG_PRINT("Variance dimensions: " << variance.size());

    Eigen::MatrixXd norm = (input.rowwise() - mean.transpose()).array().rowwise() / variance.array().sqrt().transpose();
    DEBUG_PRINT("Norm dimensions: " << norm.rows() << "x" << norm.cols());

    Eigen::MatrixXd gamma_mat = gamma.transpose().replicate(input.rows(), 1);
    Eigen::MatrixXd beta_mat = beta.transpose().replicate(input.rows(), 1);

    Eigen::MatrixXd output = norm.array() * gamma_mat.array() + beta_mat.array();
    DEBUG_PRINT("Output dimensions: " << output.rows() << "x" << output.cols());

    DEBUG_PRINT("Exiting layer_norm");
    return output;
}

// Multi-Head Attention with pthread
class MultiHeadAttention {
public:
    MultiHeadAttention(int num_heads, int d_model) : num_heads(num_heads), d_model(d_model), d_k(d_model / num_heads) {
        DEBUG_PRINT("Entering MultiHeadAttention constructor");
        initialize_weights();
        DEBUG_PRINT("Exiting MultiHeadAttention constructor");
    }

    Eigen::MatrixXd operator()(const Eigen::MatrixXd& Q, const Eigen::MatrixXd& K, const Eigen::MatrixXd& V) {
        DEBUG_PRINT("Entering MultiHeadAttention operator()");
        std::vector<Eigen::MatrixXd> heads(num_heads);
        std::vector<pthread_t> threads(num_heads);
        std::vector<std::unique_ptr<ThreadData>> data;

        for (int i = 0; i < num_heads; ++i) {
            data.push_back(std::make_unique<ThreadData>(i, &Q, &K, &V, &WQ[i], &WK[i], &WV[i], d_k, &heads[i]));
            pthread_create(&threads[i], nullptr, &MultiHeadAttention::thread_func, (void*)data[i].get());
        }

        for (int i = 0; i < num_heads; ++i) {
            pthread_join(threads[i], nullptr);
        }

        Eigen::MatrixXd concat_heads = concatenate_heads(heads);
        DEBUG_PRINT("Exiting MultiHeadAttention operator()");
        return concat_heads * WO;
    }

private:
    int num_heads;
    int d_model;
    int d_k;
    std::vector<Eigen::MatrixXd> WQ, WK, WV;
    Eigen::MatrixXd WO;

    struct ThreadData {
        int head_idx;
        const Eigen::MatrixXd* Q;
        const Eigen::MatrixXd* K;
        const Eigen::MatrixXd* V;
        const Eigen::MatrixXd* WQ;
        const Eigen::MatrixXd* WK;
        const Eigen::MatrixXd* WV;
        double d_k;
        Eigen::MatrixXd* head_output;

        ThreadData(int head_idx, const Eigen::MatrixXd* Q, const Eigen::MatrixXd* K, const Eigen::MatrixXd* V, 
                   const Eigen::MatrixXd* WQ, const Eigen::MatrixXd* WK, const Eigen::MatrixXd* WV, 
                   double d_k, Eigen::MatrixXd* head_output) 
            : head_idx(head_idx), Q(Q), K(K), V(V), WQ(WQ), WK(WK), WV(WV), d_k(d_k), head_output(head_output) {}
    };

    static void* thread_func(void* arg) {
        ThreadData* data = (ThreadData*)arg;
        DEBUG_PRINT("Thread " << data->head_idx << " starting");
        assert(data->Q != nullptr && data->K != nullptr && data->V != nullptr);
        assert(data->WQ != nullptr && data->WK != nullptr && data->WV != nullptr);

        DEBUG_PRINT("Thread " << data->head_idx << " - Q dimensions: " << data->Q->rows() << "x" << data->Q->cols());
        DEBUG_PRINT("Thread " << data->head_idx << " - WQ dimensions: " << data->WQ->rows() << "x" << data->WQ->cols());

        Eigen::MatrixXd Qi = *(data->Q) * *(data->WQ);
        Eigen::MatrixXd Ki = *(data->K) * *(data->WK);
        Eigen::MatrixXd Vi = *(data->V) * *(data->WV);

        DEBUG_PRINT("Thread " << data->head_idx << " - Qi dimensions: " << Qi.rows() << "x" << Qi.cols());
        DEBUG_PRINT("Thread " << data->head_idx << " - Ki dimensions: " << Ki.rows() << "x" << Ki.cols());
        DEBUG_PRINT("Thread " << data->head_idx << " - Vi dimensions: " << Vi.rows() << "x" << Vi.cols());

        assert(Qi.cols() == Ki.cols() && "Qi and Ki must have the same number of columns");
        assert(Qi.rows() == Vi.rows() && "Qi and Vi must have the same number of rows");

        *(data->head_output) = scaled_dot_product_attention(Qi, Ki, Vi, data->d_k);
        DEBUG_PRINT("Thread " << data->head_idx << " finished");
        return nullptr;
    }

    void initialize_weights() {
        for (int i = 0; i < num_heads; ++i) {
            WQ.push_back(Eigen::MatrixXd::Random(d_model, d_k) * 0.01);
            WK.push_back(Eigen::MatrixXd::Random(d_model, d_k) * 0.01);
            WV.push_back(Eigen::MatrixXd::Random(d_model, d_k) * 0.01);
        }
        WO = Eigen::MatrixXd::Random(num_heads * d_k, d_model) * 0.01;
    }

    Eigen::MatrixXd concatenate_heads(const std::vector<Eigen::MatrixXd>& heads) {
        Eigen::MatrixXd concat_heads(heads[0].rows(), heads.size() * heads[0].cols());
        for (int i = 0; i < heads.size(); ++i) {
            concat_heads.block(0, i * heads[0].cols(), heads[0].rows(), heads[0].cols()) = heads[i];
        }
        return concat_heads;
    }
};

// Transformer Encoder Layer
class TransformerEncoderLayer {
public:
    TransformerEncoderLayer(int d_model, int num_heads, int d_ff) : 
        d_model(d_model), num_heads(num_heads), d_ff(d_ff), 
        mha(num_heads, d_model),
        W1(Eigen::MatrixXd::Random(d_model, d_ff) * 0.01),
        W2(Eigen::MatrixXd::Random(d_ff, d_model) * 0.01),
        b1(Eigen::VectorXd::Random(d_ff) * 0.01),
        b2(Eigen::VectorXd::Random(d_model) * 0.01),
        gamma(Eigen::VectorXd::Random(d_model) * 0.01),
        beta(Eigen::VectorXd::Random(d_model) * 0.01) {}

    Eigen::MatrixXd operator()(const Eigen::MatrixXd& input) {
        Eigen::MatrixXd attn_output = mha(input, input, input);
        Eigen::MatrixXd norm1 = layer_norm(attn_output + input, gamma, beta);
        Eigen::MatrixXd ff_output = feed_forward(norm1, W1, W2, b1, b2);
        Eigen::MatrixXd norm2 = layer_norm(ff_output + norm1, gamma, beta);
        return norm2;
    }

private:
    int d_model;
    int num_heads;
    int d_ff;
    MultiHeadAttention mha;
    Eigen::MatrixXd W1, W2;
    Eigen::VectorXd b1, b2, gamma, beta;
};

// Transformer Decoder Layer
class TransformerDecoderLayer {
public:
    TransformerDecoderLayer(int d_model, int num_heads, int d_ff) : 
        d_model(d_model), num_heads(num_heads), d_ff(d_ff), 
        mha1(num_heads, d_model),
        mha2(num_heads, d_model),
        W1(Eigen::MatrixXd::Random(d_model, d_ff) * 0.01),
        W2(Eigen::MatrixXd::Random(d_ff, d_model) * 0.01),
        b1(Eigen::VectorXd::Random(d_ff) * 0.01),
        b2(Eigen::VectorXd::Random(d_model) * 0.01),
        gamma1(Eigen::VectorXd::Random(d_model) * 0.01),
        beta1(Eigen::VectorXd::Random(d_model) * 0.01),
        gamma2(Eigen::VectorXd::Random(d_model) * 0.01),
        beta2(Eigen::VectorXd::Random(d_model) * 0.01) {}

    Eigen::MatrixXd operator()(const Eigen::MatrixXd& target, const Eigen::MatrixXd& memory) {
        Eigen::MatrixXd attn_output1 = mha1(target, target, target);
        Eigen::MatrixXd norm1 = layer_norm(attn_output1 + target, gamma1, beta1);
        Eigen::MatrixXd attn_output2 = mha2(norm1, memory, memory);
        Eigen::MatrixXd norm2 = layer_norm(attn_output2 + norm1, gamma2, beta2);
        Eigen::MatrixXd ff_output = feed_forward(norm2, W1, W2, b1, b2);
        Eigen::MatrixXd norm3 = layer_norm(ff_output + norm2, gamma1, beta1);
        return norm3;
    }

private:
    int d_model;
    int num_heads;
    int d_ff;
    MultiHeadAttention mha1, mha2;
    Eigen::MatrixXd W1, W2;
    Eigen::VectorXd b1, b2, gamma1, beta1, gamma2, beta2;
};

// Full Transformer Model
class Transformer {
public:
    Transformer(int d_model, int num_heads, int d_ff, int num_layers) : 
        d_model(d_model), num_heads(num_heads), d_ff(d_ff), num_layers(num_layers) {
        for (int i = 0; i < num_layers; ++i) {
            encoder_layers.push_back(TransformerEncoderLayer(d_model, num_heads, d_ff));
            decoder_layers.push_back(TransformerDecoderLayer(d_model, num_heads, d_ff));
        }
    }

    Eigen::MatrixXd operator()(const Eigen::MatrixXd& src, const Eigen::MatrixXd& tgt) {
        Eigen::MatrixXd memory = src;
        for (int i = 0; i < num_layers; ++i) {
            memory = encoder_layers[i](memory);
        }
        Eigen::MatrixXd output = tgt;
        for (int i = 0; i < num_layers; ++i) {
            output = decoder_layers[i](output, memory);
        }
        return output;
    }

private:
    int d_model;
    int num_heads;
    int d_ff;
    int num_layers;
    std::vector<TransformerEncoderLayer> encoder_layers;
    std::vector<TransformerDecoderLayer> decoder_layers;
};

int main() {
    int seq_len = 10;
    int d_model = 512;
    int num_heads = 8;
    int d_ff = 2048;
    int num_layers = 6;
    int num_epochs = 100; // Number of epochs for training

    // Generate original sine wave data
    Eigen::MatrixXd original_input(seq_len, d_model);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            original_input(i, j) = sin(2 * M_PI * i / seq_len);
        }
    }

    // Add noisy sine wave data
    Eigen::MatrixXd input(seq_len, d_model);
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            input(i, j) = original_input(i, j) + ((double) rand() / RAND_MAX - 0.5) * 0.1;
        }
    }

    // Normalize input data
    input = (input.array() - input.mean()) / input.array().abs().maxCoeff();

    // Create target matrix (dummy data)
    Eigen::MatrixXd target(seq_len, d_model);
    target.setRandom();

    // Add input embedding matrix that is learnable
    Eigen::MatrixXd input_embedding = Eigen::MatrixXd::Random(d_model, d_model) * 0.01;

    // Apply input embedding
    input = input * input_embedding;

    // Create positional encoding
    auto pos_enc = positional_encoding(seq_len, d_model);

    // Add positional encoding to input
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            input(i, j) += pos_enc[i][j];
        }
    }

    // Add output embedding matrix that is learnable
    Eigen::MatrixXd output_embedding = Eigen::MatrixXd::Random(d_model, d_model) * 0.01;

    // Apply output embedding to target
    target = target * output_embedding;

    // Reflect the autoregressive nature of the transformer (mask future tokens)
    Eigen::MatrixXd mask = Eigen::MatrixXd::Ones(seq_len, seq_len).triangularView<Eigen::Lower>();

    // Create Transformer model
    Transformer transformer(d_model, num_heads, d_ff, num_layers);

    // Initialize weights and bias with smaller values
    Eigen::MatrixXd weights = Eigen::MatrixXd::Random(d_model, d_model) * 0.01;
    Eigen::VectorXd bias = Eigen::VectorXd::Random(d_model) * 0.01;

    // Use smaller learning rate
    double learning_rate = 0.001;

    // File to save the loss values
    std::ofstream loss_file("loss_values.txt");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // Apply the transformer model
        Eigen::MatrixXd output = transformer(input, target);

        // Compute mean squared error loss
        Eigen::MatrixXd error = output - original_input;
        double loss = error.array().square().sum() / seq_len;

        // Save the loss value
        loss_file << loss << std::endl;

        // Backpropagation (SGD)
        Eigen::MatrixXd grad_output = 2 * error / seq_len;
        Eigen::MatrixXd grad_weights = (output.transpose() * grad_output).array() / seq_len;
        Eigen::VectorXd grad_bias = grad_output.colwise().sum().transpose().array() / seq_len;

        // Clip gradients to max norm of 1
        double grad_norm = std::sqrt(grad_weights.array().square().sum() + grad_bias.array().square().sum());
        if (grad_norm > 1.0) {
            grad_weights = grad_weights / grad_norm;
            grad_bias = grad_bias / grad_norm;
        }

        // Update weights and bias
        weights -= learning_rate * grad_weights;
        bias -= learning_rate * grad_bias;

        // Print the loss for the current epoch
        std::cout << "Epoch " << epoch + 1 << "/" << num_epochs << ", Loss: " << loss << std::endl;
    }

    loss_file.close();

    // Final output after training
    Eigen::MatrixXd final_output = transformer(input, target);
    std::cout << "Final Output of Transformer Model:\n" << final_output << std::endl;

    // Save the predicted and original sine wave data to files
    std::ofstream predicted_file("predicted_values.txt");
    std::ofstream original_file("original_values.txt");
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model; ++j) {
            predicted_file << final_output(i, j) << " ";
            original_file << original_input(i, j) << " ";
        }
        predicted_file << std::endl;
        original_file << std::endl;
    }
    predicted_file.close();
    original_file.close();

    return 0;
}
