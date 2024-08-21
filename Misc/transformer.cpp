#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <pthread.h>
#include <memory>  // Include memory header for smart pointers


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
            WQ.push_back(Eigen::MatrixXd::Random(d_model, d_k));
            WK.push_back(Eigen::MatrixXd::Random(d_model, d_k));
            WV.push_back(Eigen::MatrixXd::Random(d_model, d_k));
        }
        WO = Eigen::MatrixXd::Random(num_heads * d_k, d_model);
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
        W1(Eigen::MatrixXd::Random(d_model, d_ff)),
        W2(Eigen::MatrixXd::Random(d_ff, d_model)),
        b1(Eigen::VectorXd::Random(d_ff)),
        b2(Eigen::VectorXd::Random(d_model)),
        gamma(Eigen::VectorXd::Random(d_model)),
        beta(Eigen::VectorXd::Random(d_model)) {}

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
        W1(Eigen::MatrixXd::Random(d_model, d_ff)),
        W2(Eigen::MatrixXd::Random(d_ff, d_model)),
        b1(Eigen::VectorXd::Random(d_ff)),
        b2(Eigen::VectorXd::Random(d_model)),
        gamma1(Eigen::VectorXd::Random(d_model)),
        beta1(Eigen::VectorXd::Random(d_model)),
        gamma2(Eigen::VectorXd::Random(d_model)),
        beta2(Eigen::VectorXd::Random(d_model)) {}

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

    // TODO: Add meaningful data (perhaps a noisy sine wave to start)

    // TODO: Add input embedding matrix that is learnable

    // Create positional encoding
    auto pos_enc = positional_encoding(seq_len, d_model);

    // Create input matrix (dummy data)
    Eigen::MatrixXd input(seq_len, d_model);
    input.setRandom();

    // Create target matrix (dummy data)
    Eigen::MatrixXd target(seq_len, d_model);
    target.setRandom();

    // TODO: Add output embedding matrix that is learnable

    // TODO: Reflect the autoregressive nature of the transformer (mask future tokens)

    // TODO: Implement Stochastic Gradient Descent for Cross Entropy Loss

    // Create Transformer model
    Transformer transformer(d_model, num_heads, d_ff, num_layers);

    // Apply the transformer model
    Eigen::MatrixXd output = transformer(input, target);

    // Print the output
    std::cout << "Output of Transformer Model:\n" << output << std::endl;

    return 0;
}
