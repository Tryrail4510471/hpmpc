#pragma once

#include "../config.h"
#include "../datatypes/Additive_Share.hpp"
#include "../datatypes/float_fixed_converter.hpp"
#include "../protocols/Protocols.h"
#include "functions/GEMM.hpp"
#include "functions/max_min.hpp"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>

#if FUNCTION_IDENTIFIER == 87
#define FUNCTION transformer_inference
#endif

#define RESULTTYPE DATATYPE

namespace ppti_tinybert
{
#ifndef PPTI_SEQ_LEN
constexpr int SEQ_LEN = 4;
#else
constexpr int SEQ_LEN = PPTI_SEQ_LEN;
#endif

#ifndef PPTI_HIDDEN
constexpr int HIDDEN = 8;
#else
constexpr int HIDDEN = PPTI_HIDDEN;
#endif

#ifndef PPTI_NUM_HEADS
constexpr int NUM_HEADS = 2;
#else
constexpr int NUM_HEADS = PPTI_NUM_HEADS;
#endif

#ifndef PPTI_NUM_LAYERS
constexpr int NUM_LAYERS = 4;
#else
constexpr int NUM_LAYERS = PPTI_NUM_LAYERS;
#endif

#ifndef PPTI_FFN_HIDDEN
constexpr int FFN_HIDDEN = 16;
#else
constexpr int FFN_HIDDEN = PPTI_FFN_HIDDEN;
#endif

static_assert(HIDDEN % NUM_HEADS == 0, "PPTI TinyBERT requires HIDDEN to be divisible by NUM_HEADS.");
constexpr int HEAD_DIM = HIDDEN / NUM_HEADS;
constexpr int LAYER_WEIGHT_STRIDE = 10000;

inline int linear_param_count(int in_features, int out_features)
{
    return in_features * out_features + out_features;
}

inline int expected_model_params()
{
    const int attention_params = 4 * linear_param_count(HIDDEN, HIDDEN);
    const int ffn_params = linear_param_count(HIDDEN, FFN_HIDDEN) + linear_param_count(FFN_HIDDEN, HIDDEN);
    const int layer_norm_params = 4 * HIDDEN;
    return NUM_LAYERS * (attention_params + ffn_params + layer_norm_params);
}

struct ModelWeights
{
    bool loaded = false;
    bool attempted = false;
    std::vector<float> params;
    std::size_t cursor = 0;
};

struct InputIds
{
    bool loaded = false;
    bool attempted = false;
    std::vector<int> token_ids;
    std::vector<int> token_type_ids;
    std::vector<int> position_ids;
    std::vector<int> attention_mask;
};

struct EmbeddingWeights
{
    bool loaded = false;
    bool attempted = false;
    int vocab_size = 0;
    int max_position = 0;
    int type_vocab_size = 0;
    int hidden = 0;
    std::vector<float> word_embeddings;
    std::vector<float> position_embeddings;
    std::vector<float> token_type_embeddings;
    std::vector<float> layer_norm_gamma;
    std::vector<float> layer_norm_beta;
};

inline UINT_TYPE fixed(float value, int frac_bits = FRACTIONAL)
{
    return FloatFixedConverter<FLOATTYPE, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(value, frac_bits);
}

inline float input_value(int i)
{
    return ((i % 7) - 3) * 0.0625f;
}

inline float weight_value(int i)
{
    return ((i % 11) - 5) * 0.03125f;
}

inline std::string model_file_path()
{
    const char* ppti_model_file = std::getenv("PPTI_MODEL_FILE");
    if (ppti_model_file != nullptr)
        return std::string(ppti_model_file);

    const char* model_file = std::getenv("MODEL_FILE");
    if (model_file == nullptr)
        return "";

    std::string file(model_file);
    if (!file.empty() && file[0] == '/')
        return file;

    const char* model_dir = std::getenv("MODEL_DIR");
    if (model_dir == nullptr)
        return file;

    return std::string(model_dir) + "/" + file;
}

inline std::string env_file_path(const char* env_name)
{
    const char* value = std::getenv(env_name);
    if (value == nullptr)
        return "";
    return std::string(value);
}

inline ModelWeights& model_weights()
{
    static ModelWeights weights;
    if (weights.attempted)
        return weights;

    weights.attempted = true;
    const std::string path = model_file_path();
    if (path.empty())
        return weights;

    std::ifstream in(path, std::ios::binary);
    if (!in)
        return weights;

    int total_params = 0;
    in.read(reinterpret_cast<char*>(&total_params), sizeof(int));
    if (total_params != expected_model_params())
        return weights;

    weights.params.resize(total_params);
    in.read(reinterpret_cast<char*>(weights.params.data()), sizeof(float) * total_params);
    weights.loaded = static_cast<int>(in.gcount()) == static_cast<int>(sizeof(float) * total_params);
    if (!weights.loaded)
        weights.params.clear();
    return weights;
}

inline bool take_weight_values(std::vector<float>& values)
{
    ModelWeights& weights = model_weights();
    if (!weights.loaded)
        return false;
    if (weights.cursor + values.size() > weights.params.size())
        return false;

    std::copy(weights.params.begin() + weights.cursor, weights.params.begin() + weights.cursor + values.size(), values.begin());
    weights.cursor += values.size();
    return true;
}

inline void reset_weight_cursor()
{
    model_weights().cursor = 0;
}

inline InputIds& input_ids()
{
    static InputIds input;
    if (input.attempted)
        return input;

    input.attempted = true;
    const std::string path = env_file_path("PPTI_INPUT_FILE");
    if (path.empty())
        return input;

    std::ifstream in(path, std::ios::binary);
    if (!in)
        return input;

    int seq_len = 0;
    in.read(reinterpret_cast<char*>(&seq_len), sizeof(int));
    if (seq_len != SEQ_LEN)
        return input;

    input.token_ids.resize(SEQ_LEN);
    input.token_type_ids.resize(SEQ_LEN);
    input.position_ids.resize(SEQ_LEN);
    input.attention_mask.resize(SEQ_LEN);
    in.read(reinterpret_cast<char*>(input.token_ids.data()), sizeof(int) * SEQ_LEN);
    in.read(reinterpret_cast<char*>(input.token_type_ids.data()), sizeof(int) * SEQ_LEN);
    in.read(reinterpret_cast<char*>(input.position_ids.data()), sizeof(int) * SEQ_LEN);
    in.read(reinterpret_cast<char*>(input.attention_mask.data()), sizeof(int) * SEQ_LEN);
    input.loaded = static_cast<bool>(in);
    if (!input.loaded)
    {
        input.token_ids.clear();
        input.token_type_ids.clear();
        input.position_ids.clear();
        input.attention_mask.clear();
    }
    return input;
}

inline EmbeddingWeights& embedding_weights()
{
    static EmbeddingWeights embeddings;
    if (embeddings.attempted)
        return embeddings;

    embeddings.attempted = true;
    const std::string path = env_file_path("PPTI_EMBEDDING_FILE");
    if (path.empty())
        return embeddings;

    std::ifstream in(path, std::ios::binary);
    if (!in)
        return embeddings;

    int total_params = 0;
    in.read(reinterpret_cast<char*>(&total_params), sizeof(int));
    in.read(reinterpret_cast<char*>(&embeddings.vocab_size), sizeof(int));
    in.read(reinterpret_cast<char*>(&embeddings.max_position), sizeof(int));
    in.read(reinterpret_cast<char*>(&embeddings.type_vocab_size), sizeof(int));
    in.read(reinterpret_cast<char*>(&embeddings.hidden), sizeof(int));
    if (embeddings.hidden != HIDDEN || embeddings.vocab_size <= 0 || embeddings.max_position < SEQ_LEN ||
        embeddings.type_vocab_size <= 0)
        return embeddings;

    const int expected_params = embeddings.vocab_size * HIDDEN + embeddings.max_position * HIDDEN +
                                embeddings.type_vocab_size * HIDDEN + 2 * HIDDEN;
    if (total_params != expected_params)
        return embeddings;

    embeddings.word_embeddings.resize(embeddings.vocab_size * HIDDEN);
    embeddings.position_embeddings.resize(embeddings.max_position * HIDDEN);
    embeddings.token_type_embeddings.resize(embeddings.type_vocab_size * HIDDEN);
    embeddings.layer_norm_gamma.resize(HIDDEN);
    embeddings.layer_norm_beta.resize(HIDDEN);
    in.read(reinterpret_cast<char*>(embeddings.word_embeddings.data()), sizeof(float) * embeddings.word_embeddings.size());
    in.read(reinterpret_cast<char*>(embeddings.position_embeddings.data()), sizeof(float) * embeddings.position_embeddings.size());
    in.read(reinterpret_cast<char*>(embeddings.token_type_embeddings.data()), sizeof(float) * embeddings.token_type_embeddings.size());
    in.read(reinterpret_cast<char*>(embeddings.layer_norm_gamma.data()), sizeof(float) * HIDDEN);
    in.read(reinterpret_cast<char*>(embeddings.layer_norm_beta.data()), sizeof(float) * HIDDEN);
    embeddings.loaded = static_cast<bool>(in);
    if (!embeddings.loaded)
    {
        embeddings.word_embeddings.clear();
        embeddings.position_embeddings.clear();
        embeddings.token_type_embeddings.clear();
        embeddings.layer_norm_gamma.clear();
        embeddings.layer_norm_beta.clear();
    }
    return embeddings;
}

inline bool build_embedding_inputs(std::vector<float>& values)
{
    InputIds& input = input_ids();
    EmbeddingWeights& embeddings = embedding_weights();
    if (!input.loaded || !embeddings.loaded)
        return false;

    values.assign(SEQ_LEN * HIDDEN, 0.0f);
    for (int r = 0; r < SEQ_LEN; r++)
    {
        int token_id = input.token_ids[r];
        int token_type_id = input.token_type_ids[r];
        int position_id = input.position_ids[r];
        if (token_id < 0 || token_id >= embeddings.vocab_size || token_type_id < 0 ||
            token_type_id >= embeddings.type_vocab_size || position_id < 0 || position_id >= embeddings.max_position)
            return false;

        float mean = 0.0f;
        for (int c = 0; c < HIDDEN; c++)
        {
            float value = embeddings.word_embeddings[token_id * HIDDEN + c] +
                          embeddings.position_embeddings[position_id * HIDDEN + c] +
                          embeddings.token_type_embeddings[token_type_id * HIDDEN + c];
            values[r * HIDDEN + c] = value;
            mean += value;
        }
        mean /= static_cast<float>(HIDDEN);

        float variance = 0.0f;
        for (int c = 0; c < HIDDEN; c++)
        {
            float centered = values[r * HIDDEN + c] - mean;
            variance += centered * centered;
        }
        variance = variance / static_cast<float>(HIDDEN) + 0.001f;
        float inv_std = 1.0f / std::sqrt(variance);

        for (int c = 0; c < HIDDEN; c++)
        {
            float centered = values[r * HIDDEN + c] - mean;
            values[r * HIDDEN + c] = centered * inv_std * embeddings.layer_norm_gamma[c] + embeddings.layer_norm_beta[c];
        }
    }
    return true;
}

template <typename A>
void load_inputs(std::vector<A>& x)
{
    std::vector<float> embedding_values;
    const bool from_embedding_files = build_embedding_inputs(embedding_values);
    for (int i = 0; i < static_cast<int>(x.size()); i++)
    {
        UINT_TYPE value = fixed(from_embedding_files ? embedding_values[i] : input_value(i));
#if DATAOWNER == -1
        x[i] = A(value);
#else
        x[i].template prepare_receive_and_replicate<DATAOWNER>(value);
#endif
    }
#if DATAOWNER != -1
    A::communicate();
    for (int i = 0; i < static_cast<int>(x.size()); i++)
        x[i].template complete_receive_from<DATAOWNER>();
#endif
}

template <typename A>
void load_weights(std::vector<A>& w, int offset)
{
    std::vector<float> values(w.size());
    const bool from_file = take_weight_values(values);
    for (int i = 0; i < static_cast<int>(w.size()); i++)
    {
        UINT_TYPE value = fixed(from_file ? values[i] : weight_value(i + offset));
#if MODELOWNER == -1
        w[i] = A(value);
#else
        w[i].template prepare_receive_and_replicate<MODELOWNER>(value);
#endif
    }
#if MODELOWNER != -1
    A::communicate();
    for (int i = 0; i < static_cast<int>(w.size()); i++)
        w[i].template complete_receive_from<MODELOWNER>();
#endif
}

template <typename A>
void load_layer_norm_params(std::vector<A>& gamma, std::vector<A>& beta, int offset)
{
    std::vector<float> gamma_values(gamma.size());
    std::vector<float> beta_values(beta.size());
    const bool gamma_from_file = take_weight_values(gamma_values);
    const bool beta_from_file = take_weight_values(beta_values);
    for (int i = 0; i < static_cast<int>(gamma.size()); i++)
    {
        UINT_TYPE gamma_value = fixed(gamma_from_file ? gamma_values[i] : 1.0f + weight_value(offset + i));
        UINT_TYPE beta_value = fixed(beta_from_file ? beta_values[i] : weight_value(offset + 1000 + i));
#if MODELOWNER == -1
        gamma[i] = A(gamma_value);
        beta[i] = A(beta_value);
#else
        gamma[i].template prepare_receive_and_replicate<MODELOWNER>(gamma_value);
        beta[i].template prepare_receive_and_replicate<MODELOWNER>(beta_value);
#endif
    }
#if MODELOWNER != -1
    A::communicate();
    for (int i = 0; i < static_cast<int>(gamma.size()); i++)
    {
        gamma[i].template complete_receive_from<MODELOWNER>();
        beta[i].template complete_receive_from<MODELOWNER>();
    }
#endif
}

template <typename A>
void load_bias(std::vector<A>& bias, int offset)
{
    std::vector<float> values(bias.size());
    const bool from_file = take_weight_values(values);
    for (int i = 0; i < static_cast<int>(bias.size()); i++)
    {
        UINT_TYPE value = fixed(from_file ? values[i] : weight_value(i + offset));
#if MODELOWNER == -1
        bias[i] = A(value);
#else
        bias[i].template prepare_receive_and_replicate<MODELOWNER>(value);
#endif
    }
#if MODELOWNER != -1
    A::communicate();
    for (int i = 0; i < static_cast<int>(bias.size()); i++)
        bias[i].template complete_receive_from<MODELOWNER>();
#endif
}

template <typename A>
void secure_matmul(std::vector<A>& lhs,
                   std::vector<A>& rhs,
                   std::vector<A>& out,
                   int rows,
                   int inner,
                   int cols)
{
    std::fill(out.begin(), out.end(), A(0));
    prepare_GEMM(lhs.data(), rhs.data(), out.data(), rows, cols, inner, false);
    A::communicate();
    complete_GEMM(out.data(), rows * cols);
}

template <typename A>
void add_bias(std::vector<A>& matrix, const std::vector<A>& bias, int rows, int cols)
{
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            matrix[r * cols + c] += bias[c];
}

template <typename A>
void transpose(const std::vector<A>& in, std::vector<A>& out, int rows, int cols)
{
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            out[c * rows + r] = in[r * cols + c];
}

template <typename A>
void extract_head(const std::vector<A>& in, std::vector<A>& out, int head)
{
    for (int r = 0; r < SEQ_LEN; r++)
        for (int c = 0; c < HEAD_DIM; c++)
            out[r * HEAD_DIM + c] = in[r * HIDDEN + head * HEAD_DIM + c];
}

template <typename A>
void place_head(const std::vector<A>& in, std::vector<A>& out, int head)
{
    for (int r = 0; r < SEQ_LEN; r++)
        for (int c = 0; c < HEAD_DIM; c++)
            out[r * HIDDEN + head * HEAD_DIM + c] = in[r * HEAD_DIM + c];
}

template <typename A>
void scale_public(std::vector<A>& values, float scale)
{
    UINT_TYPE fixed_scale = fixed(scale);
    for (auto& value : values)
        value = value.prepare_mult_public_fixed(fixed_scale);
    A::communicate();
    for (auto& value : values)
        value.complete_public_mult_fixed();
}

template <typename A>
void secure_gelu_poly(std::vector<A>& values)
{
    std::vector<A> squared(values.size());
    for (int i = 0; i < static_cast<int>(values.size()); i++)
    {
        squared[i] = values[i].prepare_dot(values[i]);
        squared[i].mask_and_send_dot();
    }
    A::communicate();
    for (auto& value : squared)
        value.complete_mult();

    std::vector<A> cubic(values.size());
    for (int i = 0; i < static_cast<int>(values.size()); i++)
    {
        cubic[i] = squared[i].prepare_dot(values[i]);
        cubic[i].mask_and_send_dot();
    }
    A::communicate();
    for (auto& value : cubic)
        value.complete_mult();

    const UINT_TYPE half = fixed(0.5f);
    const UINT_TYPE eighth = fixed(0.125f);
    for (auto& value : values)
        value = value.prepare_mult_public_fixed(half);
    for (auto& value : cubic)
        value = value.prepare_mult_public_fixed(eighth);
    A::communicate();
    for (auto& value : values)
        value.complete_public_mult_fixed();
    for (auto& value : cubic)
        value.complete_public_mult_fixed();

    for (int i = 0; i < static_cast<int>(values.size()); i++)
        values[i] += cubic[i];
}

template <typename A>
void reciprocal_newton(std::vector<A>& values, std::vector<A>& reciprocal, int iterations, float initial_guess)
{
    reciprocal.assign(values.size(), A(fixed(initial_guess)));
    for (int iter = 0; iter < iterations; iter++)
    {
        std::vector<A> product(values.size());
        for (int i = 0; i < static_cast<int>(values.size()); i++)
        {
            product[i] = values[i].prepare_dot(reciprocal[i]);
            product[i].mask_and_send_dot();
        }
        A::communicate();
        for (auto& value : product)
            value.complete_mult();

        for (auto& value : product)
            value = A(fixed(2.0f)) - value;

        for (int i = 0; i < static_cast<int>(values.size()); i++)
        {
            reciprocal[i] = reciprocal[i].prepare_dot(product[i]);
            reciprocal[i].mask_and_send_dot();
        }
        A::communicate();
        for (auto& value : reciprocal)
            value.complete_mult();
    }
}

template <typename A>
void reciprocal_sqrt_newton(std::vector<A>& values, std::vector<A>& reciprocal_sqrt, int iterations, float initial_guess)
{
    reciprocal_sqrt.assign(values.size(), A(fixed(initial_guess)));
    for (int iter = 0; iter < iterations; iter++)
    {
        std::vector<A> y_squared(values.size());
        for (int i = 0; i < static_cast<int>(values.size()); i++)
        {
            y_squared[i] = reciprocal_sqrt[i].prepare_dot(reciprocal_sqrt[i]);
            y_squared[i].mask_and_send_dot();
        }
        A::communicate();
        for (auto& value : y_squared)
            value.complete_mult();

        std::vector<A> xy_squared(values.size());
        for (int i = 0; i < static_cast<int>(values.size()); i++)
        {
            xy_squared[i] = values[i].prepare_dot(y_squared[i]);
            xy_squared[i].mask_and_send_dot();
        }
        A::communicate();
        for (auto& value : xy_squared)
            value.complete_mult();

        const UINT_TYPE half = fixed(0.5f);
        for (auto& value : xy_squared)
            value = value.prepare_mult_public_fixed(half);
        A::communicate();
        for (auto& value : xy_squared)
            value.complete_public_mult_fixed();

        for (auto& value : xy_squared)
            value = A(fixed(1.5f)) - value;

        for (int i = 0; i < static_cast<int>(values.size()); i++)
        {
            reciprocal_sqrt[i] = reciprocal_sqrt[i].prepare_dot(xy_squared[i]);
            reciprocal_sqrt[i].mask_and_send_dot();
        }
        A::communicate();
        for (auto& value : reciprocal_sqrt)
            value.complete_mult();
    }
}

template <typename A>
void secure_rowwise_layer_norm(std::vector<A>& values, const std::vector<A>& gamma, const std::vector<A>& beta, int rows, int cols)
{
    std::vector<A> mean(rows, A(0));
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            mean[r] += values[r * cols + c];

    const UINT_TYPE inv_cols = fixed(1.0f / static_cast<float>(cols));
    for (auto& value : mean)
        value = value.prepare_mult_public_fixed(inv_cols);
    A::communicate();
    for (auto& value : mean)
        value.complete_public_mult_fixed();

    std::vector<A> centered(values.size());
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            centered[r * cols + c] = values[r * cols + c] - mean[r];

    std::vector<A> squared(centered.size());
    for (int i = 0; i < static_cast<int>(centered.size()); i++)
    {
        squared[i] = centered[i].prepare_dot(centered[i]);
        squared[i].mask_and_send_dot();
    }
    A::communicate();
    for (auto& value : squared)
        value.complete_mult();

    std::vector<A> variance(rows, A(0));
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            variance[r] += squared[r * cols + c];

    for (auto& value : variance)
        value = value.prepare_mult_public_fixed(inv_cols);
    A::communicate();
    for (auto& value : variance)
        value.complete_public_mult_fixed();

    const A epsilon(fixed(0.001f));
    for (auto& value : variance)
        value += epsilon;

    std::vector<A> inv_std;
    reciprocal_sqrt_newton(variance, inv_std, 3, 1.0f);

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            int idx = r * cols + c;
            values[idx] = centered[idx].prepare_dot(inv_std[r]);
            values[idx].mask_and_send_dot();
        }
    }
    A::communicate();
    for (auto& value : values)
        value.complete_mult();

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            int idx = r * cols + c;
            values[idx] = values[idx].prepare_dot(gamma[c]);
            values[idx].mask_and_send_dot();
        }
    }
    A::communicate();
    for (auto& value : values)
        value.complete_mult();

    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            values[r * cols + c] += beta[c];
}

template <typename A>
void secure_rowwise_softmax_poly(std::vector<A>& scores, int rows, int cols)
{
    std::vector<A> row_max(rows);
    max_min_sint<0, BITLENGTH>(scores.data(), cols, row_max.data(), rows, true);

    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            scores[r * cols + c] = scores[r * cols + c] - row_max[r];

    std::vector<A> squared(scores.size());
    for (int i = 0; i < static_cast<int>(scores.size()); i++)
    {
        squared[i] = scores[i].prepare_dot(scores[i]);
        squared[i].mask_and_send_dot();
    }
    A::communicate();
    for (auto& value : squared)
        value.complete_mult();

    const UINT_TYPE half = fixed(0.5f);
    for (auto& value : squared)
        value = value.prepare_mult_public_fixed(half);
    A::communicate();
    for (auto& value : squared)
        value.complete_public_mult_fixed();

    // Current exp approximation: exp(x) ~= 1 + x + 0.5*x^2 after row-max stabilization.
    for (int i = 0; i < static_cast<int>(scores.size()); i++)
        scores[i] = A(fixed(1.0f)) + scores[i] + squared[i];

    std::vector<A> row_sum(rows, A(0));
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            row_sum[r] += scores[r * cols + c];

    std::vector<A> inv_sum;
    reciprocal_newton(row_sum, inv_sum, 3, 1.0f / static_cast<float>(cols));

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            int idx = r * cols + c;
            scores[idx] = scores[idx].prepare_dot(inv_sum[r]);
            scores[idx].mask_and_send_dot();
        }
    }
    A::communicate();
    for (auto& value : scores)
        value.complete_mult();
}

template <typename A>
void secure_multi_head_attention(std::vector<A>& x,
                                 std::vector<A>& wq,
                                 std::vector<A>& wk,
                                 std::vector<A>& wv,
                                 std::vector<A>& wo,
                                 std::vector<A>& bq,
                                 std::vector<A>& bk,
                                 std::vector<A>& bv,
                                 std::vector<A>& bo,
                                 std::vector<A>& out)
{
    std::vector<A> q(SEQ_LEN * HIDDEN), k(SEQ_LEN * HIDDEN), v(SEQ_LEN * HIDDEN);
    secure_matmul(x, wq, q, SEQ_LEN, HIDDEN, HIDDEN);
    secure_matmul(x, wk, k, SEQ_LEN, HIDDEN, HIDDEN);
    secure_matmul(x, wv, v, SEQ_LEN, HIDDEN, HIDDEN);
    add_bias(q, bq, SEQ_LEN, HIDDEN);
    add_bias(k, bk, SEQ_LEN, HIDDEN);
    add_bias(v, bv, SEQ_LEN, HIDDEN);

    std::vector<A> concat_context(SEQ_LEN * HIDDEN, A(0));
    for (int head = 0; head < NUM_HEADS; head++)
    {
        std::vector<A> qh(SEQ_LEN * HEAD_DIM), kh(SEQ_LEN * HEAD_DIM), vh(SEQ_LEN * HEAD_DIM);
        std::vector<A> kh_t(HEAD_DIM * SEQ_LEN), scores(SEQ_LEN * SEQ_LEN), context(SEQ_LEN * HEAD_DIM);

        extract_head(q, qh, head);
        extract_head(k, kh, head);
        extract_head(v, vh, head);

        transpose(kh, kh_t, SEQ_LEN, HEAD_DIM);
        secure_matmul(qh, kh_t, scores, SEQ_LEN, HEAD_DIM, SEQ_LEN);
        scale_public(scores, 1.0f / std::sqrt(static_cast<float>(HEAD_DIM)));
        secure_rowwise_softmax_poly(scores, SEQ_LEN, SEQ_LEN);
        secure_matmul(scores, vh, context, SEQ_LEN, SEQ_LEN, HEAD_DIM);
        place_head(context, concat_context, head);
    }

    secure_matmul(concat_context, wo, out, SEQ_LEN, HIDDEN, HIDDEN);
    add_bias(out, bo, SEQ_LEN, HIDDEN);
}

template <typename A>
void secure_tinybert_encoder_layer(std::vector<A>& hidden, int layer)
{
    const int base = layer * LAYER_WEIGHT_STRIDE;
    std::vector<A> wq(HIDDEN * HIDDEN), wk(HIDDEN * HIDDEN), wv(HIDDEN * HIDDEN), wo(HIDDEN * HIDDEN);
    std::vector<A> bq(HIDDEN), bk(HIDDEN), bv(HIDDEN), bo(HIDDEN);
    std::vector<A> w1(HIDDEN * FFN_HIDDEN), w2(FFN_HIDDEN * HIDDEN);
    std::vector<A> b1(FFN_HIDDEN), b2(HIDDEN);
    std::vector<A> attn_gamma(HIDDEN), attn_beta(HIDDEN), ffn_gamma(HIDDEN), ffn_beta(HIDDEN);

    load_weights(wq, base + 0);
    load_bias(bq, base + 800);
    load_weights(wk, base + 1000);
    load_bias(bk, base + 1800);
    load_weights(wv, base + 2000);
    load_bias(bv, base + 2800);
    load_weights(wo, base + 3000);
    load_bias(bo, base + 3800);
    load_weights(w1, base + 4000);
    load_bias(b1, base + 4800);
    load_weights(w2, base + 5000);
    load_bias(b2, base + 5800);
    load_layer_norm_params(attn_gamma, attn_beta, base + 6000);
    load_layer_norm_params(ffn_gamma, ffn_beta, base + 7000);

    std::vector<A> attn_out(SEQ_LEN * HIDDEN), attn_residual(SEQ_LEN * HIDDEN);
    secure_multi_head_attention(hidden, wq, wk, wv, wo, bq, bk, bv, bo, attn_out);
    for (int i = 0; i < static_cast<int>(hidden.size()); i++)
        attn_residual[i] = hidden[i] + attn_out[i];
    secure_rowwise_layer_norm(attn_residual, attn_gamma, attn_beta, SEQ_LEN, HIDDEN);

    std::vector<A> ffn_hidden(SEQ_LEN * FFN_HIDDEN), ffn_out(SEQ_LEN * HIDDEN);
    secure_matmul(attn_residual, w1, ffn_hidden, SEQ_LEN, HIDDEN, FFN_HIDDEN);
    add_bias(ffn_hidden, b1, SEQ_LEN, FFN_HIDDEN);
    secure_gelu_poly(ffn_hidden);
    secure_matmul(ffn_hidden, w2, ffn_out, SEQ_LEN, FFN_HIDDEN, HIDDEN);
    add_bias(ffn_out, b2, SEQ_LEN, HIDDEN);

    for (int i = 0; i < static_cast<int>(hidden.size()); i++)
        hidden[i] = attn_residual[i] + ffn_out[i];
    secure_rowwise_layer_norm(hidden, ffn_gamma, ffn_beta, SEQ_LEN, HIDDEN);
}
}  // namespace ppti_tinybert

template <typename Share>
void transformer_inference(DATATYPE* res)
{
    using A = Additive_Share<DATATYPE, Share>;
    using namespace ppti_tinybert;

    std::vector<A> hidden(SEQ_LEN * HIDDEN);
    reset_weight_cursor();
    if (model_weights().loaded)
    {
        print_online("PPTI TinyBERT: loaded model weights from file.");
    }
    else
    {
        print_online("PPTI TinyBERT: using deterministic dummy weights.");
    }
    if (embedding_weights().loaded)
    {
        print_online("PPTI TinyBERT: loaded embedding weights from file.");
    }
    else
    {
        print_online("PPTI TinyBERT: using deterministic dummy embeddings.");
    }
    if (input_ids().loaded)
    {
        print_online("PPTI TinyBERT: loaded input ids from file.");
    }
    else
    {
        print_online("PPTI TinyBERT: using deterministic dummy input ids.");
    }
    if (std::getenv("PPTI_VALIDATE_MODEL_ONLY") != nullptr)
    {
        bool model_ok = model_weights().loaded || env_file_path("PPTI_MODEL_FILE").empty();
        bool embeddings_ok = embedding_weights().loaded || env_file_path("PPTI_EMBEDDING_FILE").empty();
        bool input_ok = input_ids().loaded || env_file_path("PPTI_INPUT_FILE").empty();
        res[0] = (model_ok && embeddings_ok && input_ok) ? 1 : 0;
        print_online("PPTI TinyBERT: model-file validation only.");
        return;
    }
    print_online("PPTI TinyBERT: sharing input embeddings...");
    load_inputs(hidden);

    for (int layer = 0; layer < NUM_LAYERS; layer++)
    {
        print_online("PPTI TinyBERT: encoder layer...");
        secure_tinybert_encoder_layer(hidden, layer);
    }

    hidden[0].prepare_reveal_to_all();
    A::communicate();
    res[0] = hidden[0].complete_reveal_to_all();
    print_online("PPTI TinyBERT smoke test completed.");
}
