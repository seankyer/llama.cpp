#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef GGML_USE_CUDA
#    include <cuda_runtime.h>
#endif

static int g_fails = 0;

static void expect(bool cond, const char * msg) {
    if (cond) {
        return;
    }
    fprintf(stderr, "FAIL: %s\n", msg);
    g_fails++;
}

static llama_context_params make_params() {
    llama_context_params params = llama_context_default_params();
    params.n_ctx                = 512;
    params.n_batch              = 512;
    params.n_ubatch             = 512;
    return params;
}

struct KvBuf {
    void * ptr  = nullptr;
    size_t size = 0;

    bool alloc(size_t nbytes) {
        size = nbytes;
#ifdef GGML_USE_CUDA
        if (cudaMalloc(&ptr, nbytes) != cudaSuccess || ptr == nullptr) {
            ptr = nullptr;
            return false;
        }
        return true;
#else
        ptr = aligned_alloc(128, nbytes);
        return ptr != nullptr;
#endif
    }

    void clear(int value) {
#ifdef GGML_USE_CUDA
        cudaMemset(ptr, value, size);
#else
        memset(ptr, value, size);
#endif
    }

    std::vector<uint8_t> read() const {
        std::vector<uint8_t> host(size);
#ifdef GGML_USE_CUDA
        cudaMemcpy(host.data(), ptr, size, cudaMemcpyDeviceToHost);
#else
        memcpy(host.data(), ptr, size);
#endif
        return host;
    }

    ~KvBuf() {
        if (ptr == nullptr) {
            return;
        }
#ifdef GGML_USE_CUDA
        cudaFree(ptr);
#else
        free(ptr);
#endif
        ptr = nullptr;
    }
};

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const char * text) {
    std::vector<llama_token> tokens(256);
    const int32_t            ntok =
        llama_tokenize(vocab, text, (int32_t) strlen(text), tokens.data(), (int32_t) tokens.size(), true, true);
    if (ntok <= 0) {
        return {};
    }
    tokens.resize((size_t) ntok);
    return tokens;
}

static bool decode_tokens(llama_context * ctx, const llama_token * tokens, int32_t ntok) {
    std::vector<llama_token> copy(tokens, tokens + ntok);
    llama_batch              batch = llama_batch_get_one(copy.data(), ntok);
    return llama_decode(ctx, batch) == 0;
}

static std::vector<llama_token> generate(llama_context *                  ctx,
                                         llama_sampler *                  sampler,
                                         const llama_vocab *              vocab,
                                         const std::vector<llama_token> & prompt,
                                         int                              n_new) {
    std::vector<llama_token> out;
    if (!decode_tokens(ctx, prompt.data(), (int32_t) prompt.size())) {
        return {};
    }
    for (int idx = 0; idx < n_new; idx++) {
        const llama_token token = llama_sampler_sample(sampler, ctx, -1);
        out.push_back(token);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }
        if (!decode_tokens(ctx, &token, 1)) {
            return {};
        }
    }
    return out;
}

static const char * model_path_from_args(int argc, char ** argv) {
    const char * path = getenv("LLAMACPP_TEST_MODELFILE");
    for (int idx = 1; idx < argc; idx++) {
        if (strcmp(argv[idx], "-m") == 0 && idx + 1 < argc) {
            path = argv[++idx];
        } else if (argv[idx][0] != '-') {
            path = argv[idx];
        }
    }
    return path;
}

int main(int argc, char ** argv) {
    const char * model_path = model_path_from_args(argc, argv);
    if (model_path == nullptr) {
        fprintf(stderr, "usage: %s [-m] model.gguf\n", argv[0]);
        return 1;
    }

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
#ifdef GGML_USE_CUDA
    model_params.n_gpu_layers = -1;
#else
    model_params.n_gpu_layers = 0;
#endif
    llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (model == nullptr) {
        fprintf(stderr, "failed to load %s\n", model_path);
        llama_backend_free();
        return 1;
    }
    const llama_vocab *            vocab  = llama_model_get_vocab(model);
    const std::vector<llama_token> prompt = tokenize(vocab, "The capital of France is");
    expect(!prompt.empty(), "tokenize");

    llama_context_params params = make_params();
#ifndef GGML_USE_CUDA
    params.offload_kqv = false;
#endif
    const size_t kv_bytes = llama_kv_size(model, params);
    expect(kv_bytes > 0, "llama_kv_size");
    fprintf(stderr, "kv_size %zu\n", kv_bytes);

    llama_sampler * sampler = llama_sampler_init_greedy();
    expect(sampler != nullptr, "sampler");

    params.kv_data        = nullptr;
    params.kv_size        = 0;
    llama_context * owned = llama_init_from_model(model, params);
    expect(owned != nullptr, "owned context");
    const size_t owned_self = owned == nullptr ? 0 : llama_kv_self_size(owned);
    expect(owned_self == kv_bytes, "owned context allocates the kv cache");

    KvBuf buf;
    expect(buf.alloc(kv_bytes), "allocate caller buffer");
    if (buf.ptr != nullptr) {
        buf.clear(0xAB);
        llama_context_params external_params = params;
        external_params.kv_data              = buf.ptr;
        external_params.kv_size              = kv_bytes;
        llama_context * bound                = llama_init_from_model(model, external_params);
        expect(bound != nullptr, "bind context");
        if (bound != nullptr) {
            expect(llama_kv_self_size(bound) == 0, "bound context owns no kv bytes");
            llama_free(bound);
        }
        const std::vector<uint8_t> intact = buf.read();
        bool                       still  = !intact.empty();
        for (uint8_t byte : intact) {
            if (byte != 0xAB) {
                still = false;
                break;
            }
        }
        expect(still, "destroy leaves the caller buffer intact");

        buf.clear(0);
        const std::vector<uint8_t> before = buf.read();
        external_params.kv_meta           = nullptr;
        external_params.kv_meta_size      = 0;
        llama_context * first             = llama_init_from_model(model, external_params);
        expect(first != nullptr, "external context");
        std::vector<llama_token> external_tokens;
        std::vector<uint8_t>     meta;
        llama_token              replay = 0;
        if (first != nullptr) {
            expect(decode_tokens(first, prompt.data(), (int32_t) prompt.size()), "external prompt");
            replay = llama_sampler_sample(sampler, first, -1);
            external_tokens.push_back(replay);
            const size_t meta_bytes = llama_kv_meta_size(first);
            meta.resize(meta_bytes);
            expect(meta_bytes > 0 && llama_kv_meta_get(first, meta.data(), meta.size()) == meta_bytes, "meta get");
            expect(llama_kv_self_size(first) == 0, "external self size stays 0");
            llama_free(first);
        }
        const std::vector<uint8_t> after_free = buf.read();
        expect(after_free != before, "window bytes change across a turn");

        external_params.kv_meta                  = meta.data();
        external_params.kv_meta_size             = meta.size();
        const std::vector<uint8_t> before_second = buf.read();
        llama_context *            second        = llama_init_from_model(model, external_params);
        expect(second != nullptr, "second context on the same pointer");
        if (second != nullptr) {
            expect(buf.read() == before_second, "second create does not copy or clear the cache");
            expect(llama_kv_self_size(second) == 0, "second context owns no kv bytes");
            expect(decode_tokens(second, &replay, 1), "replay");
            for (int idx = 1; idx < 8; idx++) {
                const llama_token token = llama_sampler_sample(sampler, second, -1);
                external_tokens.push_back(token);
                if (llama_vocab_is_eog(vocab, token)) {
                    break;
                }
                if (!decode_tokens(second, &token, 1)) {
                    expect(false, "external continuation");
                    break;
                }
            }
            llama_free(second);
        }

        if (owned != nullptr) {
            const std::vector<llama_token> owned_tokens = generate(owned, sampler, vocab, prompt, 8);
            expect(owned_tokens == external_tokens, "owned and external tokens match");
        }
    }

    if (owned != nullptr) {
        llama_free(owned);
    }

    KvBuf tiny;
    expect(tiny.alloc(4096), "tiny buffer");
    if (tiny.ptr != nullptr) {
        llama_context_params too_small = params;
        too_small.kv_data              = tiny.ptr;
        too_small.kv_size              = tiny.size;
        too_small.kv_meta              = nullptr;
        too_small.kv_meta_size         = 0;
        llama_context * rejected       = llama_init_from_model(model, too_small);
        expect(rejected == nullptr, "too-small buffer fails init");
        if (rejected != nullptr) {
            llama_free(rejected);
        }
    }

    llama_sampler_free(sampler);
    llama_model_free(model);
    llama_backend_free();
    if (g_fails != 0) {
        fprintf(stderr, "%d failure(s)\n", g_fails);
        return 1;
    }
    fprintf(stderr, "ok\n");
    return 0;
}
