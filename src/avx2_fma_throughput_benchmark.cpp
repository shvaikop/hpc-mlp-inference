#include <immintrin.h>
#include <x86intrin.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iomanip>

#ifndef NOINLINE
#define NOINLINE __attribute__((noinline))
#endif

alignas(32) volatile float sink[8];

static inline uint64_t rdtsc_start() {
    _mm_lfence();
    return __rdtsc();
}

static inline uint64_t rdtsc_stop() {
    unsigned aux;
    uint64_t t = __rdtscp(&aux);
    _mm_lfence();
    return t;
}

static inline void escape(__m256 v) {
    asm volatile("" : : "x"(v) : "memory");
}

NOINLINE void avx2_fma_benchmark(std::uint64_t iters) {
    const __m256 mul = _mm256_set1_ps(0.999999f);
    const __m256 add = _mm256_set1_ps(0.999999f);

    __m256 c0  = _mm256_set1_ps(1.0f);
    __m256 c1  = _mm256_set1_ps(2.0f);
    __m256 c2  = _mm256_set1_ps(3.0f);
    __m256 c3  = _mm256_set1_ps(4.0f);
    __m256 c4  = _mm256_set1_ps(5.0f);
    __m256 c5  = _mm256_set1_ps(6.0f);
    __m256 c6  = _mm256_set1_ps(7.0f);
    __m256 c7  = _mm256_set1_ps(8.0f);
    __m256 c8  = _mm256_set1_ps(9.0f);
    __m256 c9  = _mm256_set1_ps(10.0f);
    __m256 c10 = _mm256_set1_ps(11.0f);
    __m256 c11 = _mm256_set1_ps(12.0f);

    for (std::uint64_t i = 0; i < iters; ++i) {
        c0  = _mm256_fmadd_ps(c0,  mul, add);
        c1  = _mm256_fmadd_ps(c1,  mul, add);
        c2  = _mm256_fmadd_ps(c2,  mul, add);
        c3  = _mm256_fmadd_ps(c3,  mul, add);
        c4  = _mm256_fmadd_ps(c4,  mul, add);
        c5  = _mm256_fmadd_ps(c5,  mul, add);
        c6  = _mm256_fmadd_ps(c6,  mul, add);
        c7  = _mm256_fmadd_ps(c7,  mul, add);
        c8  = _mm256_fmadd_ps(c8,  mul, add);
        c9  = _mm256_fmadd_ps(c9,  mul, add);
        c10 = _mm256_fmadd_ps(c10, mul, add);
        c11 = _mm256_fmadd_ps(c11, mul, add);

        c0  = _mm256_fmadd_ps(c0,  mul, add);
        c1  = _mm256_fmadd_ps(c1,  mul, add);
        c2  = _mm256_fmadd_ps(c2,  mul, add);
        c3  = _mm256_fmadd_ps(c3,  mul, add);
        c4  = _mm256_fmadd_ps(c4,  mul, add);
        c5  = _mm256_fmadd_ps(c5,  mul, add);
        c6  = _mm256_fmadd_ps(c6,  mul, add);
        c7  = _mm256_fmadd_ps(c7,  mul, add);
        c8  = _mm256_fmadd_ps(c8,  mul, add);
        c9  = _mm256_fmadd_ps(c9,  mul, add);
        c10 = _mm256_fmadd_ps(c10, mul, add);
        c11 = _mm256_fmadd_ps(c11, mul, add);
    }

    __m256 sum = c0;
    sum = _mm256_add_ps(sum, c1);
    sum = _mm256_add_ps(sum, c2);
    sum = _mm256_add_ps(sum, c3);
    sum = _mm256_add_ps(sum, c4);
    sum = _mm256_add_ps(sum, c5);
    sum = _mm256_add_ps(sum, c6);
    sum = _mm256_add_ps(sum, c7);
    sum = _mm256_add_ps(sum, c8);
    sum = _mm256_add_ps(sum, c9);
    sum = _mm256_add_ps(sum, c10);
    sum = _mm256_add_ps(sum, c11);

    escape(sum);
}

int main(int argc, char** argv) {
    std::uint64_t iters = 100000000ULL;
    if (argc >= 2) {
        iters = std::strtoull(argv[1], nullptr, 10);
    }

    constexpr std::uint64_t warmup_iters = 1000000ULL;
    constexpr std::uint64_t fma_per_iter = 20;
    constexpr std::uint64_t fp32_flops_per_fma = 16; // 8 lanes * (mul + add)

    avx2_fma_benchmark(warmup_iters);

    uint64_t start = rdtsc_start();
    avx2_fma_benchmark(iters);
    uint64_t end = rdtsc_stop();

    const std::uint64_t cycles = end - start;
    const long double fma_insts =
        static_cast<long double>(iters) * static_cast<long double>(fma_per_iter);
    const long double flops =
        fma_insts * static_cast<long double>(fp32_flops_per_fma);

    const long double fma_per_cycle = fma_insts / static_cast<long double>(cycles);
    const long double cycles_per_fma = static_cast<long double>(cycles) / fma_insts;
    const long double flops_per_cycle = flops / static_cast<long double>(cycles);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Iterations:              " << iters << "\n";
    std::cout << "FMA instructions:         " << fma_insts << "\n";
    std::cout << "FP32 FLOPs:               " << flops << "\n";
    std::cout << "TSC cycles:               " << cycles << "\n";
    std::cout << "FMA instructions / TSC tick: " << fma_per_cycle << "\n";
    std::cout << "Cycles / FMA instruction: " << cycles_per_fma << "\n";
    std::cout << "FP32 FLOPs / TSC tick:       " << flops_per_cycle << "\n\n";

    std::cout << "Interpretation:\n";
    std::cout << "  ~1.0 FMA/cycle  => likely one 256-bit FMA pipeline per core\n";
    std::cout << "  ~2.0 FMA/cycle  => likely two 256-bit FMA pipelines per core\n";
    std::cout << "  ~16 FP32 FLOP/cycle => one 256-bit FMA/cycle\n";
    std::cout << "  ~32 FP32 FLOP/cycle => two 256-bit FMA/cycle\n";

    return 0;
}