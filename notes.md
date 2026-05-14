

### Matrix multiplication dimensions

```aiignore
(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$ ./mlp_infer --export-dir ../cifar10_embedding_mlp/export --input ../cifar10_embedding_mlp/export/test_input_all/test_input_embeddings_all.bin --batch-size 16 --num-batches 1
================================================================================
  Input dim    : 512
  Num classes  : 10
  Batch size   : 16
  Total samples: 16
  Total batches: 1
================================================================================
FlatMatrix dimensions: 16 x 512
FlatMatrix dimensions: 4096 x 512
Multiplying matrix 16 x 512 by matrix 4096 x 512
FlatMatrix dimensions: 16 x 4096
FlatMatrix dimensions: 4096 x 4096
Multiplying matrix 16 x 4096 by matrix 4096 x 4096
FlatMatrix dimensions: 16 x 4096
FlatMatrix dimensions: 4096 x 4096
Multiplying matrix 16 x 4096 by matrix 4096 x 4096
FlatMatrix dimensions: 16 x 4096
FlatMatrix dimensions: 10 x 4096
Multiplying matrix 16 x 4096 by matrix 10 x 4096
Processed batch 1/1 in 580.646 ms
```

### Sparse or Dense?

```aiignore
FlatMatrix dimensions: 16 x 512
--- Matrix Statistics ---
Dimensions      : 16 x 512
Density         : 97.57%
Sparsity        : 2.43%

--- Value Distribution ---
Min               Max       Mean     StdDev
0.0000        16.0689     0.7420     0.9374

--- Range Spread ---
[   0.00,    3.21]: 8041  ###################
[   3.21,    6.43]: 133   
[   6.43,    9.64]: 5     
[   9.64,   12.86]: 8     
[  12.86,   16.07]: 5     
-------------------------
FlatMatrix dimensions: 4096 x 512
--- Matrix Statistics ---
Dimensions      : 4096 x 512
Density         : 100.00%
Sparsity        : 0.00%

--- Value Distribution ---
Min               Max       Mean     StdDev
-0.2119        0.1966    -0.0066     0.0287

--- Range Spread ---
[  -0.21,   -0.13]: 319   
[  -0.13,   -0.05]: 113187 #
[  -0.05,    0.03]: 1835851 #################
[   0.03,    0.11]: 147345 #
[   0.11,    0.20]: 450   
-------------------------
Multiplying matrix 16 x 512 by matrix 512 x 4096
```

```aiignore
FlatMatrix dimensions: 16 x 4096
--- Matrix Statistics ---
Dimensions      : 16 x 4096
Density         : 2.45%
Sparsity        : 97.55%

--- Value Distribution ---
Min               Max       Mean     StdDev
0.0000         4.2964     0.0176     0.1487

--- Range Spread ---
[   0.00,    0.86]: 65014 ###################
[   0.86,    1.72]: 399   
[   1.72,    2.58]: 94    
[   2.58,    3.44]: 26    
[   3.44,    4.30]: 3     
-------------------------
FlatMatrix dimensions: 4096 x 4096
--- Matrix Statistics ---
Dimensions      : 4096 x 4096
Density         : 100.00%
Sparsity        : 0.00%

--- Value Distribution ---
Min               Max       Mean     StdDev
-0.1639        0.2979    -0.0033     0.0124

--- Range Spread ---
[  -0.16,   -0.07]: 10026 
[  -0.07,    0.02]: 16460857 ###################
[   0.02,    0.11]: 305913 
[   0.11,    0.21]: 415   
[   0.21,    0.30]: 5     
-------------------------
Multiplying matrix 16 x 4096 by matrix 4096 x 4096
```

```aiignore
FlatMatrix dimensions: 16 x 4096
--- Matrix Statistics ---
Dimensions      : 16 x 4096
Density         : 8.20%
Sparsity        : 91.80%

--- Value Distribution ---
Min               Max       Mean     StdDev
0.0000         3.5565     0.0392     0.1989

--- Range Spread ---
[   0.00,    0.71]: 64401 ###################
[   0.71,    1.42]: 790   
[   1.42,    2.13]: 224   
[   2.13,    2.85]: 101   
[   2.85,    3.56]: 20    
-------------------------
FlatMatrix dimensions: 4096 x 4096
--- Matrix Statistics ---
Dimensions      : 4096 x 4096
Density         : 100.00%
Sparsity        : 0.00%

--- Value Distribution ---
Min               Max       Mean     StdDev
-0.1739        0.2677    -0.0017     0.0138

--- Range Spread ---
[  -0.17,   -0.09]: 1408  
[  -0.09,    0.00]: 10579262 ############
[   0.00,    0.09]: 6194084 #######
[   0.09,    0.18]: 2448  
[   0.18,    0.27]: 14    
-------------------------
Multiplying matrix 16 x 4096 by matrix 4096 x 4096
```

```aiignore
FlatMatrix dimensions: 16 x 4096
--- Matrix Statistics ---
Dimensions      : 16 x 4096
Density         : 22.98%
Sparsity        : 77.02%

--- Value Distribution ---
Min               Max       Mean     StdDev
0.0000         9.1748     0.1772     0.5535

--- Range Spread ---
[   0.00,    1.83]: 64160 ###################
[   1.83,    3.67]: 1009  
[   3.67,    5.50]: 280   
[   5.50,    7.34]: 75    
[   7.34,    9.17]: 12    
-------------------------
FlatMatrix dimensions: 10 x 4096
--- Matrix Statistics ---
Dimensions      : 10 x 4096
Density         : 100.00%
Sparsity        : 0.00%

--- Value Distribution ---
Min               Max       Mean     StdDev
-0.2589        0.0479    -0.0017     0.0168

--- Range Spread ---
[  -0.26,   -0.20]: 3     
[  -0.20,   -0.14]: 14    
[  -0.14,   -0.07]: 229   
[  -0.07,   -0.01]: 6754  ###
[  -0.01,    0.05]: 33960 ################
-------------------------
Multiplying matrix 16 x 4096 by matrix 4096 x 10
```


### Initial FlatMatrix multiplication results

TODO

### Attempting compiler vectorization

#### AVX2 compiler flag

```aiignore
add_compile_options(-O3)

add_executable(flat_matrix_benchmark
        src/flat_matrix_benchmark.cpp
)

target_compile_options(flat_matrix_benchmark PRIVATE
#        -march=native
        -mavx2
        -mfma
        -ffast-math
        -fopt-info-vec-optimized
        -fopt-info-vec-missed
)
```

```aiignore
/home/pshvaiko/Documents/HPC_PROJECT_MLP_INFERENCE/include/FlatMatrix.hpp:50:9: missed: couldn't vectorize loop
/home/pshvaiko/Documents/HPC_PROJECT_MLP_INFERENCE/include/FlatMatrix.hpp:50:9: missed: not vectorized: multiple nested loops.
/home/pshvaiko/Documents/HPC_PROJECT_MLP_INFERENCE/include/FlatMatrix.hpp:58:35: missed: couldn't vectorize loop
/home/pshvaiko/Documents/HPC_PROJECT_MLP_INFERENCE/include/FlatMatrix.hpp:58:35: missed: not vectorized: multiple nested loops.
/home/pshvaiko/Documents/HPC_PROJECT_MLP_INFERENCE/include/FlatMatrix.hpp:62:39: missed: couldn't vectorize loop
/home/pshvaiko/Documents/HPC_PROJECT_MLP_INFERENCE/include/FlatMatrix.hpp:62:39: missed: not vectorized: control flow in loop.
/home/pshvaiko/Documents/HPC_PROJECT_MLP_INFERENCE/include/FlatMatrix.hpp:66:43: optimized: loop vectorized using 32 byte vectors
/home/pshvaiko/Documents/HPC_PROJECT_MLP_INFERENCE/include/FlatMatrix.hpp:66:43: optimized: loop vectorized using 16 byte vectors
/home/pshvaiko/Documents/HPC_PROJECT_MLP_INFERENCE/include/FlatMatrix.hpp:58:35: missed: couldn't vectorize loop
/home/pshvaiko/Documents/HPC_PROJECT_MLP_INFERENCE/include/FlatMatrix.hpp:58:35: missed: not vectorized: multiple nested loops.
/home/pshvaiko/Documents/HPC_PROJECT_MLP_INFERENCE/include/FlatMatrix.hpp:62:39: missed: couldn't vectorize loop
/home/pshvaiko/Documents/HPC_PROJECT_MLP_INFERENCE/include/FlatMatrix.hpp:62:39: missed: not vectorized: control flow in loop.
/home/pshvaiko/Documents/HPC_PROJECT_MLP_INFERENCE/include/FlatMatrix.hpp:66:43: optimized: loop vectorized using 32 byte vectors
/home/pshvaiko/Documents/HPC_PROJECT_MLP_INFERENCE/include/FlatMatrix.hpp:66:43: optimized: loop vectorized using 16 byte vectors
```

Loop starts on line 48
```aiignore
    void multiply_transposed_rhs(const FlatMatrix& rhs, FlatMatrix& out) const {
        // std::cout << "Multiplying matrix " << rows_ << " x " << cols_ << " by matrix " << rhs.cols() << " x " << rhs.rows() << "\n";
        if (cols_ != rhs.cols_) {
            throw std::runtime_error("Dimension mismatch.");
        }

        if (out.rows() != rows_ || out.cols() != rhs.rows()) {
            throw std::runtime_error("Output matrix has incorrect dimensions.");
        }

        for (std::size_t i = 0; i < rows_; ++i) {
            const T* x = &data_[i * cols_];
            T* y = &out.data_[i * out.cols_];

            for (std::size_t j = 0; j < rhs.rows_; ++j) {
                const T* w = &rhs.data_[j * rhs.cols_];

                T sum = 0.0f;
                for (std::size_t k = 0; k < cols_; ++k) {
                    sum += x[k] * w[k];
                }

                y[j] = sum;
            }
        }
    }
```

```aiignore
(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$ ./flat_matrix_benchmark 
Starting FlatMatrix Benchmarks...
Note: All should have similar runtimes because FlatMatrix is a Dense storage format.
----------------------------------------------------------
[   Dense x Sparse   ] Average Time: 69.319 ms
[   Sparse x Dense   ] Average Time: 70.747 ms
[  Sparse x Sparse   ] Average Time: 58.094 ms
[   Dense x Dense    ] Average Time: 73.441 ms
(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$ ./flat_matrix_benchmark 
Starting FlatMatrix Benchmarks...
Note: All should have similar runtimes because FlatMatrix is a Dense storage format.
----------------------------------------------------------
[   Dense x Sparse   ] Average Time: 70.755 ms
[   Sparse x Dense   ] Average Time: 75.149 ms
[  Sparse x Sparse   ] Average Time: 70.476 ms
[   Dense x Dense    ] Average Time: 75.100 ms
(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$ ./flat_matrix_benchmark 
Starting FlatMatrix Benchmarks...
Note: All should have similar runtimes because FlatMatrix is a Dense storage format.
----------------------------------------------------------
[   Dense x Sparse   ] Average Time: 70.006 ms
[   Sparse x Dense   ] Average Time: 50.789 ms
[  Sparse x Sparse   ] Average Time: 53.811 ms
[   Dense x Dense    ] Average Time: 64.704 ms
(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$
```


```aiignore
(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$ nm ./flat_matrix_benchmark_avx2 | grep "multiply_transposed_rhs"
0000000000007370 W _ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_
(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$ objdump -d --disassemble="_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_" ./flat_matrix_benchmark_avx2 

./flat_matrix_benchmark_avx2:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

0000000000007370 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_>:
    7370:       f3 0f 1e fa             endbr64
    7374:       55                      push   %rbp
    7375:       48 89 f0                mov    %rsi,%rax
    7378:       48 89 e5                mov    %rsp,%rbp
    737b:       41 57                   push   %r15
    737d:       41 56                   push   %r14
    737f:       41 55                   push   %r13
    7381:       41 54                   push   %r12
    7383:       53                      push   %rbx
    7384:       48 83 e4 e0             and    $0xffffffffffffffe0,%rsp
    7388:       48 83 ec 60             sub    $0x60,%rsp
    738c:       48 8b 76 08             mov    0x8(%rsi),%rsi
    7390:       48 8b 5f 08             mov    0x8(%rdi),%rbx
    7394:       48 89 74 24 50          mov    %rsi,0x50(%rsp)
    7399:       48 39 f3                cmp    %rsi,%rbx
    739c:       0f 85 33 02 00 00       jne    75d5 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x265>
    73a2:       48 8b 37                mov    (%rdi),%rsi
    73a5:       48 89 f9                mov    %rdi,%rcx
    73a8:       48 89 74 24 18          mov    %rsi,0x18(%rsp)
    73ad:       48 3b 32                cmp    (%rdx),%rsi
    73b0:       0f 85 51 02 00 00       jne    7607 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x297>
    73b6:       48 8b 38                mov    (%rax),%rdi
    73b9:       4c 8b 7a 08             mov    0x8(%rdx),%r15
    73bd:       48 89 7c 24 10          mov    %rdi,0x10(%rsp)
    73c2:       49 39 ff                cmp    %rdi,%r15
    73c5:       0f 85 3c 02 00 00       jne    7607 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x297>
    73cb:       48 85 f6                test   %rsi,%rsi
    73ce:       0f 84 dc 01 00 00       je     75b0 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x240>
    73d4:       4c 8b 59 10             mov    0x10(%rcx),%r11
    73d8:       4c 8b 62 10             mov    0x10(%rdx),%r12
    73dc:       4d 85 ff                test   %r15,%r15
    73df:       0f 84 cb 01 00 00       je     75b0 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x240>
    73e5:       48 89 de                mov    %rbx,%rsi
    73e8:       4c 8b 50 10             mov    0x10(%rax),%r10
    73ec:       48 8d 43 ff             lea    -0x1(%rbx),%rax
    73f0:       48 89 d9                mov    %rbx,%rcx
    73f3:       83 e6 07                and    $0x7,%esi
    73f6:       48 89 44 24 48          mov    %rax,0x48(%rsp)
    73fb:       48 89 d8                mov    %rbx,%rax
    73fe:       48 c1 e9 03             shr    $0x3,%rcx
    7402:       48 89 74 24 40          mov    %rsi,0x40(%rsp)
    7407:       48 83 e0 f8             and    $0xfffffffffffffff8,%rax
    740b:       48 c1 e1 05             shl    $0x5,%rcx
    740f:       45 31 f6                xor    %r14d,%r14d
    7412:       4c 89 7c 24 08          mov    %r15,0x8(%rsp)
    7417:       31 ff                   xor    %edi,%edi
    7419:       48 89 44 24 38          mov    %rax,0x38(%rsp)
    741e:       31 c0                   xor    %eax,%eax
    7420:       48 8b 74 24 10          mov    0x10(%rsp),%rsi
    7425:       4c 89 64 24 30          mov    %r12,0x30(%rsp)
    742a:       4e 8d 2c b5 00 00 00    lea    0x0(,%r14,4),%r13
    7431:       00 
    7432:       4d 8d 0c 84             lea    (%r12,%rax,4),%r9
    7436:       48 89 7c 24 28          mov    %rdi,0x28(%rsp)
    743b:       4b 8d 14 2b             lea    (%r11,%r13,1),%rdx
    743f:       48 01 c6                add    %rax,%rsi
    7442:       48 89 44 24 20          mov    %rax,0x20(%rsp)
    7447:       49 8d 34 b4             lea    (%r12,%rsi,4),%rsi
    744b:       48 89 74 24 58          mov    %rsi,0x58(%rsp)
    7450:       31 f6                   xor    %esi,%esi
    7452:       66 0f 1f 44 00 00       nopw   0x0(%rax,%rax,1)
    7458:       4c 8d 04 b5 00 00 00    lea    0x0(,%rsi,4),%r8
    745f:       00 
    7460:       48 85 db                test   %rbx,%rbx
    7463:       0f 84 57 01 00 00       je     75c0 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x250>
    7469:       48 83 7c 24 48 06       cmpq   $0x6,0x48(%rsp)
    746f:       0f 86 51 01 00 00       jbe    75c6 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x256>
    7475:       4b 8d 3c 02             lea    (%r10,%r8,1),%rdi
    7479:       31 c0                   xor    %eax,%eax
    747b:       c5 f8 57 c0             vxorps %xmm0,%xmm0,%xmm0
    747f:       90                      nop
    7480:       c5 fc 10 24 07          vmovups (%rdi,%rax,1),%ymm4
    7485:       c5 dc 59 0c 02          vmulps (%rdx,%rax,1),%ymm4,%ymm1
    748a:       48 83 c0 20             add    $0x20,%rax
    748e:       c5 fc 58 c1             vaddps %ymm1,%ymm0,%ymm0
    7492:       48 39 c1                cmp    %rax,%rcx
    7495:       75 e9                   jne    7480 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x110>
    7497:       c4 e3 7d 19 c3 01       vextractf128 $0x1,%ymm0,%xmm3
    749d:       48 83 7c 24 40 00       cmpq   $0x0,0x40(%rsp)
    74a3:       c5 e0 58 c8             vaddps %xmm0,%xmm3,%xmm1
    74a7:       c5 f8 58 c3             vaddps %xmm3,%xmm0,%xmm0
    74ab:       c5 f0 12 d1             vmovhlps %xmm1,%xmm1,%xmm2
    74af:       c5 e8 58 d1             vaddps %xmm1,%xmm2,%xmm2
    74b3:       c5 e8 c6 ca 55          vshufps $0x55,%xmm2,%xmm2,%xmm1
    74b8:       c5 f0 58 ca             vaddps %xmm2,%xmm1,%xmm1
    74bc:       0f 84 a3 00 00 00       je     7565 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x1f5>
    74c2:       48 8b 44 24 38          mov    0x38(%rsp),%rax
    74c7:       48 89 df                mov    %rbx,%rdi
    74ca:       48 29 c7                sub    %rax,%rdi
    74cd:       4c 8d 67 ff             lea    -0x1(%rdi),%r12
    74d1:       49 83 fc 02             cmp    $0x2,%r12
    74d5:       76 34                   jbe    750b <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x19b>
    74d7:       4c 8d 24 30             lea    (%rax,%rsi,1),%r12
    74db:       4e 8d 3c 30             lea    (%rax,%r14,1),%r15
    74df:       c4 81 78 10 2c bb       vmovups (%r11,%r15,4),%xmm5
    74e5:       c4 82 51 b8 04 a2       vfmadd231ps (%r10,%r12,4),%xmm5,%xmm0
    74eb:       49 89 fc                mov    %rdi,%r12
    74ee:       49 83 e4 fc             and    $0xfffffffffffffffc,%r12
    74f2:       4c 01 e0                add    %r12,%rax
    74f5:       83 e7 03                and    $0x3,%edi
    74f8:       c5 f8 12 c8             vmovhlps %xmm0,%xmm0,%xmm1
    74fc:       c5 f0 58 c0             vaddps %xmm0,%xmm1,%xmm0
    7500:       c5 f8 c6 c8 55          vshufps $0x55,%xmm0,%xmm0,%xmm1
    7505:       c5 f0 58 c8             vaddps %xmm0,%xmm1,%xmm1
    7509:       74 5a                   je     7565 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x1f5>
    750b:       48 8d 3c 85 00 00 00    lea    0x0(,%rax,4),%rdi
    7512:       00 
    7513:       4d 8d 24 3a             lea    (%r10,%rdi,1),%r12
    7517:       4d 8d 3c 3b             lea    (%r11,%rdi,1),%r15
    751b:       c4 81 7a 10 34 2f       vmovss (%r15,%r13,1),%xmm6
    7521:       c4 82 49 b9 0c 04       vfmadd231ss (%r12,%r8,1),%xmm6,%xmm1
    7527:       4c 8d 60 01             lea    0x1(%rax),%r12
    752b:       49 39 dc                cmp    %rbx,%r12
    752e:       73 35                   jae    7565 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x1f5>
    7530:       4c 8d 67 04             lea    0x4(%rdi),%r12
    7534:       48 83 c0 02             add    $0x2,%rax
    7538:       4f 8d 3c 22             lea    (%r10,%r12,1),%r15
    753c:       4d 01 dc                add    %r11,%r12
    753f:       c4 81 7a 10 3c 07       vmovss (%r15,%r8,1),%xmm7
    7545:       c4 82 41 b9 0c 2c       vfmadd231ss (%r12,%r13,1),%xmm7,%xmm1
    754b:       48 39 d8                cmp    %rbx,%rax
    754e:       73 15                   jae    7565 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x1f5>
    7550:       49 8d 04 3b             lea    (%r11,%rdi,1),%rax
    7554:       4d 01 d0                add    %r10,%r8
    7557:       c4 a1 7a 10 7c 28 08    vmovss 0x8(%rax,%r13,1),%xmm7
    755e:       c4 a2 41 b9 4c 07 08    vfmadd231ss 0x8(%rdi,%r8,1),%xmm7,%xmm1
    7565:       48 8b 44 24 50          mov    0x50(%rsp),%rax
    756a:       c4 c1 7a 11 09          vmovss %xmm1,(%r9)
    756f:       49 83 c1 04             add    $0x4,%r9
    7573:       48 01 c6                add    %rax,%rsi
    7576:       48 8b 44 24 58          mov    0x58(%rsp),%rax
    757b:       49 39 c1                cmp    %rax,%r9
    757e:       0f 85 d4 fe ff ff       jne    7458 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0xe8>
    7584:       48 8b 7c 24 28          mov    0x28(%rsp),%rdi
    7589:       48 8b 44 24 20          mov    0x20(%rsp),%rax
    758e:       49 01 de                add    %rbx,%r14
    7591:       48 8b 74 24 08          mov    0x8(%rsp),%rsi
    7596:       4c 8b 64 24 30          mov    0x30(%rsp),%r12
    759b:       48 83 c7 01             add    $0x1,%rdi
    759f:       48 01 f0                add    %rsi,%rax
    75a2:       48 39 7c 24 18          cmp    %rdi,0x18(%rsp)
    75a7:       0f 85 73 fe ff ff       jne    7420 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0xb0>
    75ad:       c5 f8 77                vzeroupper
    75b0:       48 8d 65 d8             lea    -0x28(%rbp),%rsp
    75b4:       5b                      pop    %rbx
    75b5:       41 5c                   pop    %r12
    75b7:       41 5d                   pop    %r13
    75b9:       41 5e                   pop    %r14
    75bb:       41 5f                   pop    %r15
    75bd:       5d                      pop    %rbp
    75be:       c3                      ret
    75bf:       90                      nop
    75c0:       c5 f0 57 c9             vxorps %xmm1,%xmm1,%xmm1
    75c4:       eb 9f                   jmp    7565 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x1f5>
    75c6:       c5 f8 57 c0             vxorps %xmm0,%xmm0,%xmm0
    75ca:       31 c0                   xor    %eax,%eax
    75cc:       c5 f0 57 c9             vxorps %xmm1,%xmm1,%xmm1
    75d0:       e9 f2 fe ff ff          jmp    74c7 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x157>
    75d5:       bf 10 00 00 00          mov    $0x10,%edi
    75da:       e8 51 ae ff ff          call   2430 <__cxa_allocate_exception@plt>
    75df:       48 8d 35 b1 fa 00 00    lea    0xfab1(%rip),%rsi        # 17097 <_IO_stdin_used+0x97>
    75e6:       48 89 c7                mov    %rax,%rdi
    75e9:       48 89 c3                mov    %rax,%rbx
    75ec:       e8 1f ae ff ff          call   2410 <_ZNSt13runtime_errorC1EPKc@plt>
    75f1:       48 8b 15 e8 39 01 00    mov    0x139e8(%rip),%rdx        # 1afe0 <_ZNSt13runtime_errorD1Ev@GLIBCXX_3.4>
    75f8:       48 8d 35 21 35 01 00    lea    0x13521(%rip),%rsi        # 1ab20 <_ZTISt13runtime_error@GLIBCXX_3.4>
    75ff:       48 89 df                mov    %rbx,%rdi
    7602:       e8 59 b0 ff ff          call   2660 <__cxa_throw@plt>
    7607:       bf 10 00 00 00          mov    $0x10,%edi
    760c:       e8 1f ae ff ff          call   2430 <__cxa_allocate_exception@plt>
    7611:       48 8d 35 68 fe 00 00    lea    0xfe68(%rip),%rsi        # 17480 <_IO_stdin_used+0x480>
    7618:       48 89 c7                mov    %rax,%rdi
    761b:       48 89 c3                mov    %rax,%rbx
    761e:       e8 ed ad ff ff          call   2410 <_ZNSt13runtime_errorC1EPKc@plt>
    7623:       eb cc                   jmp    75f1 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x281>
    7625:       f3 0f 1e fa             endbr64
    7629:       49 89 c4                mov    %rax,%r12
    762c:       eb 06                   jmp    7634 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x2c4>
    762e:       f3 0f 1e fa             endbr64
    7632:       eb f5                   jmp    7629 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x2b9>
    7634:       48 89 df                mov    %rbx,%rdi
    7637:       c5 f8 77                vzeroupper
    763a:       e8 71 ae ff ff          call   24b0 <__cxa_free_exception@plt>
    763f:       4c 89 e7                mov    %r12,%rdi
    7642:       e8 29 b0 ff ff          call   2670 <_Unwind_Resume@plt>

Disassembly of section .fini:
(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$
```


#### Native compiler flag

```aiignore
add_executable(flat_matrix_benchmark_native
        src/flat_matrix_benchmark.cpp
)

target_compile_options(flat_matrix_benchmark_native PRIVATE
        -march=native
        -mfma
        -ffast-math
        -fopt-info-vec-optimized
        -fopt-info-vec-missed
)
```

```
(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$ ./flat_matrix_benchmark_native 
Starting FlatMatrix Benchmarks...
Note: All should have similar runtimes because FlatMatrix is a Dense storage format.
----------------------------------------------------------
[   Dense x Sparse   ] Average Time: 73.070 ms
[   Sparse x Dense   ] Average Time: 76.155 ms
[  Sparse x Sparse   ] Average Time: 76.272 ms
[   Dense x Dense    ] Average Time: 76.309 ms
(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$ ./flat_matrix_benchmark_native 
Starting FlatMatrix Benchmarks...
Note: All should have similar runtimes because FlatMatrix is a Dense storage format.
----------------------------------------------------------
[   Dense x Sparse   ] Average Time: 73.167 ms
[   Sparse x Dense   ] Average Time: 66.993 ms
[  Sparse x Sparse   ] Average Time: 74.291 ms
[   Dense x Dense    ] Average Time: 74.627 ms
(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$ ./flat_matrix_benchmark_native 
Starting FlatMatrix Benchmarks...
Note: All should have similar runtimes because FlatMatrix is a Dense storage format.
----------------------------------------------------------
[   Dense x Sparse   ] Average Time: 70.327 ms
[   Sparse x Dense   ] Average Time: 54.470 ms
[  Sparse x Sparse   ] Average Time: 51.336 ms
[   Dense x Dense    ] Average Time: 75.393 ms
(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$
```


```aiignore
(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$ nm ./flat_matrix_benchmark_native | grep "multiply_transposed_rhs"
0000000000007380 W _ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_
(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$ objdump -d --disassemble="_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_" ./flat_matrix_benchmark_native

./flat_matrix_benchmark_native:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

0000000000007380 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_>:
    7380:       f3 0f 1e fa             endbr64
    7384:       55                      push   %rbp
    7385:       48 89 f0                mov    %rsi,%rax
    7388:       48 89 e5                mov    %rsp,%rbp
    738b:       41 57                   push   %r15
    738d:       41 56                   push   %r14
    738f:       41 55                   push   %r13
    7391:       41 54                   push   %r12
    7393:       53                      push   %rbx
    7394:       48 83 e4 e0             and    $0xffffffffffffffe0,%rsp
    7398:       48 83 ec 60             sub    $0x60,%rsp
    739c:       48 8b 76 08             mov    0x8(%rsi),%rsi
    73a0:       48 8b 5f 08             mov    0x8(%rdi),%rbx
    73a4:       48 89 74 24 50          mov    %rsi,0x50(%rsp)
    73a9:       48 39 f3                cmp    %rsi,%rbx
    73ac:       0f 85 33 02 00 00       jne    75e5 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x265>
    73b2:       48 8b 37                mov    (%rdi),%rsi
    73b5:       48 89 f9                mov    %rdi,%rcx
    73b8:       48 89 74 24 18          mov    %rsi,0x18(%rsp)
    73bd:       48 3b 32                cmp    (%rdx),%rsi
    73c0:       0f 85 51 02 00 00       jne    7617 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x297>
    73c6:       48 8b 38                mov    (%rax),%rdi
    73c9:       4c 8b 7a 08             mov    0x8(%rdx),%r15
    73cd:       48 89 7c 24 10          mov    %rdi,0x10(%rsp)
    73d2:       49 39 ff                cmp    %rdi,%r15
    73d5:       0f 85 3c 02 00 00       jne    7617 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x297>
    73db:       48 85 f6                test   %rsi,%rsi
    73de:       0f 84 dc 01 00 00       je     75c0 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x240>
    73e4:       4c 8b 59 10             mov    0x10(%rcx),%r11
    73e8:       4c 8b 62 10             mov    0x10(%rdx),%r12
    73ec:       4d 85 ff                test   %r15,%r15
    73ef:       0f 84 cb 01 00 00       je     75c0 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x240>
    73f5:       48 89 de                mov    %rbx,%rsi
    73f8:       4c 8b 50 10             mov    0x10(%rax),%r10
    73fc:       48 8d 43 ff             lea    -0x1(%rbx),%rax
    7400:       48 89 d9                mov    %rbx,%rcx
    7403:       83 e6 07                and    $0x7,%esi
    7406:       48 89 44 24 48          mov    %rax,0x48(%rsp)
    740b:       48 89 d8                mov    %rbx,%rax
    740e:       48 c1 e9 03             shr    $0x3,%rcx
    7412:       48 89 74 24 40          mov    %rsi,0x40(%rsp)
    7417:       48 83 e0 f8             and    $0xfffffffffffffff8,%rax
    741b:       48 c1 e1 05             shl    $0x5,%rcx
    741f:       45 31 f6                xor    %r14d,%r14d
    7422:       4c 89 7c 24 08          mov    %r15,0x8(%rsp)
    7427:       31 ff                   xor    %edi,%edi
    7429:       48 89 44 24 38          mov    %rax,0x38(%rsp)
    742e:       31 c0                   xor    %eax,%eax
    7430:       48 8b 74 24 10          mov    0x10(%rsp),%rsi
    7435:       4c 89 64 24 30          mov    %r12,0x30(%rsp)
    743a:       4e 8d 2c b5 00 00 00    lea    0x0(,%r14,4),%r13
    7441:       00 
    7442:       4d 8d 0c 84             lea    (%r12,%rax,4),%r9
    7446:       48 89 7c 24 28          mov    %rdi,0x28(%rsp)
    744b:       4b 8d 14 2b             lea    (%r11,%r13,1),%rdx
    744f:       48 01 c6                add    %rax,%rsi
    7452:       48 89 44 24 20          mov    %rax,0x20(%rsp)
    7457:       49 8d 34 b4             lea    (%r12,%rsi,4),%rsi
    745b:       48 89 74 24 58          mov    %rsi,0x58(%rsp)
    7460:       31 f6                   xor    %esi,%esi
    7462:       66 0f 1f 44 00 00       nopw   0x0(%rax,%rax,1)
    7468:       4c 8d 04 b5 00 00 00    lea    0x0(,%rsi,4),%r8
    746f:       00 
    7470:       48 85 db                test   %rbx,%rbx
    7473:       0f 84 57 01 00 00       je     75d0 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x250>
    7479:       48 83 7c 24 48 06       cmpq   $0x6,0x48(%rsp)
    747f:       0f 86 51 01 00 00       jbe    75d6 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x256>
    7485:       4b 8d 3c 02             lea    (%r10,%r8,1),%rdi
    7489:       31 c0                   xor    %eax,%eax
    748b:       c5 f8 57 c0             vxorps %xmm0,%xmm0,%xmm0
    748f:       90                      nop
    7490:       c5 fc 10 24 07          vmovups (%rdi,%rax,1),%ymm4
    7495:       c5 dc 59 0c 02          vmulps (%rdx,%rax,1),%ymm4,%ymm1
    749a:       48 83 c0 20             add    $0x20,%rax
    749e:       c5 fc 58 c1             vaddps %ymm1,%ymm0,%ymm0
    74a2:       48 39 c1                cmp    %rax,%rcx
    74a5:       75 e9                   jne    7490 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x110>
    74a7:       c4 e3 7d 19 c3 01       vextractf128 $0x1,%ymm0,%xmm3
    74ad:       48 83 7c 24 40 00       cmpq   $0x0,0x40(%rsp)
    74b3:       c5 e0 58 c8             vaddps %xmm0,%xmm3,%xmm1
    74b7:       c5 f8 58 c3             vaddps %xmm3,%xmm0,%xmm0
    74bb:       c5 f0 12 d1             vmovhlps %xmm1,%xmm1,%xmm2
    74bf:       c5 e8 58 d1             vaddps %xmm1,%xmm2,%xmm2
    74c3:       c5 e8 c6 ca 55          vshufps $0x55,%xmm2,%xmm2,%xmm1
    74c8:       c5 f0 58 ca             vaddps %xmm2,%xmm1,%xmm1
    74cc:       0f 84 a3 00 00 00       je     7575 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x1f5>
    74d2:       48 8b 44 24 38          mov    0x38(%rsp),%rax
    74d7:       48 89 df                mov    %rbx,%rdi
    74da:       48 29 c7                sub    %rax,%rdi
    74dd:       4c 8d 67 ff             lea    -0x1(%rdi),%r12
    74e1:       49 83 fc 02             cmp    $0x2,%r12
    74e5:       76 34                   jbe    751b <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x19b>
    74e7:       4c 8d 24 30             lea    (%rax,%rsi,1),%r12
    74eb:       4e 8d 3c 30             lea    (%rax,%r14,1),%r15
    74ef:       c4 81 78 10 2c bb       vmovups (%r11,%r15,4),%xmm5
    74f5:       c4 82 51 b8 04 a2       vfmadd231ps (%r10,%r12,4),%xmm5,%xmm0
    74fb:       49 89 fc                mov    %rdi,%r12
    74fe:       49 83 e4 fc             and    $0xfffffffffffffffc,%r12
    7502:       4c 01 e0                add    %r12,%rax
    7505:       83 e7 03                and    $0x3,%edi
    7508:       c5 f8 12 c8             vmovhlps %xmm0,%xmm0,%xmm1
    750c:       c5 f0 58 c0             vaddps %xmm0,%xmm1,%xmm0
    7510:       c5 f8 c6 c8 55          vshufps $0x55,%xmm0,%xmm0,%xmm1
    7515:       c5 f0 58 c8             vaddps %xmm0,%xmm1,%xmm1
    7519:       74 5a                   je     7575 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x1f5>
    751b:       48 8d 3c 85 00 00 00    lea    0x0(,%rax,4),%rdi
    7522:       00 
    7523:       4d 8d 24 3a             lea    (%r10,%rdi,1),%r12
    7527:       4d 8d 3c 3b             lea    (%r11,%rdi,1),%r15
    752b:       c4 81 7a 10 34 2f       vmovss (%r15,%r13,1),%xmm6
    7531:       c4 82 49 b9 0c 04       vfmadd231ss (%r12,%r8,1),%xmm6,%xmm1
    7537:       4c 8d 60 01             lea    0x1(%rax),%r12
    753b:       49 39 dc                cmp    %rbx,%r12
    753e:       73 35                   jae    7575 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x1f5>
    7540:       4c 8d 67 04             lea    0x4(%rdi),%r12
    7544:       48 83 c0 02             add    $0x2,%rax
    7548:       4f 8d 3c 22             lea    (%r10,%r12,1),%r15
    754c:       4d 01 dc                add    %r11,%r12
    754f:       c4 81 7a 10 3c 07       vmovss (%r15,%r8,1),%xmm7
    7555:       c4 82 41 b9 0c 2c       vfmadd231ss (%r12,%r13,1),%xmm7,%xmm1
    755b:       48 39 d8                cmp    %rbx,%rax
    755e:       73 15                   jae    7575 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x1f5>
    7560:       49 8d 04 3b             lea    (%r11,%rdi,1),%rax
    7564:       4d 01 d0                add    %r10,%r8
    7567:       c4 a1 7a 10 7c 28 08    vmovss 0x8(%rax,%r13,1),%xmm7
    756e:       c4 a2 41 b9 4c 07 08    vfmadd231ss 0x8(%rdi,%r8,1),%xmm7,%xmm1
    7575:       48 8b 44 24 50          mov    0x50(%rsp),%rax
    757a:       c4 c1 7a 11 09          vmovss %xmm1,(%r9)
    757f:       49 83 c1 04             add    $0x4,%r9
    7583:       48 01 c6                add    %rax,%rsi
    7586:       48 8b 44 24 58          mov    0x58(%rsp),%rax
    758b:       49 39 c1                cmp    %rax,%r9
    758e:       0f 85 d4 fe ff ff       jne    7468 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0xe8>
    7594:       48 8b 7c 24 28          mov    0x28(%rsp),%rdi
    7599:       48 8b 44 24 20          mov    0x20(%rsp),%rax
    759e:       49 01 de                add    %rbx,%r14
    75a1:       48 8b 74 24 08          mov    0x8(%rsp),%rsi
    75a6:       4c 8b 64 24 30          mov    0x30(%rsp),%r12
    75ab:       48 83 c7 01             add    $0x1,%rdi
    75af:       48 01 f0                add    %rsi,%rax
    75b2:       48 39 7c 24 18          cmp    %rdi,0x18(%rsp)
    75b7:       0f 85 73 fe ff ff       jne    7430 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0xb0>
    75bd:       c5 f8 77                vzeroupper
    75c0:       48 8d 65 d8             lea    -0x28(%rbp),%rsp
    75c4:       5b                      pop    %rbx
    75c5:       41 5c                   pop    %r12
    75c7:       41 5d                   pop    %r13
    75c9:       41 5e                   pop    %r14
    75cb:       41 5f                   pop    %r15
    75cd:       5d                      pop    %rbp
    75ce:       c3                      ret
    75cf:       90                      nop
    75d0:       c5 f0 57 c9             vxorps %xmm1,%xmm1,%xmm1
    75d4:       eb 9f                   jmp    7575 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x1f5>
    75d6:       c5 f8 57 c0             vxorps %xmm0,%xmm0,%xmm0
    75da:       31 c0                   xor    %eax,%eax
    75dc:       c5 f0 57 c9             vxorps %xmm1,%xmm1,%xmm1
    75e0:       e9 f2 fe ff ff          jmp    74d7 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x157>
    75e5:       bf 10 00 00 00          mov    $0x10,%edi
    75ea:       e8 41 ae ff ff          call   2430 <__cxa_allocate_exception@plt>
    75ef:       48 8d 35 a1 fa 00 00    lea    0xfaa1(%rip),%rsi        # 17097 <_IO_stdin_used+0x97>
    75f6:       48 89 c7                mov    %rax,%rdi
    75f9:       48 89 c3                mov    %rax,%rbx
    75fc:       e8 0f ae ff ff          call   2410 <_ZNSt13runtime_errorC1EPKc@plt>
    7601:       48 8b 15 d8 39 01 00    mov    0x139d8(%rip),%rdx        # 1afe0 <_ZNSt13runtime_errorD1Ev@GLIBCXX_3.4>
    7608:       48 8d 35 11 35 01 00    lea    0x13511(%rip),%rsi        # 1ab20 <_ZTISt13runtime_error@GLIBCXX_3.4>
    760f:       48 89 df                mov    %rbx,%rdi
    7612:       e8 49 b0 ff ff          call   2660 <__cxa_throw@plt>
    7617:       bf 10 00 00 00          mov    $0x10,%edi
    761c:       e8 0f ae ff ff          call   2430 <__cxa_allocate_exception@plt>
    7621:       48 8d 35 58 fe 00 00    lea    0xfe58(%rip),%rsi        # 17480 <_IO_stdin_used+0x480>
    7628:       48 89 c7                mov    %rax,%rdi
    762b:       48 89 c3                mov    %rax,%rbx
    762e:       e8 dd ad ff ff          call   2410 <_ZNSt13runtime_errorC1EPKc@plt>
    7633:       eb cc                   jmp    7601 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x281>
    7635:       f3 0f 1e fa             endbr64
    7639:       49 89 c4                mov    %rax,%r12
    763c:       eb 06                   jmp    7644 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x2c4>
    763e:       f3 0f 1e fa             endbr64
    7642:       eb f5                   jmp    7639 <_ZNK10FlatMatrixIfE23multiply_transposed_rhsERKS0_RS0_+0x2b9>
    7644:       48 89 df                mov    %rbx,%rdi
    7647:       c5 f8 77                vzeroupper
    764a:       e8 61 ae ff ff          call   24b0 <__cxa_free_exception@plt>
    764f:       4c 89 e7                mov    %r12,%rdi
    7652:       e8 19 b0 ff ff          call   2670 <_Unwind_Resume@plt>

Disassembly of section .fini:
(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$
```


### Structure the code for better vectorization by compiler (without openmp simd)

```aiignore
(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$ nm ./flat_matrix_benchmark_dot4 | grep dot4
0000000000007810 W _ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm
(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$ objdump -d --disassemble="_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm" ./flat_matrix_benchmark_dot4

./flat_matrix_benchmark_dot4:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

0000000000007810 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm>:
    7810:       f3 0f 1e fa             endbr64
    7814:       55                      push   %rbp
    7815:       48 89 e5                mov    %rsp,%rbp
    7818:       41 57                   push   %r15
    781a:       41 56                   push   %r14
    781c:       41 55                   push   %r13
    781e:       41 54                   push   %r12
    7820:       53                      push   %rbx
    7821:       48 83 e4 e0             and    $0xffffffffffffffe0,%rsp
    7825:       48 83 ec 48             sub    $0x48,%rsp
    7829:       48 89 7c 24 10          mov    %rdi,0x10(%rsp)
    782e:       48 89 4c 24 b8          mov    %rcx,-0x48(%rsp)
    7833:       48 85 c9                test   %rcx,%rcx
    7836:       0f 84 83 05 00 00       je     7dbf <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x5af>
    783c:       4c 89 cf                mov    %r9,%rdi
    783f:       4c 89 c9                mov    %r9,%rcx
    7842:       48 89 d0                mov    %rdx,%rax
    7845:       49 89 f2                mov    %rsi,%r10
    7848:       48 c1 e7 04             shl    $0x4,%rdi
    784c:       49 8d 50 fc             lea    -0x4(%r8),%rdx
    7850:       48 c1 e9 03             shr    $0x3,%rcx
    7854:       4c 89 cb                mov    %r9,%rbx
    7857:       48 89 7c 24 e8          mov    %rdi,-0x18(%rsp)
    785c:       4a 8d 3c ce             lea    (%rsi,%r9,8),%rdi
    7860:       48 c1 e1 05             shl    $0x5,%rcx
    7864:       48 89 d6                mov    %rdx,%rsi
    7867:       48 89 7c 24 a0          mov    %rdi,-0x60(%rsp)
    786c:       48 83 e2 fc             and    $0xfffffffffffffffc,%rdx
    7870:       48 c1 ee 02             shr    $0x2,%rsi
    7874:       49 89 ce                mov    %rcx,%r14
    7877:       48 83 c2 04             add    $0x4,%rdx
    787b:       31 c9                   xor    %ecx,%ecx
    787d:       4c 89 44 24 d0          mov    %r8,-0x30(%rsp)
    7882:       49 89 db                mov    %rbx,%r11
    7885:       4a 8d 3c 8d 00 00 00    lea    0x0(,%r9,4),%rdi
    788c:       00 
    788d:       4e 8d 24 85 00 00 00    lea    0x0(,%r8,4),%r12
    7894:       00 
    7895:       48 c7 44 24 18 00 00    movq   $0x0,0x18(%rsp)
    789c:       00 00 
    789e:       48 89 7c 24 20          mov    %rdi,0x20(%rsp)
    78a3:       4b 8d 3c 09             lea    (%r9,%r9,1),%rdi
    78a7:       48 89 7c 24 a8          mov    %rdi,-0x58(%rsp)
    78ac:       49 8d 79 ff             lea    -0x1(%r9),%rdi
    78b0:       48 89 3c 24             mov    %rdi,(%rsp)
    78b4:       4c 89 cf                mov    %r9,%rdi
    78b7:       48 83 e7 f8             and    $0xfffffffffffffff8,%rdi
    78bb:       49 83 f8 03             cmp    $0x3,%r8
    78bf:       48 0f 47 ca             cmova  %rdx,%rcx
    78c3:       48 89 7c 24 e0          mov    %rdi,-0x20(%rsp)
    78c8:       48 8d 56 01             lea    0x1(%rsi),%rdx
    78cc:       48 c1 e2 04             shl    $0x4,%rdx
    78d0:       48 8d 3c 8d 00 00 00    lea    0x0(,%rcx,4),%rdi
    78d7:       00 
    78d8:       48 89 54 24 98          mov    %rdx,-0x68(%rsp)
    78dd:       48 89 7c 24 90          mov    %rdi,-0x70(%rsp)
    78e2:       4c 89 cf                mov    %r9,%rdi
    78e5:       45 31 c9                xor    %r9d,%r9d
    78e8:       48 0f af f9             imul   %rcx,%rdi
    78ec:       4c 89 4c 24 d8          mov    %r9,-0x28(%rsp)
    78f1:       48 89 4c 24 b0          mov    %rcx,-0x50(%rsp)
    78f6:       48 89 7c 24 88          mov    %rdi,-0x78(%rsp)
    78fb:       48 89 df                mov    %rbx,%rdi
    78fe:       83 e7 07                and    $0x7,%edi
    7901:       48 89 7c 24 f0          mov    %rdi,-0x10(%rsp)
    7906:       66 2e 0f 1f 84 00 00    cs nopw 0x0(%rax,%rax,1)
    790d:       00 00 00 
    7910:       48 8b 7c 24 10          mov    0x10(%rsp),%rdi
    7915:       48 8b 74 24 18          mov    0x18(%rsp),%rsi
    791a:       48 83 7c 24 d0 03       cmpq   $0x3,-0x30(%rsp)
    7920:       48 8d 14 b7             lea    (%rdi,%rsi,4),%rdx
    7924:       0f 86 3f 03 00 00       jbe    7c69 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x459>
    792a:       48 8b 7c 24 98          mov    -0x68(%rsp),%rdi
    792f:       48 89 44 24 30          mov    %rax,0x30(%rsp)
    7934:       4c 89 d6                mov    %r10,%rsi
    7937:       4c 8b 44 24 a8          mov    -0x58(%rsp),%r8
    793c:       48 8b 4c 24 a0          mov    -0x60(%rsp),%rcx
    7941:       4c 89 74 24 28          mov    %r14,0x28(%rsp)
    7946:       48 89 44 24 c8          mov    %rax,-0x38(%rsp)
    794b:       48 01 c7                add    %rax,%rdi
    794e:       4c 89 64 24 c0          mov    %r12,-0x40(%rsp)
    7953:       48 89 7c 24 f8          mov    %rdi,-0x8(%rsp)
    7958:       31 ff                   xor    %edi,%edi
    795a:       66 0f 1f 44 00 00       nopw   0x0(%rax,%rax,1)
    7960:       49 8d 04 3b             lea    (%r11,%rdi,1),%rax
    7964:       48 89 44 24 40          mov    %rax,0x40(%rsp)
    7969:       4b 8d 04 03             lea    (%r11,%r8,1),%rax
    796d:       48 89 44 24 38          mov    %rax,0x38(%rsp)
    7972:       4d 85 db                test   %r11,%r11
    7975:       0f 84 55 04 00 00       je     7dd0 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x5c0>
    797b:       48 83 3c 24 06          cmpq   $0x6,(%rsp)
    7980:       0f 86 53 04 00 00       jbe    7dd9 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x5c9>
    7986:       48 8b 44 24 20          mov    0x20(%rsp),%rax
    798b:       c5 e8 57 d2             vxorps %xmm2,%xmm2,%xmm2
    798f:       4c 8b 74 24 28          mov    0x28(%rsp),%r14
    7994:       c5 fc 28 ca             vmovaps %ymm2,%ymm1
    7998:       c5 fc 28 da             vmovaps %ymm2,%ymm3
    799c:       c5 fc 28 e2             vmovaps %ymm2,%ymm4
    79a0:       48 8d 1c 06             lea    (%rsi,%rax,1),%rbx
    79a4:       4c 8d 0c 01             lea    (%rcx,%rax,1),%r9
    79a8:       31 c0                   xor    %eax,%eax
    79aa:       66 0f 1f 44 00 00       nopw   0x0(%rax,%rax,1)
    79b0:       c5 fc 10 04 02          vmovups (%rdx,%rax,1),%ymm0
    79b5:       c4 e2 7d b8 24 06       vfmadd231ps (%rsi,%rax,1),%ymm0,%ymm4
    79bb:       c4 e2 7d b8 1c 03       vfmadd231ps (%rbx,%rax,1),%ymm0,%ymm3
    79c1:       c4 e2 7d b8 0c 01       vfmadd231ps (%rcx,%rax,1),%ymm0,%ymm1
    79c7:       c4 c2 7d b8 14 01       vfmadd231ps (%r9,%rax,1),%ymm0,%ymm2
    79cd:       48 83 c0 20             add    $0x20,%rax
    79d1:       49 39 c6                cmp    %rax,%r14
    79d4:       75 da                   jne    79b0 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x1a0>
    79d6:       c4 e3 7d 19 d7 01       vextractf128 $0x1,%ymm2,%xmm7
    79dc:       c4 c3 7d 19 c8 01       vextractf128 $0x1,%ymm1,%xmm8
    79e2:       c4 c3 7d 19 d9 01       vextractf128 $0x1,%ymm3,%xmm9
    79e8:       48 83 7c 24 f0 00       cmpq   $0x0,-0x10(%rsp)
    79ee:       c5 c0 58 ea             vaddps %xmm2,%xmm7,%xmm5
    79f2:       c5 b0 58 f3             vaddps %xmm3,%xmm9,%xmm6
    79f6:       c4 c3 7d 19 e3 01       vextractf128 $0x1,%ymm4,%xmm11
    79fc:       4c 89 74 24 28          mov    %r14,0x28(%rsp)
    7a01:       c4 c1 60 58 d9          vaddps %xmm9,%xmm3,%xmm3
    7a06:       c5 e8 58 d7             vaddps %xmm7,%xmm2,%xmm2
    7a0a:       c5 d0 12 c5             vmovhlps %xmm5,%xmm5,%xmm0
    7a0e:       c5 f8 58 c5             vaddps %xmm5,%xmm0,%xmm0
    7a12:       c5 b8 58 e9             vaddps %xmm1,%xmm8,%xmm5
    7a16:       c4 c1 70 58 c8          vaddps %xmm8,%xmm1,%xmm1
    7a1b:       c5 78 c6 e0 55          vshufps $0x55,%xmm0,%xmm0,%xmm12
    7a20:       c5 18 58 e0             vaddps %xmm0,%xmm12,%xmm12
    7a24:       c5 d0 12 c5             vmovhlps %xmm5,%xmm5,%xmm0
    7a28:       c5 f8 58 c5             vaddps %xmm5,%xmm0,%xmm0
    7a2c:       c5 f8 c6 e8 55          vshufps $0x55,%xmm0,%xmm0,%xmm5
    7a31:       c5 d0 58 e8             vaddps %xmm0,%xmm5,%xmm5
    7a35:       c5 c8 12 c6             vmovhlps %xmm6,%xmm6,%xmm0
    7a39:       c5 f8 58 c6             vaddps %xmm6,%xmm0,%xmm0
    7a3d:       c5 a0 58 f4             vaddps %xmm4,%xmm11,%xmm6
    7a41:       c4 c1 58 58 e3          vaddps %xmm11,%xmm4,%xmm4
    7a46:       c4 c1 50 14 ec          vunpcklps %xmm12,%xmm5,%xmm5
    7a4b:       c5 78 c6 d0 55          vshufps $0x55,%xmm0,%xmm0,%xmm10
    7a50:       c5 28 58 d0             vaddps %xmm0,%xmm10,%xmm10
    7a54:       c5 c8 12 c6             vmovhlps %xmm6,%xmm6,%xmm0
    7a58:       c5 f8 58 c6             vaddps %xmm6,%xmm0,%xmm0
    7a5c:       c5 f8 c6 f0 55          vshufps $0x55,%xmm0,%xmm0,%xmm6
    7a61:       c5 c8 58 f0             vaddps %xmm0,%xmm6,%xmm6
    7a65:       c4 c1 48 14 c2          vunpcklps %xmm10,%xmm6,%xmm0
    7a6a:       c5 f8 16 c5             vmovlhps %xmm5,%xmm0,%xmm0
    7a6e:       0f 84 b0 01 00 00       je     7c24 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x414>
    7a74:       48 8b 44 24 e0          mov    -0x20(%rsp),%rax
    7a79:       4d 89 d9                mov    %r11,%r9
    7a7c:       49 29 c1                sub    %rax,%r9
    7a7f:       49 8d 59 ff             lea    -0x1(%r9),%rbx
    7a83:       48 83 fb 02             cmp    $0x2,%rbx
    7a87:       0f 86 b4 00 00 00       jbe    7b41 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x331>
    7a8d:       48 8b 5c 24 40          mov    0x40(%rsp),%rbx
    7a92:       4c 8b 7c 24 18          mov    0x18(%rsp),%r15
    7a97:       4d 8d 24 00             lea    (%r8,%rax,1),%r12
    7a9b:       4c 8b 74 24 10          mov    0x10(%rsp),%r14
    7aa0:       4c 8d 2c 03             lea    (%rbx,%rax,1),%r13
    7aa4:       48 8b 5c 24 38          mov    0x38(%rsp),%rbx
    7aa9:       49 01 c7                add    %rax,%r15
    7aac:       c4 81 78 10 2c be       vmovups (%r14,%r15,4),%xmm5
    7ab2:       4c 8d 3c 38             lea    (%rax,%rdi,1),%r15
    7ab6:       c4 82 51 b8 1c aa       vfmadd231ps (%r10,%r13,4),%xmm5,%xmm3
    7abc:       48 01 c3                add    %rax,%rbx
    7abf:       c4 82 51 b8 24 ba       vfmadd231ps (%r10,%r15,4),%xmm5,%xmm4
    7ac5:       c4 82 51 b8 0c a2       vfmadd231ps (%r10,%r12,4),%xmm5,%xmm1
    7acb:       c4 c2 69 98 2c 9a       vfmadd132ps (%r10,%rbx,4),%xmm2,%xmm5
    7ad1:       4c 89 cb                mov    %r9,%rbx
    7ad4:       48 83 e3 fc             and    $0xfffffffffffffffc,%rbx
    7ad8:       48 01 d8                add    %rbx,%rax
    7adb:       41 83 e1 03             and    $0x3,%r9d
    7adf:       c5 e0 12 d3             vmovhlps %xmm3,%xmm3,%xmm2
    7ae3:       c5 e8 58 d3             vaddps %xmm3,%xmm2,%xmm2
    7ae7:       c5 f8 28 c1             vmovaps %xmm1,%xmm0
    7aeb:       c5 d0 12 cd             vmovhlps %xmm5,%xmm5,%xmm1
    7aef:       c5 f0 58 cd             vaddps %xmm5,%xmm1,%xmm1
    7af3:       c5 f0 c6 e9 55          vshufps $0x55,%xmm1,%xmm1,%xmm5
    7af8:       c5 d0 58 e9             vaddps %xmm1,%xmm5,%xmm5
    7afc:       c5 f8 12 c8             vmovhlps %xmm0,%xmm0,%xmm1
    7b00:       c5 f0 58 c8             vaddps %xmm0,%xmm1,%xmm1
    7b04:       c5 f0 c6 c1 55          vshufps $0x55,%xmm1,%xmm1,%xmm0
    7b09:       c5 f8 58 c1             vaddps %xmm1,%xmm0,%xmm0
    7b0d:       c5 f8 28 c8             vmovaps %xmm0,%xmm1
    7b11:       c5 e8 c6 c2 55          vshufps $0x55,%xmm2,%xmm2,%xmm0
    7b16:       c5 f8 58 c2             vaddps %xmm2,%xmm0,%xmm0
    7b1a:       c5 f0 14 cd             vunpcklps %xmm5,%xmm1,%xmm1
    7b1e:       c5 f8 28 d8             vmovaps %xmm0,%xmm3
    7b22:       c5 d8 12 c4             vmovhlps %xmm4,%xmm4,%xmm0
    7b26:       c5 f8 58 c4             vaddps %xmm4,%xmm0,%xmm0
    7b2a:       c5 f8 c6 d0 55          vshufps $0x55,%xmm0,%xmm0,%xmm2
    7b2f:       c5 e8 58 d0             vaddps %xmm0,%xmm2,%xmm2
    7b33:       c5 e8 14 c3             vunpcklps %xmm3,%xmm2,%xmm0
    7b37:       c5 f8 16 c1             vmovlhps %xmm1,%xmm0,%xmm0
    7b3b:       0f 84 e3 00 00 00       je     7c24 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x414>
    7b41:       48 8b 5c 24 38          mov    0x38(%rsp),%rbx
    7b46:       4c 8b 74 24 40          mov    0x40(%rsp),%r14
    7b4b:       4c 8d 3c 85 00 00 00    lea    0x0(,%rax,4),%r15
    7b52:       00 
    7b53:       4c 8d 24 07             lea    (%rdi,%rax,1),%r12
    7b57:       4c 89 7c 24 08          mov    %r15,0x8(%rsp)
    7b5c:       4d 8d 3c 00             lea    (%r8,%rax,1),%r15
    7b60:       c4 81 7a 10 0c a2       vmovss (%r10,%r12,4),%xmm1
    7b66:       49 89 dd                mov    %rbx,%r13
    7b69:       4d 8d 0c 06             lea    (%r14,%rax,1),%r9
    7b6d:       c4 81 7a 10 14 ba       vmovss (%r10,%r15,4),%xmm2
    7b73:       49 01 c5                add    %rax,%r13
    7b76:       c4 83 71 21 0c 8a 10    vinsertps $0x10,(%r10,%r9,4),%xmm1,%xmm1
    7b7d:       4c 8d 48 01             lea    0x1(%rax),%r9
    7b81:       c4 83 69 21 14 aa 10    vinsertps $0x10,(%r10,%r13,4),%xmm2,%xmm2
    7b88:       c5 f0 16 ca             vmovlhps %xmm2,%xmm1,%xmm1
    7b8c:       c4 e2 79 18 14 82       vbroadcastss (%rdx,%rax,4),%xmm2
    7b92:       c4 e2 71 b8 c2          vfmadd231ps %xmm2,%xmm1,%xmm0
    7b97:       4d 39 d9                cmp    %r11,%r9
    7b9a:       0f 83 84 00 00 00       jae    7c24 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x414>
    7ba0:       4c 8b 7c 24 08          mov    0x8(%rsp),%r15
    7ba5:       4d 8d 2c 39             lea    (%r9,%rdi,1),%r13
    7ba9:       4f 8d 24 0e             lea    (%r14,%r9,1),%r12
    7bad:       48 83 c0 02             add    $0x2,%rax
    7bb1:       c4 81 7a 10 0c aa       vmovss (%r10,%r13,4),%xmm1
    7bb7:       c4 83 71 21 0c a2 10    vinsertps $0x10,(%r10,%r12,4),%xmm1,%xmm1
    7bbe:       c4 a2 79 18 5c 3a 04    vbroadcastss 0x4(%rdx,%r15,1),%xmm3
    7bc5:       4f 8d 3c 01             lea    (%r9,%r8,1),%r15
    7bc9:       49 01 d9                add    %rbx,%r9
    7bcc:       c4 81 7a 10 14 ba       vmovss (%r10,%r15,4),%xmm2
    7bd2:       c4 83 69 21 14 8a 10    vinsertps $0x10,(%r10,%r9,4),%xmm2,%xmm2
    7bd9:       c5 f0 16 ca             vmovlhps %xmm2,%xmm1,%xmm1
    7bdd:       c4 e2 61 b8 c1          vfmadd231ps %xmm1,%xmm3,%xmm0
    7be2:       4c 39 d8                cmp    %r11,%rax
    7be5:       73 3d                   jae    7c24 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x414>
    7be7:       4c 8d 24 38             lea    (%rax,%rdi,1),%r12
    7beb:       49 01 c6                add    %rax,%r14
    7bee:       4d 8d 2c 00             lea    (%r8,%rax,1),%r13
    7bf2:       48 01 c3                add    %rax,%rbx
    7bf5:       c4 81 7a 10 14 aa       vmovss (%r10,%r13,4),%xmm2
    7bfb:       c4 81 7a 10 0c a2       vmovss (%r10,%r12,4),%xmm1
    7c01:       c4 c3 69 21 14 9a 10    vinsertps $0x10,(%r10,%rbx,4),%xmm2,%xmm2
    7c08:       c4 83 71 21 0c b2 10    vinsertps $0x10,(%r10,%r14,4),%xmm1,%xmm1
    7c0f:       48 8b 44 24 08          mov    0x8(%rsp),%rax
    7c14:       c5 f0 16 ca             vmovlhps %xmm2,%xmm1,%xmm1
    7c18:       c4 e2 79 18 54 02 08    vbroadcastss 0x8(%rdx,%rax,1),%xmm2
    7c1f:       c4 e2 71 b8 c2          vfmadd231ps %xmm2,%xmm1,%xmm0
    7c24:       48 8b 5c 24 e8          mov    -0x18(%rsp),%rbx
    7c29:       48 8b 44 24 30          mov    0x30(%rsp),%rax
    7c2e:       48 01 de                add    %rbx,%rsi
    7c31:       48 01 d9                add    %rbx,%rcx
    7c34:       48 8b 5c 24 20          mov    0x20(%rsp),%rbx
    7c39:       c5 f8 11 00             vmovups %xmm0,(%rax)
    7c3d:       48 83 c0 10             add    $0x10,%rax
    7c41:       48 01 df                add    %rbx,%rdi
    7c44:       49 01 d8                add    %rbx,%r8
    7c47:       48 8b 5c 24 f8          mov    -0x8(%rsp),%rbx
    7c4c:       48 89 44 24 30          mov    %rax,0x30(%rsp)
    7c51:       48 39 d8                cmp    %rbx,%rax
    7c54:       0f 85 06 fd ff ff       jne    7960 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x150>
    7c5a:       4c 8b 74 24 28          mov    0x28(%rsp),%r14
    7c5f:       48 8b 44 24 c8          mov    -0x38(%rsp),%rax
    7c64:       4c 8b 64 24 c0          mov    -0x40(%rsp),%r12
    7c69:       48 8b 74 24 d0          mov    -0x30(%rsp),%rsi
    7c6e:       4e 8d 2c 20             lea    (%rax,%r12,1),%r13
    7c72:       48 39 74 24 b0          cmp    %rsi,-0x50(%rsp)
    7c77:       0f 83 21 01 00 00       jae    7d9e <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x58e>
    7c7d:       48 8b 7c 24 90          mov    -0x70(%rsp),%rdi
    7c82:       48 8b 74 24 88          mov    -0x78(%rsp),%rsi
    7c87:       4c 8b 4c 24 18          mov    0x18(%rsp),%r9
    7c8c:       48 8b 5c 24 10          mov    0x10(%rsp),%rbx
    7c91:       48 01 c7                add    %rax,%rdi
    7c94:       4d 85 db                test   %r11,%r11
    7c97:       0f 84 57 01 00 00       je     7df4 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x5e4>
    7c9d:       48 83 3c 24 06          cmpq   $0x6,(%rsp)
    7ca2:       0f 86 52 01 00 00       jbe    7dfa <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x5ea>
    7ca8:       49 8d 0c b2             lea    (%r10,%rsi,4),%rcx
    7cac:       31 c0                   xor    %eax,%eax
    7cae:       c5 f0 57 c9             vxorps %xmm1,%xmm1,%xmm1
    7cb2:       66 0f 1f 44 00 00       nopw   0x0(%rax,%rax,1)
    7cb8:       c5 fc 10 3c 01          vmovups (%rcx,%rax,1),%ymm7
    7cbd:       c5 c4 59 04 02          vmulps (%rdx,%rax,1),%ymm7,%ymm0
    7cc2:       48 83 c0 20             add    $0x20,%rax
    7cc6:       c5 f4 58 c8             vaddps %ymm0,%ymm1,%ymm1
    7cca:       49 39 c6                cmp    %rax,%r14
    7ccd:       75 e9                   jne    7cb8 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x4a8>
    7ccf:       c4 e3 7d 19 ca 01       vextractf128 $0x1,%ymm1,%xmm2
    7cd5:       48 83 7c 24 f0 00       cmpq   $0x0,-0x10(%rsp)
    7cdb:       c5 e8 58 c1             vaddps %xmm1,%xmm2,%xmm0
    7cdf:       c5 e8 58 d1             vaddps %xmm1,%xmm2,%xmm2
    7ce3:       c5 f8 12 d8             vmovhlps %xmm0,%xmm0,%xmm3
    7ce7:       c5 e0 58 d8             vaddps %xmm0,%xmm3,%xmm3
    7ceb:       c5 e0 c6 c3 55          vshufps $0x55,%xmm3,%xmm3,%xmm0
    7cf0:       c5 f8 58 c3             vaddps %xmm3,%xmm0,%xmm0
    7cf4:       0f 84 90 00 00 00       je     7d8a <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x57a>
    7cfa:       48 8b 44 24 e0          mov    -0x20(%rsp),%rax
    7cff:       4c 89 d9                mov    %r11,%rcx
    7d02:       48 29 c1                sub    %rax,%rcx
    7d05:       4c 8d 41 ff             lea    -0x1(%rcx),%r8
    7d09:       49 83 f8 02             cmp    $0x2,%r8
    7d0d:       76 31                   jbe    7d40 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x530>
    7d0f:       4c 8d 3c 06             lea    (%rsi,%rax,1),%r15
    7d13:       4e 8d 04 08             lea    (%rax,%r9,1),%r8
    7d17:       c4 81 78 10 04 ba       vmovups (%r10,%r15,4),%xmm0
    7d1d:       c4 a2 69 98 04 83       vfmadd132ps (%rbx,%r8,4),%xmm2,%xmm0
    7d23:       c5 f8 12 c8             vmovhlps %xmm0,%xmm0,%xmm1
    7d27:       c5 f0 58 c8             vaddps %xmm0,%xmm1,%xmm1
    7d2b:       c5 f0 c6 c1 55          vshufps $0x55,%xmm1,%xmm1,%xmm0
    7d30:       c5 f8 58 c1             vaddps %xmm1,%xmm0,%xmm0
    7d34:       f6 c1 03                test   $0x3,%cl
    7d37:       74 51                   je     7d8a <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x57a>
    7d39:       48 83 e1 fc             and    $0xfffffffffffffffc,%rcx
    7d3d:       48 01 c8                add    %rcx,%rax
    7d40:       48 8d 0c 06             lea    (%rsi,%rax,1),%rcx
    7d44:       4c 8d 04 85 00 00 00    lea    0x0(,%rax,4),%r8
    7d4b:       00 
    7d4c:       c4 c1 7a 10 3c 8a       vmovss (%r10,%rcx,4),%xmm7
    7d52:       48 8d 48 01             lea    0x1(%rax),%rcx
    7d56:       c4 e2 41 b9 04 82       vfmadd231ss (%rdx,%rax,4),%xmm7,%xmm0
    7d5c:       4c 39 d9                cmp    %r11,%rcx
    7d5f:       73 29                   jae    7d8a <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x57a>
    7d61:       48 01 f1                add    %rsi,%rcx
    7d64:       48 83 c0 02             add    $0x2,%rax
    7d68:       c4 a1 7a 10 7c 02 04    vmovss 0x4(%rdx,%r8,1),%xmm7
    7d6f:       c4 c2 41 b9 04 8a       vfmadd231ss (%r10,%rcx,4),%xmm7,%xmm0
    7d75:       4c 39 d8                cmp    %r11,%rax
    7d78:       73 10                   jae    7d8a <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x57a>
    7d7a:       48 01 f0                add    %rsi,%rax
    7d7d:       c4 c1 7a 10 3c 82       vmovss (%r10,%rax,4),%xmm7
    7d83:       c4 a2 41 b9 44 02 08    vfmadd231ss 0x8(%rdx,%r8,1),%xmm7,%xmm0
    7d8a:       c5 fa 11 07             vmovss %xmm0,(%rdi)
    7d8e:       48 83 c7 04             add    $0x4,%rdi
    7d92:       4c 01 de                add    %r11,%rsi
    7d95:       4c 39 ef                cmp    %r13,%rdi
    7d98:       0f 85 f6 fe ff ff       jne    7c94 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x484>
    7d9e:       48 83 44 24 d8 01       addq   $0x1,-0x28(%rsp)
    7da4:       4c 89 e8                mov    %r13,%rax
    7da7:       48 8b 7c 24 d8          mov    -0x28(%rsp),%rdi
    7dac:       4c 01 5c 24 18          add    %r11,0x18(%rsp)
    7db1:       48 39 7c 24 b8          cmp    %rdi,-0x48(%rsp)
    7db6:       0f 85 54 fb ff ff       jne    7910 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x100>
    7dbc:       c5 f8 77                vzeroupper
    7dbf:       48 8d 65 d8             lea    -0x28(%rbp),%rsp
    7dc3:       5b                      pop    %rbx
    7dc4:       41 5c                   pop    %r12
    7dc6:       41 5d                   pop    %r13
    7dc8:       41 5e                   pop    %r14
    7dca:       41 5f                   pop    %r15
    7dcc:       5d                      pop    %rbp
    7dcd:       c3                      ret
    7dce:       66 90                   xchg   %ax,%ax
    7dd0:       c5 f8 57 c0             vxorps %xmm0,%xmm0,%xmm0
    7dd4:       e9 4b fe ff ff          jmp    7c24 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x414>
    7dd9:       c5 f8 57 c0             vxorps %xmm0,%xmm0,%xmm0
    7ddd:       31 c0                   xor    %eax,%eax
    7ddf:       c5 f8 28 d0             vmovaps %xmm0,%xmm2
    7de3:       c5 f8 28 c8             vmovaps %xmm0,%xmm1
    7de7:       c5 f8 28 d8             vmovaps %xmm0,%xmm3
    7deb:       c5 f8 28 e0             vmovaps %xmm0,%xmm4
    7def:       e9 85 fc ff ff          jmp    7a79 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x269>
    7df4:       c5 f8 57 c0             vxorps %xmm0,%xmm0,%xmm0
    7df8:       eb 90                   jmp    7d8a <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x57a>
    7dfa:       c5 e8 57 d2             vxorps %xmm2,%xmm2,%xmm2
    7dfe:       31 c0                   xor    %eax,%eax
    7e00:       c5 f8 57 c0             vxorps %xmm0,%xmm0,%xmm0
    7e04:       e9 f6 fe ff ff          jmp    7cff <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x4ef>

Disassembly of section .fini:
(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$
```


### Structure the code for better vectorization by compiler (WITH OPENMP SIMD)

```aiignore
(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$ nm ./flat_matrix_benchmark_dot4_openmp_simd | grep dot4
0000000000007810 W _ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm
(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$ objdump -d --disassemble="_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm" ./flat_matrix_benchmark_dot4_openmp_simd 

./flat_matrix_benchmark_dot4_openmp_simd:     file format elf64-x86-64


Disassembly of section .init:

Disassembly of section .plt:

Disassembly of section .plt.got:

Disassembly of section .plt.sec:

Disassembly of section .text:

0000000000007810 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm>:
    7810:       f3 0f 1e fa             endbr64
    7814:       55                      push   %rbp
    7815:       48 89 e5                mov    %rsp,%rbp
    7818:       41 57                   push   %r15
    781a:       41 56                   push   %r14
    781c:       41 55                   push   %r13
    781e:       41 54                   push   %r12
    7820:       53                      push   %rbx
    7821:       48 89 cb                mov    %rcx,%rbx
    7824:       48 83 e4 e0             and    $0xffffffffffffffe0,%rsp
    7828:       48 81 ec 60 01 00 00    sub    $0x160,%rsp
    782f:       48 89 4c 24 20          mov    %rcx,0x20(%rsp)
    7834:       64 48 8b 0c 25 28 00    mov    %fs:0x28,%rcx
    783b:       00 00 
    783d:       48 89 8c 24 58 01 00    mov    %rcx,0x158(%rsp)
    7844:       00 
    7845:       31 c9                   xor    %ecx,%ecx
    7847:       48 85 db                test   %rbx,%rbx
    784a:       0f 84 c5 07 00 00       je     8015 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x805>
    7850:       4b 8d 04 09             lea    (%r9,%r9,1),%rax
    7854:       49 89 f4                mov    %rsi,%r12
    7857:       4d 89 c7                mov    %r8,%r15
    785a:       4d 89 cd                mov    %r9,%r13
    785d:       48 89 44 24 18          mov    %rax,0x18(%rsp)
    7862:       4c 89 c8                mov    %r9,%rax
    7865:       49 89 fa                mov    %rdi,%r10
    7868:       4a 8d 1c 85 00 00 00    lea    0x0(,%r8,4),%rbx
    786f:       00 
    7870:       48 c1 e0 04             shl    $0x4,%rax
    7874:       4e 8d 1c 8d 00 00 00    lea    0x0(,%r9,4),%r11
    787b:       00 
    787c:       48 8d 34 1a             lea    (%rdx,%rbx,1),%rsi
    7880:       48 c7 44 24 28 00 00    movq   $0x0,0x28(%rsp)
    7887:       00 00 
    7889:       48 89 44 24 10          mov    %rax,0x10(%rsp)
    788e:       4b 8d 04 cc             lea    (%r12,%r9,8),%rax
    7892:       c5 d1 ef ed             vpxor  %xmm5,%xmm5,%xmm5
    7896:       48 89 44 24 08          mov    %rax,0x8(%rsp)
    789b:       4c 89 c8                mov    %r9,%rax
    789e:       48 c1 e8 03             shr    $0x3,%rax
    78a2:       48 c1 e0 05             shl    $0x5,%rax
    78a6:       49 89 c6                mov    %rax,%r14
    78a9:       4c 89 c8                mov    %r9,%rax
    78ac:       48 83 e0 f8             and    $0xfffffffffffffff8,%rax
    78b0:       4c 89 74 24 48          mov    %r14,0x48(%rsp)
    78b5:       49 89 de                mov    %rbx,%r14
    78b8:       4c 89 db                mov    %r11,%rbx
    78bb:       48 89 44 24 50          mov    %rax,0x50(%rsp)
    78c0:       49 8d 40 fc             lea    -0x4(%r8),%rax
    78c4:       48 83 e0 fc             and    $0xfffffffffffffffc,%rax
    78c8:       48 83 c0 08             add    $0x8,%rax
    78cc:       48 89 84 24 80 00 00    mov    %rax,0x80(%rsp)
    78d3:       00 
    78d4:       4c 89 c8                mov    %r9,%rax
    78d7:       83 e0 07                and    $0x7,%eax
    78da:       48 89 44 24 58          mov    %rax,0x58(%rsp)
    78df:       90                      nop
    78e0:       48 89 f0                mov    %rsi,%rax
    78e3:       4c 29 f0                sub    %r14,%rax
    78e6:       48 89 84 24 88 00 00    mov    %rax,0x88(%rsp)
    78ed:       00 
    78ee:       49 83 ff 03             cmp    $0x3,%r15
    78f2:       0f 86 4d 07 00 00       jbe    8045 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x835>
    78f8:       48 89 74 24 40          mov    %rsi,0x40(%rsp)
    78fd:       48 8b 54 24 08          mov    0x8(%rsp),%rdx
    7902:       4c 89 e1                mov    %r12,%rcx
    7905:       31 ff                   xor    %edi,%edi
    7907:       4c 89 74 24 38          mov    %r14,0x38(%rsp)
    790c:       4c 8b 44 24 18          mov    0x18(%rsp),%r8
    7911:       b8 04 00 00 00          mov    $0x4,%eax
    7916:       4c 8d 9c 24 a0 00 00    lea    0xa0(%rsp),%r11
    791d:       00 
    791e:       4c 89 5c 24 68          mov    %r11,0x68(%rsp)
    7923:       4c 8d 9c 24 c0 00 00    lea    0xc0(%rsp),%r11
    792a:       00 
    792b:       4c 89 5c 24 60          mov    %r11,0x60(%rsp)
    7930:       4c 8d 9c 24 e0 00 00    lea    0xe0(%rsp),%r11
    7937:       00 
    7938:       4c 89 5c 24 78          mov    %r11,0x78(%rsp)
    793d:       4c 8d 9c 24 00 01 00    lea    0x100(%rsp),%r11
    7944:       00 
    7945:       4c 89 7c 24 30          mov    %r15,0x30(%rsp)
    794a:       4c 89 5c 24 70          mov    %r11,0x70(%rsp)
    794f:       4c 8b 5c 24 10          mov    0x10(%rsp),%r11
    7954:       0f 1f 40 00             nopl   0x0(%rax)
    7958:       49 8d 74 3d 00          lea    0x0(%r13,%rdi,1),%rsi
    795d:       48 89 b4 24 98 00 00    mov    %rsi,0x98(%rsp)
    7964:       00 
    7965:       4b 8d 74 05 00          lea    0x0(%r13,%r8,1),%rsi
    796a:       48 89 b4 24 90 00 00    mov    %rsi,0x90(%rsp)
    7971:       00 
    7972:       48 8b 74 24 68          mov    0x68(%rsp),%rsi
    7977:       c5 fd 7f 2e             vmovdqa %ymm5,(%rsi)
    797b:       48 8b 74 24 60          mov    0x60(%rsp),%rsi
    7980:       c5 fd 7f 2e             vmovdqa %ymm5,(%rsi)
    7984:       48 8b 74 24 78          mov    0x78(%rsp),%rsi
    7989:       c5 fd 7f 2e             vmovdqa %ymm5,(%rsi)
    798d:       48 8b 74 24 70          mov    0x70(%rsp),%rsi
    7992:       c5 fd 7f 2e             vmovdqa %ymm5,(%rsi)
    7996:       4d 85 ed                test   %r13,%r13
    7999:       0f 84 a5 03 00 00       je     7d44 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x534>
    799f:       49 83 fd 07             cmp    $0x7,%r13
    79a3:       0f 86 8e 06 00 00       jbe    8037 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x827>
    79a9:       c5 e0 57 db             vxorps %xmm3,%xmm3,%xmm3
    79ad:       4c 8b 7c 24 48          mov    0x48(%rsp),%r15
    79b2:       4c 8d 34 0b             lea    (%rbx,%rcx,1),%r14
    79b6:       31 f6                   xor    %esi,%esi
    79b8:       4c 8d 0c 13             lea    (%rbx,%rdx,1),%r9
    79bc:       c5 fc 28 d3             vmovaps %ymm3,%ymm2
    79c0:       c5 fc 28 cb             vmovaps %ymm3,%ymm1
    79c4:       c5 fc 28 e3             vmovaps %ymm3,%ymm4
    79c8:       0f 1f 84 00 00 00 00    nopl   0x0(%rax,%rax,1)
    79cf:       00 
    79d0:       c4 c1 7c 10 04 32       vmovups (%r10,%rsi,1),%ymm0
    79d6:       c4 e2 7d b8 1c 31       vfmadd231ps (%rcx,%rsi,1),%ymm0,%ymm3
    79dc:       c4 c2 7d b8 14 36       vfmadd231ps (%r14,%rsi,1),%ymm0,%ymm2
    79e2:       c4 e2 7d b8 0c 32       vfmadd231ps (%rdx,%rsi,1),%ymm0,%ymm1
    79e8:       c4 c2 7d b8 24 31       vfmadd231ps (%r9,%rsi,1),%ymm0,%ymm4
    79ee:       48 83 c6 20             add    $0x20,%rsi
    79f2:       4c 39 fe                cmp    %r15,%rsi
    79f5:       75 d9                   jne    79d0 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x1c0>
    79f7:       c5 fc 29 9c 24 00 01    vmovaps %ymm3,0x100(%rsp)
    79fe:       00 00 
    7a00:       48 83 7c 24 58 00       cmpq   $0x0,0x58(%rsp)
    7a06:       c5 fc 29 94 24 e0 00    vmovaps %ymm2,0xe0(%rsp)
    7a0d:       00 00 
    7a0f:       c5 fc 29 8c 24 c0 00    vmovaps %ymm1,0xc0(%rsp)
    7a16:       00 00 
    7a18:       c5 fc 29 a4 24 a0 00    vmovaps %ymm4,0xa0(%rsp)
    7a1f:       00 00 
    7a21:       0f 84 1d 03 00 00       je     7d44 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x534>
    7a27:       48 8b 74 24 50          mov    0x50(%rsp),%rsi
    7a2c:       4c 8b bc 24 98 00 00    mov    0x98(%rsp),%r15
    7a33:       00 
    7a34:       4c 8d 0c 3e             lea    (%rsi,%rdi,1),%r9
    7a38:       c4 c1 7a 10 1c b2       vmovss (%r10,%rsi,4),%xmm3
    7a3e:       4c 8d 34 b5 00 00 00    lea    0x0(,%rsi,4),%r14
    7a45:       00 
    7a46:       c5 fa 10 bc 24 a0 00    vmovss 0xa0(%rsp),%xmm7
    7a4d:       00 00 
    7a4f:       c4 81 7a 10 14 8c       vmovss (%r12,%r9,4),%xmm2
    7a55:       c4 e2 61 a9 94 24 00    vfmadd213ss 0x100(%rsp),%xmm3,%xmm2
    7a5c:       01 00 00 
    7a5f:       4d 8d 0c 37             lea    (%r15,%rsi,1),%r9
    7a63:       c4 81 7a 10 0c 8c       vmovss (%r12,%r9,4),%xmm1
    7a69:       4e 8d 0c 06             lea    (%rsi,%r8,1),%r9
    7a6d:       c4 e2 61 a9 8c 24 e0    vfmadd213ss 0xe0(%rsp),%xmm3,%xmm1
    7a74:       00 00 00 
    7a77:       c4 81 7a 10 04 8c       vmovss (%r12,%r9,4),%xmm0
    7a7d:       4c 8b 8c 24 90 00 00    mov    0x90(%rsp),%r9
    7a84:       00 
    7a85:       c4 e2 61 a9 84 24 c0    vfmadd213ss 0xc0(%rsp),%xmm3,%xmm0
    7a8c:       00 00 00 
    7a8f:       49 01 f1                add    %rsi,%r9
    7a92:       c5 fa 11 94 24 00 01    vmovss %xmm2,0x100(%rsp)
    7a99:       00 00 
    7a9b:       c4 82 41 99 1c 8c       vfmadd132ss (%r12,%r9,4),%xmm7,%xmm3
    7aa1:       4c 8d 4e 01             lea    0x1(%rsi),%r9
    7aa5:       c5 fa 11 8c 24 e0 00    vmovss %xmm1,0xe0(%rsp)
    7aac:       00 00 
    7aae:       c5 fa 11 84 24 c0 00    vmovss %xmm0,0xc0(%rsp)
    7ab5:       00 00 
    7ab7:       c5 fa 11 9c 24 a0 00    vmovss %xmm3,0xa0(%rsp)
    7abe:       00 00 
    7ac0:       4d 39 e9                cmp    %r13,%r9
    7ac3:       0f 83 7b 02 00 00       jae    7d44 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x534>
    7ac9:       4e 8d 3c 0f             lea    (%rdi,%r9,1),%r15
    7acd:       c4 81 7a 10 64 32 04    vmovss 0x4(%r10,%r14,1),%xmm4
    7ad4:       c4 82 59 b9 14 bc       vfmadd231ss (%r12,%r15,4),%xmm4,%xmm2
    7ada:       4c 8b bc 24 98 00 00    mov    0x98(%rsp),%r15
    7ae1:       00 
    7ae2:       4d 01 cf                add    %r9,%r15
    7ae5:       c4 82 59 b9 0c bc       vfmadd231ss (%r12,%r15,4),%xmm4,%xmm1
    7aeb:       4f 8d 3c 08             lea    (%r8,%r9,1),%r15
    7aef:       c4 82 59 b9 04 bc       vfmadd231ss (%r12,%r15,4),%xmm4,%xmm0
    7af5:       4c 8b bc 24 90 00 00    mov    0x90(%rsp),%r15
    7afc:       00 
    7afd:       c5 fa 11 94 24 00 01    vmovss %xmm2,0x100(%rsp)
    7b04:       00 00 
    7b06:       4d 01 f9                add    %r15,%r9
    7b09:       c4 82 61 99 24 8c       vfmadd132ss (%r12,%r9,4),%xmm3,%xmm4
    7b0f:       4c 8d 4e 02             lea    0x2(%rsi),%r9
    7b13:       c5 fa 11 8c 24 e0 00    vmovss %xmm1,0xe0(%rsp)
    7b1a:       00 00 
    7b1c:       c5 fa 11 84 24 c0 00    vmovss %xmm0,0xc0(%rsp)
    7b23:       00 00 
    7b25:       c5 fa 11 a4 24 a0 00    vmovss %xmm4,0xa0(%rsp)
    7b2c:       00 00 
    7b2e:       4d 39 e9                cmp    %r13,%r9
    7b31:       0f 83 0d 02 00 00       jae    7d44 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x534>
    7b37:       4e 8d 3c 0f             lea    (%rdi,%r9,1),%r15
    7b3b:       c4 81 7a 10 5c 32 08    vmovss 0x8(%r10,%r14,1),%xmm3
    7b42:       c4 82 61 b9 14 bc       vfmadd231ss (%r12,%r15,4),%xmm3,%xmm2
    7b48:       4c 8b bc 24 98 00 00    mov    0x98(%rsp),%r15
    7b4f:       00 
    7b50:       4d 01 cf                add    %r9,%r15
    7b53:       c4 82 61 b9 0c bc       vfmadd231ss (%r12,%r15,4),%xmm3,%xmm1
    7b59:       4f 8d 3c 08             lea    (%r8,%r9,1),%r15
    7b5d:       c4 82 61 b9 04 bc       vfmadd231ss (%r12,%r15,4),%xmm3,%xmm0
    7b63:       4c 8b bc 24 90 00 00    mov    0x90(%rsp),%r15
    7b6a:       00 
    7b6b:       c5 fa 11 94 24 00 01    vmovss %xmm2,0x100(%rsp)
    7b72:       00 00 
    7b74:       4d 01 f9                add    %r15,%r9
    7b77:       c4 82 59 99 1c 8c       vfmadd132ss (%r12,%r9,4),%xmm4,%xmm3
    7b7d:       4c 8d 4e 03             lea    0x3(%rsi),%r9
    7b81:       c5 fa 11 8c 24 e0 00    vmovss %xmm1,0xe0(%rsp)
    7b88:       00 00 
    7b8a:       c5 fa 11 84 24 c0 00    vmovss %xmm0,0xc0(%rsp)
    7b91:       00 00 
    7b93:       c5 fa 11 9c 24 a0 00    vmovss %xmm3,0xa0(%rsp)
    7b9a:       00 00 
    7b9c:       4d 39 e9                cmp    %r13,%r9
    7b9f:       0f 83 9f 01 00 00       jae    7d44 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x534>
    7ba5:       4e 8d 3c 0f             lea    (%rdi,%r9,1),%r15
    7ba9:       c4 81 7a 10 64 32 0c    vmovss 0xc(%r10,%r14,1),%xmm4
    7bb0:       c4 82 59 b9 14 bc       vfmadd231ss (%r12,%r15,4),%xmm4,%xmm2
    7bb6:       4c 8b bc 24 98 00 00    mov    0x98(%rsp),%r15
    7bbd:       00 
    7bbe:       4d 01 cf                add    %r9,%r15
    7bc1:       c4 82 59 b9 0c bc       vfmadd231ss (%r12,%r15,4),%xmm4,%xmm1
    7bc7:       4f 8d 3c 08             lea    (%r8,%r9,1),%r15
    7bcb:       c4 82 59 b9 04 bc       vfmadd231ss (%r12,%r15,4),%xmm4,%xmm0
    7bd1:       4c 8b bc 24 90 00 00    mov    0x90(%rsp),%r15
    7bd8:       00 
    7bd9:       c5 fa 11 94 24 00 01    vmovss %xmm2,0x100(%rsp)
    7be0:       00 00 
    7be2:       4d 01 f9                add    %r15,%r9
    7be5:       c4 82 61 99 24 8c       vfmadd132ss (%r12,%r9,4),%xmm3,%xmm4
    7beb:       4c 8d 4e 04             lea    0x4(%rsi),%r9
    7bef:       c5 fa 11 8c 24 e0 00    vmovss %xmm1,0xe0(%rsp)
    7bf6:       00 00 
    7bf8:       c5 fa 11 84 24 c0 00    vmovss %xmm0,0xc0(%rsp)
    7bff:       00 00 
    7c01:       c5 fa 11 a4 24 a0 00    vmovss %xmm4,0xa0(%rsp)
    7c08:       00 00 
    7c0a:       4d 39 e9                cmp    %r13,%r9
    7c0d:       0f 83 31 01 00 00       jae    7d44 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x534>
    7c13:       4e 8d 3c 0f             lea    (%rdi,%r9,1),%r15
    7c17:       c4 81 7a 10 5c 32 10    vmovss 0x10(%r10,%r14,1),%xmm3
    7c1e:       c4 82 61 b9 14 bc       vfmadd231ss (%r12,%r15,4),%xmm3,%xmm2
    7c24:       4c 8b bc 24 98 00 00    mov    0x98(%rsp),%r15
    7c2b:       00 
    7c2c:       4d 01 cf                add    %r9,%r15
    7c2f:       c4 82 61 b9 0c bc       vfmadd231ss (%r12,%r15,4),%xmm3,%xmm1
    7c35:       4f 8d 3c 08             lea    (%r8,%r9,1),%r15
    7c39:       c4 82 61 b9 04 bc       vfmadd231ss (%r12,%r15,4),%xmm3,%xmm0
    7c3f:       4c 8b bc 24 90 00 00    mov    0x90(%rsp),%r15
    7c46:       00 
    7c47:       c5 fa 11 94 24 00 01    vmovss %xmm2,0x100(%rsp)
    7c4e:       00 00 
    7c50:       4d 01 f9                add    %r15,%r9
    7c53:       c4 82 59 99 1c 8c       vfmadd132ss (%r12,%r9,4),%xmm4,%xmm3
    7c59:       4c 8d 4e 05             lea    0x5(%rsi),%r9
    7c5d:       c5 fa 11 8c 24 e0 00    vmovss %xmm1,0xe0(%rsp)
    7c64:       00 00 
    7c66:       c5 fa 11 84 24 c0 00    vmovss %xmm0,0xc0(%rsp)
    7c6d:       00 00 
    7c6f:       c5 fa 11 9c 24 a0 00    vmovss %xmm3,0xa0(%rsp)
    7c76:       00 00 
    7c78:       4d 39 e9                cmp    %r13,%r9
    7c7b:       0f 83 c3 00 00 00       jae    7d44 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x534>
    7c81:       4e 8d 3c 0f             lea    (%rdi,%r9,1),%r15
    7c85:       c4 81 7a 10 64 32 14    vmovss 0x14(%r10,%r14,1),%xmm4
    7c8c:       48 83 c6 06             add    $0x6,%rsi
    7c90:       c4 82 59 b9 14 bc       vfmadd231ss (%r12,%r15,4),%xmm4,%xmm2
    7c96:       4c 8b bc 24 98 00 00    mov    0x98(%rsp),%r15
    7c9d:       00 
    7c9e:       4d 01 cf                add    %r9,%r15
    7ca1:       c4 82 59 b9 0c bc       vfmadd231ss (%r12,%r15,4),%xmm4,%xmm1
    7ca7:       4f 8d 3c 08             lea    (%r8,%r9,1),%r15
    7cab:       c4 82 59 b9 04 bc       vfmadd231ss (%r12,%r15,4),%xmm4,%xmm0
    7cb1:       4c 8b bc 24 90 00 00    mov    0x90(%rsp),%r15
    7cb8:       00 
    7cb9:       c5 fa 11 94 24 00 01    vmovss %xmm2,0x100(%rsp)
    7cc0:       00 00 
    7cc2:       4d 01 f9                add    %r15,%r9
    7cc5:       c4 82 61 99 24 8c       vfmadd132ss (%r12,%r9,4),%xmm3,%xmm4
    7ccb:       c5 fa 11 8c 24 e0 00    vmovss %xmm1,0xe0(%rsp)
    7cd2:       00 00 
    7cd4:       c5 fa 11 84 24 c0 00    vmovss %xmm0,0xc0(%rsp)
    7cdb:       00 00 
    7cdd:       c5 fa 11 a4 24 a0 00    vmovss %xmm4,0xa0(%rsp)
    7ce4:       00 00 
    7ce6:       4c 39 ee                cmp    %r13,%rsi
    7ce9:       73 59                   jae    7d44 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x534>
    7ceb:       4c 8d 0c 37             lea    (%rdi,%rsi,1),%r9
    7cef:       c4 81 7a 10 5c 32 18    vmovss 0x18(%r10,%r14,1),%xmm3
    7cf6:       49 01 f7                add    %rsi,%r15
    7cf9:       c4 82 61 b9 14 8c       vfmadd231ss (%r12,%r9,4),%xmm3,%xmm2
    7cff:       4c 8b 8c 24 98 00 00    mov    0x98(%rsp),%r9
    7d06:       00 
    7d07:       49 01 f1                add    %rsi,%r9
    7d0a:       c4 82 61 b9 0c 8c       vfmadd231ss (%r12,%r9,4),%xmm3,%xmm1
    7d10:       4d 8d 0c 30             lea    (%r8,%rsi,1),%r9
    7d14:       c4 82 61 b9 04 8c       vfmadd231ss (%r12,%r9,4),%xmm3,%xmm0
    7d1a:       c4 82 59 99 1c bc       vfmadd132ss (%r12,%r15,4),%xmm4,%xmm3
    7d20:       c5 fa 11 94 24 00 01    vmovss %xmm2,0x100(%rsp)
    7d27:       00 00 
    7d29:       c5 fa 11 8c 24 e0 00    vmovss %xmm1,0xe0(%rsp)
    7d30:       00 00 
    7d32:       c5 fa 11 84 24 c0 00    vmovss %xmm0,0xc0(%rsp)
    7d39:       00 00 
    7d3b:       c5 fa 11 9c 24 a0 00    vmovss %xmm3,0xa0(%rsp)
    7d42:       00 00 
    7d44:       48 01 df                add    %rbx,%rdi
    7d47:       49 01 d8                add    %rbx,%r8
    7d4a:       4c 01 d9                add    %r11,%rcx
    7d4d:       4c 01 da                add    %r11,%rdx
    7d50:       c5 fc 28 b4 24 00 01    vmovaps 0x100(%rsp),%ymm6
    7d57:       00 00 
    7d59:       48 8b b4 24 88 00 00    mov    0x88(%rsp),%rsi
    7d60:       00 
    7d61:       c5 fc 28 bc 24 e0 00    vmovaps 0xe0(%rsp),%ymm7
    7d68:       00 00 
    7d6a:       c4 e3 7d 19 f0 01       vextractf128 $0x1,%ymm6,%xmm0
    7d70:       c5 f8 58 84 24 00 01    vaddps 0x100(%rsp),%xmm0,%xmm0
    7d77:       00 00 
    7d79:       c5 fc 28 b4 24 c0 00    vmovaps 0xc0(%rsp),%ymm6
    7d80:       00 00 
    7d82:       c5 f8 12 d0             vmovhlps %xmm0,%xmm0,%xmm2
    7d86:       c5 e8 58 d0             vaddps %xmm0,%xmm2,%xmm2
    7d8a:       c5 e8 c6 ca 55          vshufps $0x55,%xmm2,%xmm2,%xmm1
    7d8f:       c5 f0 58 ca             vaddps %xmm2,%xmm1,%xmm1
    7d93:       c4 e3 7d 19 fa 01       vextractf128 $0x1,%ymm7,%xmm2
    7d99:       c5 e8 58 94 24 e0 00    vaddps 0xe0(%rsp),%xmm2,%xmm2
    7da0:       00 00 
    7da2:       c5 fc 28 bc 24 a0 00    vmovaps 0xa0(%rsp),%ymm7
    7da9:       00 00 
    7dab:       c5 e8 12 c2             vmovhlps %xmm2,%xmm2,%xmm0
    7daf:       c5 f8 58 c2             vaddps %xmm2,%xmm0,%xmm0
    7db3:       c5 f8 c6 d0 55          vshufps $0x55,%xmm0,%xmm0,%xmm2
    7db8:       c5 e8 58 d0             vaddps %xmm0,%xmm2,%xmm2
    7dbc:       c4 e3 7d 19 f0 01       vextractf128 $0x1,%ymm6,%xmm0
    7dc2:       c5 f8 58 84 24 c0 00    vaddps 0xc0(%rsp),%xmm0,%xmm0
    7dc9:       00 00 
    7dcb:       c5 f8 12 d8             vmovhlps %xmm0,%xmm0,%xmm3
    7dcf:       c5 f0 14 ca             vunpcklps %xmm2,%xmm1,%xmm1
    7dd3:       c5 e0 58 d8             vaddps %xmm0,%xmm3,%xmm3
    7dd7:       c5 e0 c6 c3 55          vshufps $0x55,%xmm3,%xmm3,%xmm0
    7ddc:       c5 f8 58 c3             vaddps %xmm3,%xmm0,%xmm0
    7de0:       c4 e3 7d 19 fb 01       vextractf128 $0x1,%ymm7,%xmm3
    7de6:       c5 e0 58 9c 24 a0 00    vaddps 0xa0(%rsp),%xmm3,%xmm3
    7ded:       00 00 
    7def:       c5 e0 12 e3             vmovhlps %xmm3,%xmm3,%xmm4
    7df3:       c5 d8 58 e3             vaddps %xmm3,%xmm4,%xmm4
    7df7:       c5 d8 c6 dc 55          vshufps $0x55,%xmm4,%xmm4,%xmm3
    7dfc:       c5 e0 58 dc             vaddps %xmm4,%xmm3,%xmm3
    7e00:       c5 f8 14 c3             vunpcklps %xmm3,%xmm0,%xmm0
    7e04:       c5 f0 16 c0             vmovlhps %xmm0,%xmm1,%xmm0
    7e08:       c5 f8 11 44 86 f0       vmovups %xmm0,-0x10(%rsi,%rax,4)
    7e0e:       48 8d 70 04             lea    0x4(%rax),%rsi
    7e12:       48 39 b4 24 80 00 00    cmp    %rsi,0x80(%rsp)
    7e19:       00 
    7e1a:       74 0c                   je     7e28 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x618>
    7e1c:       48 89 f0                mov    %rsi,%rax
    7e1f:       e9 34 fb ff ff          jmp    7958 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x148>
    7e24:       0f 1f 40 00             nopl   0x0(%rax)
    7e28:       48 8b 74 24 40          mov    0x40(%rsp),%rsi
    7e2d:       4c 8b 74 24 38          mov    0x38(%rsp),%r14
    7e32:       4c 8b 7c 24 30          mov    0x30(%rsp),%r15
    7e37:       4c 39 f8                cmp    %r15,%rax
    7e3a:       0f 83 b6 01 00 00       jae    7ff6 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x7e6>
    7e40:       48 8b bc 24 88 00 00    mov    0x88(%rsp),%rdi
    7e47:       00 
    7e48:       4c 8b 5c 24 48          mov    0x48(%rsp),%r11
    7e4d:       4c 8d 8c 24 20 01 00    lea    0x120(%rsp),%r9
    7e54:       00 
    7e55:       48 8d 14 87             lea    (%rdi,%rax,4),%rdx
    7e59:       49 0f af c5             imul   %r13,%rax
    7e5d:       0f 1f 00                nopl   (%rax)
    7e60:       c4 c1 7d 7f 29          vmovdqa %ymm5,(%r9)
    7e65:       4d 85 ed                test   %r13,%r13
    7e68:       0f 84 4a 01 00 00       je     7fb8 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x7a8>
    7e6e:       49 83 fd 07             cmp    $0x7,%r13
    7e72:       0f 86 c6 01 00 00       jbe    803e <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x82e>
    7e78:       49 8d 3c 84             lea    (%r12,%rax,4),%rdi
    7e7c:       31 c9                   xor    %ecx,%ecx
    7e7e:       c5 f0 57 c9             vxorps %xmm1,%xmm1,%xmm1
    7e82:       66 0f 1f 44 00 00       nopw   0x0(%rax,%rax,1)
    7e88:       c4 c1 7c 10 34 0a       vmovups (%r10,%rcx,1),%ymm6
    7e8e:       c5 cc 59 04 0f          vmulps (%rdi,%rcx,1),%ymm6,%ymm0
    7e93:       48 83 c1 20             add    $0x20,%rcx
    7e97:       c5 f4 58 c8             vaddps %ymm0,%ymm1,%ymm1
    7e9b:       4c 39 d9                cmp    %r11,%rcx
    7e9e:       75 e8                   jne    7e88 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x678>
    7ea0:       c5 fc 29 8c 24 20 01    vmovaps %ymm1,0x120(%rsp)
    7ea7:       00 00 
    7ea9:       48 83 7c 24 58 00       cmpq   $0x0,0x58(%rsp)
    7eaf:       0f 84 03 01 00 00       je     7fb8 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x7a8>
    7eb5:       48 8b 4c 24 50          mov    0x50(%rsp),%rcx
    7eba:       4c 8d 04 08             lea    (%rax,%rcx,1),%r8
    7ebe:       48 8d 3c 8d 00 00 00    lea    0x0(,%rcx,4),%rdi
    7ec5:       00 
    7ec6:       c5 fa 10 b4 24 20 01    vmovss 0x120(%rsp),%xmm6
    7ecd:       00 00 
    7ecf:       c4 81 7a 10 04 84       vmovss (%r12,%r8,4),%xmm0
    7ed5:       c4 c2 49 99 04 8a       vfmadd132ss (%r10,%rcx,4),%xmm6,%xmm0
    7edb:       4c 8d 41 01             lea    0x1(%rcx),%r8
    7edf:       c5 fa 11 84 24 20 01    vmovss %xmm0,0x120(%rsp)
    7ee6:       00 00 
    7ee8:       4d 39 e8                cmp    %r13,%r8
    7eeb:       0f 83 c7 00 00 00       jae    7fb8 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x7a8>
    7ef1:       49 01 c0                add    %rax,%r8
    7ef4:       c4 c1 7a 10 7c 3a 04    vmovss 0x4(%r10,%rdi,1),%xmm7
    7efb:       c4 82 41 b9 04 84       vfmadd231ss (%r12,%r8,4),%xmm7,%xmm0
    7f01:       4c 8d 41 02             lea    0x2(%rcx),%r8
    7f05:       c5 fa 11 84 24 20 01    vmovss %xmm0,0x120(%rsp)
    7f0c:       00 00 
    7f0e:       4d 39 e8                cmp    %r13,%r8
    7f11:       0f 83 a1 00 00 00       jae    7fb8 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x7a8>
    7f17:       49 01 c0                add    %rax,%r8
    7f1a:       c4 c1 7a 10 74 3a 08    vmovss 0x8(%r10,%rdi,1),%xmm6
    7f21:       c4 82 49 b9 04 84       vfmadd231ss (%r12,%r8,4),%xmm6,%xmm0
    7f27:       4c 8d 41 03             lea    0x3(%rcx),%r8
    7f2b:       c5 fa 11 84 24 20 01    vmovss %xmm0,0x120(%rsp)
    7f32:       00 00 
    7f34:       4d 39 e8                cmp    %r13,%r8
    7f37:       73 7f                   jae    7fb8 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x7a8>
    7f39:       49 01 c0                add    %rax,%r8
    7f3c:       c4 c1 7a 10 7c 3a 0c    vmovss 0xc(%r10,%rdi,1),%xmm7
    7f43:       c4 82 41 b9 04 84       vfmadd231ss (%r12,%r8,4),%xmm7,%xmm0
    7f49:       4c 8d 41 04             lea    0x4(%rcx),%r8
    7f4d:       c5 fa 11 84 24 20 01    vmovss %xmm0,0x120(%rsp)
    7f54:       00 00 
    7f56:       4d 39 e8                cmp    %r13,%r8
    7f59:       73 5d                   jae    7fb8 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x7a8>
    7f5b:       49 01 c0                add    %rax,%r8
    7f5e:       c4 81 7a 10 34 84       vmovss (%r12,%r8,4),%xmm6
    7f64:       c4 c2 49 b9 44 3a 10    vfmadd231ss 0x10(%r10,%rdi,1),%xmm6,%xmm0
    7f6b:       4c 8d 41 05             lea    0x5(%rcx),%r8
    7f6f:       c5 fa 11 84 24 20 01    vmovss %xmm0,0x120(%rsp)
    7f76:       00 00 
    7f78:       4d 39 e8                cmp    %r13,%r8
    7f7b:       73 3b                   jae    7fb8 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x7a8>
    7f7d:       49 01 c0                add    %rax,%r8
    7f80:       48 83 c1 06             add    $0x6,%rcx
    7f84:       c4 81 7a 10 3c 84       vmovss (%r12,%r8,4),%xmm7
    7f8a:       c4 c2 41 b9 44 3a 14    vfmadd231ss 0x14(%r10,%rdi,1),%xmm7,%xmm0
    7f91:       c5 fa 11 84 24 20 01    vmovss %xmm0,0x120(%rsp)
    7f98:       00 00 
    7f9a:       4c 39 e9                cmp    %r13,%rcx
    7f9d:       73 19                   jae    7fb8 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x7a8>
    7f9f:       48 01 c1                add    %rax,%rcx
    7fa2:       c4 c1 7a 10 74 3a 18    vmovss 0x18(%r10,%rdi,1),%xmm6
    7fa9:       c4 c2 49 b9 04 8c       vfmadd231ss (%r12,%rcx,4),%xmm6,%xmm0
    7faf:       c5 fa 11 84 24 20 01    vmovss %xmm0,0x120(%rsp)
    7fb6:       00 00 
    7fb8:       c5 fc 28 bc 24 20 01    vmovaps 0x120(%rsp),%ymm7
    7fbf:       00 00 
    7fc1:       48 83 c2 04             add    $0x4,%rdx
    7fc5:       4c 01 e8                add    %r13,%rax
    7fc8:       c4 e3 7d 19 f9 01       vextractf128 $0x1,%ymm7,%xmm1
    7fce:       c5 f0 58 8c 24 20 01    vaddps 0x120(%rsp),%xmm1,%xmm1
    7fd5:       00 00 
    7fd7:       c5 f0 12 c1             vmovhlps %xmm1,%xmm1,%xmm0
    7fdb:       c5 f8 58 c1             vaddps %xmm1,%xmm0,%xmm0
    7fdf:       c5 f8 c6 c8 55          vshufps $0x55,%xmm0,%xmm0,%xmm1
    7fe4:       c5 f0 58 c0             vaddps %xmm0,%xmm1,%xmm0
    7fe8:       c5 fa 11 42 fc          vmovss %xmm0,-0x4(%rdx)
    7fed:       48 39 f2                cmp    %rsi,%rdx
    7ff0:       0f 85 6a fe ff ff       jne    7e60 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x650>
    7ff6:       48 83 44 24 28 01       addq   $0x1,0x28(%rsp)
    7ffc:       4c 01 f6                add    %r14,%rsi
    7fff:       48 8b 44 24 28          mov    0x28(%rsp),%rax
    8004:       49 01 da                add    %rbx,%r10
    8007:       48 39 44 24 20          cmp    %rax,0x20(%rsp)
    800c:       0f 85 ce f8 ff ff       jne    78e0 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0xd0>
    8012:       c5 f8 77                vzeroupper
    8015:       48 8b 84 24 58 01 00    mov    0x158(%rsp),%rax
    801c:       00 
    801d:       64 48 2b 04 25 28 00    sub    %fs:0x28,%rax
    8024:       00 00 
    8026:       75 24                   jne    804c <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x83c>
    8028:       48 8d 65 d8             lea    -0x28(%rbp),%rsp
    802c:       5b                      pop    %rbx
    802d:       41 5c                   pop    %r12
    802f:       41 5d                   pop    %r13
    8031:       41 5e                   pop    %r14
    8033:       41 5f                   pop    %r15
    8035:       5d                      pop    %rbp
    8036:       c3                      ret
    8037:       31 f6                   xor    %esi,%esi
    8039:       e9 ee f9 ff ff          jmp    7a2c <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x21c>
    803e:       31 c9                   xor    %ecx,%ecx
    8040:       e9 75 fe ff ff          jmp    7eba <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x6aa>
    8045:       31 c0                   xor    %eax,%eax
    8047:       e9 eb fd ff ff          jmp    7e37 <_ZN10FlatMatrixIfE41multiply_transposed_rhs_float_dot4_kernelEPKfS2_Pfmmm+0x627>
    804c:       e8 2f a5 ff ff          call   2580 <__stack_chk_fail@plt>

Disassembly of section .fini:
(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$
```



### Optimizing memory utilization and allocation

(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$ ./mlp_infer_avx2_max --export-dir ../cifar10_embedding_mlp/export --input ../cifar10_embedding_mlp/export/test_input_all/test_input_embeddings_all.bin --batch-size 128 --num-batches 1
================================================================================
Input dim    : 512
Num classes  : 10
Batch size   : 128
Total samples: 128
Total batches: 1
================================================================================
Processed batch 1/1 in 410.554 ms
================================================================================
Benchmark Results
================================================================================
Mean latency : 410.554 ms
Std dev      : 0 ms
Throughput   : 311.774 samples/sec
================================================================================
Wrote logits to: cpp_output_logits.bin
Wrote preds  to: cpp_output_preds.bin

--- Full Inference Profile ---
Step                      |      Elapsed |     Step Dur
-------------------------------------------------------
Batch_0_start             |          0 us |          0 us
Layer_0_start             |        222 us |        222 us
Before_matrix_init        |        224 us |          1 us
After_matrix_init         |       7795 us |       7570 us
After_matrix_mul          |      24950 us |      17155 us
After_matrix_add          |      25139 us |        189 us
After_matrix_relu         |      25248 us |        108 us
Layer_1_start             |      25960 us |        711 us
Before_matrix_init        |      25960 us |          0 us
After_matrix_init         |      73719 us |      47758 us
After_matrix_mul          |     223420 us |     149701 us
After_matrix_add          |     223565 us |        144 us
After_matrix_relu         |     223664 us |         99 us
Layer_2_start             |     227167 us |       3502 us
Before_matrix_init        |     227167 us |          0 us
After_matrix_init         |     267436 us |      40268 us
After_matrix_mul          |     406737 us |     139301 us
After_matrix_add          |     406890 us |        153 us
After_matrix_relu         |     406999 us |        108 us
Layer_3_start             |     410052 us |       3052 us
Before_matrix_init        |     410053 us |          1 us
After_matrix_init         |     410252 us |        198 us
After_matrix_mul          |     410551 us |        299 us
After_matrix_add          |     410552 us |          1 us
After_matrix_relu         |     410553 us |          0 us
Batch_0_end               |     410553 us |          0 us
-------------------------------------------------------




(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$ ./mlp_infer_avx2_max --export-dir ../cifar10_embedding_mlp/export --input ../cifar10_embedding_mlp/export/test_input_all/test_input_embeddings_all.bin --batch-size 256 --num-batches 1
================================================================================
Input dim    : 512
Num classes  : 10
Batch size   : 256
Total samples: 256
Total batches: 1
================================================================================
Processed batch 1/1 in 686.944 ms
================================================================================
Benchmark Results
================================================================================
Mean latency : 686.944 ms
Std dev      : 0 ms
Throughput   : 372.665 samples/sec
================================================================================
Wrote logits to: cpp_output_logits.bin
Wrote preds  to: cpp_output_preds.bin

--- Full Inference Profile ---
Step                      |      Elapsed |     Step Dur
-------------------------------------------------------
Batch_0_start             |          0 us |          0 us
Layer_0_start             |        464 us |        464 us
Before_matrix_init        |        466 us |          1 us
After_matrix_init         |       9877 us |       9410 us
After_matrix_mul          |      42715 us |      32838 us
After_matrix_add          |      43130 us |        414 us
After_matrix_relu         |      43414 us |        284 us
Layer_1_start             |      43950 us |        536 us
Before_matrix_init        |      43951 us |          0 us
After_matrix_init         |      93555 us |      49604 us
After_matrix_mul          |     370892 us |     277336 us
After_matrix_add          |     371191 us |        299 us
After_matrix_relu         |     371412 us |        221 us
Layer_2_start             |     373938 us |       2525 us
Before_matrix_init        |     373939 us |          0 us
After_matrix_init         |     415314 us |      41374 us
After_matrix_mul          |     683354 us |     268040 us
After_matrix_add          |     683665 us |        311 us
After_matrix_relu         |     683893 us |        227 us
Layer_3_start             |     685913 us |       2019 us
Before_matrix_init        |     685914 us |          1 us
After_matrix_init         |     686362 us |        448 us
After_matrix_mul          |     686941 us |        579 us
After_matrix_add          |     686943 us |          1 us
After_matrix_relu         |     686943 us |          0 us
Batch_0_end               |     686943 us |          0 us
-------------------------------------------------------



(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$ ./mlp_infer_avx2_max --export-dir ../cifar10_embedding_mlp/export --input ../cifar10_embedding_mlp/export/test_input_all/test_input_embeddings_all.bin --batch-size 64 --num-batches 1
================================================================================
Input dim    : 512
Num classes  : 10
Batch size   : 64
Total samples: 64
Total batches: 1
================================================================================
Processed batch 1/1 in 256.735 ms
================================================================================
Benchmark Results
================================================================================
Mean latency : 256.735 ms
Std dev      : 0 ms
Throughput   : 249.284 samples/sec
================================================================================
Wrote logits to: cpp_output_logits.bin
Wrote preds  to: cpp_output_preds.bin

--- Full Inference Profile ---
Step                      |      Elapsed |     Step Dur
-------------------------------------------------------
Batch_0_start             |          0 us |          0 us
Layer_0_start             |        115 us |        115 us
Before_matrix_init        |        117 us |          1 us
After_matrix_init         |       6484 us |       6367 us
After_matrix_mul          |      15779 us |       9294 us
After_matrix_add          |      15865 us |         85 us
After_matrix_relu         |      15903 us |         38 us
Layer_1_start             |      16391 us |        488 us
Before_matrix_init        |      16392 us |          0 us
After_matrix_init         |      62921 us |      46528 us
After_matrix_mul          |     143358 us |      80437 us
After_matrix_add          |     143430 us |         71 us
After_matrix_relu         |     143474 us |         44 us
Layer_2_start             |     145632 us |       2157 us
Before_matrix_init        |     145632 us |          0 us
After_matrix_init         |     185386 us |      39753 us
After_matrix_mul          |     254376 us |      68989 us
After_matrix_add          |     254458 us |         82 us
After_matrix_relu         |     254500 us |         42 us
Layer_3_start             |     256497 us |       1997 us
Before_matrix_init        |     256498 us |          1 us
After_matrix_init         |     256599 us |        101 us
After_matrix_mul          |     256733 us |        133 us
After_matrix_add          |     256734 us |          0 us
After_matrix_relu         |     256734 us |          0 us
Batch_0_end               |     256735 us |          0 us
-------------------------------------------------------



(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$ ./mlp_infer_avx2_max --export-dir ../cifar10_embedding_mlp/export --input ../cifar10_embedding_mlp/export/test_input_all/test_input_embeddings_all.bin --batch-size 512 --num-batches 1
================================================================================
Input dim    : 512
Num classes  : 10
Batch size   : 512
Total samples: 512
Total batches: 1
================================================================================
Processed batch 1/1 in 1291.05 ms
================================================================================
Benchmark Results
================================================================================
Mean latency : 1291.05 ms
Std dev      : 0 ms
Throughput   : 396.575 samples/sec
================================================================================
Wrote logits to: cpp_output_logits.bin
Wrote preds  to: cpp_output_preds.bin

--- Full Inference Profile ---
Step                      |      Elapsed |     Step Dur
-------------------------------------------------------
Batch_0_start             |          0 us |          0 us
Layer_0_start             |       1209 us |       1209 us
Before_matrix_init        |       1211 us |          1 us
After_matrix_init         |      16600 us |      15389 us
After_matrix_mul          |      82604 us |      66003 us
After_matrix_add          |      83446 us |        842 us
After_matrix_relu         |      84128 us |        681 us
Layer_1_start             |      84703 us |        575 us
Before_matrix_init        |      84704 us |          0 us
After_matrix_init         |     140873 us |      56169 us
After_matrix_mul          |     681820 us |     540946 us
After_matrix_add          |     682387 us |        567 us
After_matrix_relu         |     682882 us |        494 us
Layer_2_start             |     685095 us |       2212 us
Before_matrix_init        |     685096 us |          1 us
After_matrix_init         |     727015 us |      41918 us
After_matrix_mul          |    1285011 us |     557996 us
After_matrix_add          |    1285722 us |        711 us
After_matrix_relu         |    1286300 us |        577 us
Layer_3_start             |    1288743 us |       2443 us
Before_matrix_init        |    1288745 us |          1 us
After_matrix_init         |    1289342 us |        596 us
After_matrix_mul          |    1291049 us |       1707 us
After_matrix_add          |    1291052 us |          2 us
After_matrix_relu         |    1291052 us |          0 us
Batch_0_end               |    1291053 us |          1 us
-------------------------------------------------------



(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$ ./mlp_infer_avx2_max --export-dir ../cifar10_embedding_mlp/export --input ../cifar10_embedding_mlp/export/test_input_all/test_input_embeddings_all.bin --batch-size 1024 --num-batches 1
================================================================================
Input dim    : 512
Num classes  : 10
Batch size   : 1024
Total samples: 1024
Total batches: 1
================================================================================
Processed batch 1/1 in 2674.58 ms
================================================================================
Benchmark Results
================================================================================
Mean latency : 2674.58 ms
Std dev      : 0 ms
Throughput   : 382.864 samples/sec
================================================================================
Wrote logits to: cpp_output_logits.bin
Wrote preds  to: cpp_output_preds.bin

--- Full Inference Profile ---
Step                      |      Elapsed |     Step Dur
-------------------------------------------------------
Batch_0_start             |          0 us |          0 us
Layer_0_start             |       2426 us |       2426 us
Before_matrix_init        |       2429 us |          2 us
After_matrix_init         |      24843 us |      22414 us
After_matrix_mul          |     151765 us |     126922 us
After_matrix_add          |     153498 us |       1732 us
After_matrix_relu         |     155076 us |       1578 us
Layer_1_start             |     155616 us |        539 us
Before_matrix_init        |     155616 us |          0 us
After_matrix_init         |     214667 us |      59051 us
After_matrix_mul          |    1318072 us |    1103405 us
After_matrix_add          |    1319253 us |       1181 us
After_matrix_relu         |    1320243 us |        989 us
Layer_2_start             |    1323268 us |       3025 us
Before_matrix_init        |    1323268 us |          0 us
After_matrix_init         |    1383797 us |      60528 us
After_matrix_mul          |    2663602 us |    1279804 us
After_matrix_add          |    2665022 us |       1420 us
After_matrix_relu         |    2666171 us |       1149 us
Layer_3_start             |    2669253 us |       3082 us
Before_matrix_init        |    2669254 us |          1 us
After_matrix_init         |    2670927 us |       1673 us
After_matrix_mul          |    2674573 us |       3645 us
After_matrix_add          |    2674577 us |          4 us
After_matrix_relu         |    2674577 us |          0 us
Batch_0_end               |    2674578 us |          1 us
-------------------------------------------------------



(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$ ./mlp_infer_avx2_max --export-dir ../cifar10_embedding_mlp/export --input ../cifar10_embedding_mlp/export/test_input_all/test_input_embeddings_all.bin --batch-size 2048 --num-batches 1
================================================================================
Input dim    : 512
Num classes  : 10
Batch size   : 2048
Total samples: 2048
Total batches: 1
================================================================================
Processed batch 1/1 in 5567.46 ms
================================================================================
Benchmark Results
================================================================================
Mean latency : 5567.46 ms
Std dev      : 0 ms
Throughput   : 367.852 samples/sec
================================================================================
Wrote logits to: cpp_output_logits.bin
Wrote preds  to: cpp_output_preds.bin

--- Full Inference Profile ---
Step                      |      Elapsed |     Step Dur
-------------------------------------------------------
Batch_0_start             |          0 us |          0 us
Layer_0_start             |       3773 us |       3773 us
Before_matrix_init        |       3775 us |          2 us
After_matrix_init         |      42439 us |      38663 us
After_matrix_mul          |     263670 us |     221231 us
After_matrix_add          |     267670 us |       4000 us
After_matrix_relu         |     270359 us |       2689 us
Layer_1_start             |     270990 us |        630 us
Before_matrix_init        |     270990 us |          0 us
After_matrix_init         |     351267 us |      80276 us
After_matrix_mul          |    2852094 us |    2500826 us
After_matrix_add          |    2854465 us |       2370 us
After_matrix_relu         |    2856542 us |       2077 us
Layer_2_start             |    2860555 us |       4012 us
Before_matrix_init        |    2860556 us |          1 us
After_matrix_init         |    2944065 us |      83508 us
After_matrix_mul          |    5524459 us |    2580394 us
After_matrix_add          |    5527211 us |       2752 us
After_matrix_relu         |    5529763 us |       2551 us
Layer_3_start             |    5535170 us |       5407 us
Before_matrix_init        |    5535172 us |          2 us
After_matrix_init         |    5557982 us |      22809 us
After_matrix_mul          |    5564777 us |       6795 us
After_matrix_add          |    5564786 us |          8 us
After_matrix_relu         |    5564786 us |          0 us
Batch_0_end               |    5567462 us |       2676 us
-------------------------------------------------------



(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE/build$ ./mlp_infer_avx2_max --export-dir ../cifar10_embedding_mlp/export --input ../cifar10_embedding_mlp/export/test_input_all/test_input_embeddings_all.bin --batch-size 4096 --num-batches 1
================================================================================
Input dim    : 512
Num classes  : 10
Batch size   : 4096
Total samples: 4096
Total batches: 1
================================================================================
Processed batch 1/1 in 11303.7 ms
================================================================================
Benchmark Results
================================================================================
Mean latency : 11303.7 ms
Std dev      : 0 ms
Throughput   : 362.361 samples/sec
================================================================================
Wrote logits to: cpp_output_logits.bin
Wrote preds  to: cpp_output_preds.bin

--- Full Inference Profile ---
Step                      |      Elapsed |     Step Dur
-------------------------------------------------------
Batch_0_start             |          0 us |          0 us
Layer_0_start             |       5227 us |       5227 us
Before_matrix_init        |       5229 us |          1 us
After_matrix_init         |      72736 us |      67507 us
After_matrix_mul          |     498761 us |     426024 us
After_matrix_add          |     505200 us |       6438 us
After_matrix_relu         |     509484 us |       4283 us
Layer_1_start             |     510457 us |        973 us
Before_matrix_init        |     510457 us |          0 us
After_matrix_init         |     635816 us |     125358 us
After_matrix_mul          |    5769547 us |    5133730 us
After_matrix_add          |    5775141 us |       5594 us
After_matrix_relu         |    5779663 us |       4521 us
Layer_2_start             |    5787865 us |       8201 us
Before_matrix_init        |    5787866 us |          1 us
After_matrix_init         |    5936082 us |     148215 us
After_matrix_mul          |   11220309 us |    5284227 us
After_matrix_add          |   11226131 us |       5821 us
After_matrix_relu         |   11230715 us |       4583 us
Layer_3_start             |   11239501 us |       8786 us
Before_matrix_init        |   11239503 us |          1 us
After_matrix_init         |   11284772 us |      45269 us
After_matrix_mul          |   11298364 us |      13591 us
After_matrix_add          |   11298384 us |         20 us
After_matrix_relu         |   11298384 us |          0 us
Batch_0_end               |   11303660 us |       5275 us
-------------------------------------------------------




