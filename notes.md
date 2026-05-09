

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
