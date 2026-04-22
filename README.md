### Model generation instructions

Setup a conda environment with needed dependencies

```
conda create -n mlp-cifar python=3.11 -y
conda activate mlp-cifar
pip install torch torchvision torchaudio jupyter matplotlib numpy tqdm ipykernel
```

To run the model generation Jupyter notebook
```
python -m ipykernel install --user --name mlp-cifar --display-name "Python (mlp-cifar)"
jupyter notebook
```

This is how `cifar10_embedding_mlp/export/` directory should look like expecpt `test_input_all` and `test_input_small` will not be there. The files that are in them will just exist in `export` directory. I created those two directories manually.

```
(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE$ tree ./cifar10_embedding_mlp/export/
./cifar10_embedding_mlp/export/
├── layer_00_bias.bin
├── layer_00_weight.bin
├── layer_01_bias.bin
├── layer_01_weight.bin
├── layer_02_bias.bin
├── layer_02_weight.bin
├── layer_03_bias.bin
├── layer_03_weight.bin
├── metadata.json
├── test_input_all
│   ├── test_input_embeddings_all.bin
│   ├── test_labels_int64_all.bin
│   ├── test_output_logits_all.bin
│   ├── test_output_pred_int64_all.bin
│   └── test_shapes_all.json
└── test_input_small
    ├── test_input_embeddings.bin
    ├── test_labels_int64.bin
    ├── test_output_logits.bin
    ├── test_output_pred_int64.bin
    └── test_shapes.json

3 directories, 19 files
(base) pshvaiko@ps-tp:~/Documents/HPC_PROJECT_MLP_INFERENCE$
```

To compile:
```
g++ -O3 -std=c++17 -o mlp_infer ./src/main.cpp
```

To run:
```
./mlp_infer cifar10_embedding_mlp/export   cifar10_embedding_mlp/export/test_input_embeddings_all.bin   cpp_output_logits_all.bin   cpp_output_preds_all.bin
```
