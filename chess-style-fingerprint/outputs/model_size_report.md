# Model Vector Sizes and Parameter Counts

## Model 1 (Neural Hybrid)

| model   | preset | input_tensor      | move_encoder_channels | residual_blocks | move_vector_dim | transformer_layers | transformer_heads | transformer_width | board_embedding_dim | feature_vector_dim | feature_embedding_dim | fusion_vector_dim | logit_dim | trainable_params |
| ------- | ------ | ----------------- | --------------------- | --------------- | --------------- | ------------------ | ----------------- | ----------------- | ------------------- | ------------------ | --------------------- | ----------------- | --------- | ---------------- |
| Model 1 | tiny   | (B, 80, 34, 8, 8) | 16                    | 1               | 64              | 1                  | 2                 | 128               | 128                 | 33                 | 32                    | 160               | 2         | 181624           |
| Model 1 | small  | (B, 80, 34, 8, 8) | 32                    | 2               | 128             | 2                  | 4                 | 256               | 256                 | 33                 | 64                    | 320               | 2         | 1254350          |
| Model 1 | paper  | (B, 80, 34, 8, 8) | 64                    | 6               | 320             | 12                 | 8                 | 1024              | 512                 | 33                 | 64                    | 576               | 2         | 102302134        |

## Model 2 (Classical)

| model   | classifier    | input_vector_dim | output_dim | parameter_count | count_note                                                                              |
| ------- | ------------- | ---------------- | ---------- | --------------- | --------------------------------------------------------------------------------------- |
| Model 2 | Linear SVM    | 33               | 2          | 34              | Binary pairwise case: one linear separator has d weights + 1 bias.                      |
| Model 2 | RBF SVM       | 33               | 2          | data-dependent  | Depends on number of support vectors (not fixed before fitting).                        |
| Model 2 | Random Forest | 33               | 2          | data-dependent  | Uses 300 trees, max_depth=4 in this project; exact node counts depend on fitted splits. |

### Presentation shorthand

- Model 1 (small preset): 34 -> 128 -> 256 + 64 -> 320 -> 2 logits, 1,254,350 trainable params.
- Model 2 (pairwise): 33 -> 2 scores/probabilities; Linear SVM has 34 learned parameters in binary case.