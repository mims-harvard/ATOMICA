# Checkpoint for MASIF-Ligand benchmark

`masif_excluded_pretrain.pt` is an ATOMICA pretrained model, not the general released one. It was
pretrained on a corpus with MaSIF-similar structures removed, so the encoder has not seen any
protein resembling a MaSIF-ligand test pocket.

A protein-ligand pretraining item was removed if any of its receptor chains was a MaSIF-ligand test
receptor chain, shared an MMseqs2 cluster with one at 30% sequence identity and 80% coverage, or
reached a Foldseek TM-score of at least 0.5 to one. That removed 29,094 of the 105,090
protein-ligand complexes in the training split. The validation split was filtered by the same rule,
2,282 of 7,273 items, so checkpoint selection never saw a MaSIF-similar structure either. Other
interaction types were unchanged. A separate pretraining run was then trained on the filtered
corpus with the same architecture, objective, hyperparameters and noise schedule as the full model,
and the lowest-validation-loss checkpoint was selected.

| file | contents |
|---|---|
| `config.json` | model architecture, 4 layers, PS_300 fragmentation |
| `masif_excluded_pretrain.pt` | weights, 8.4M parameters |
