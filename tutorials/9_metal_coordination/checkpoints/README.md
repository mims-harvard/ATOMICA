# Checkpoints

This tutorial needs one checkpoint, the released pretrained ATOMICA model, which is the same one
the rest of the repository uses. Download it into the repository-root `checkpoints/`:

```bash
hf download ada-f/ATOMICA --repo-type model --local-dir checkpoints \
  --include "ATOMICA_checkpoints/pretrain/**"
```

Nothing else needs to be uploaded or downloaded. The probe is a logistic regression that fits in
minutes, so no trained head is stored.
