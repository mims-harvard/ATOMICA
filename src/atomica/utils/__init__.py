#!/usr/bin/python
# -*- coding:utf-8 -*-


def pickled_checkpoint_error(path, config_flag, weights_flag):
    """The error raised when someone passes a pickled-model ``.ckpt``.

    Training writes three files per checkpoint: ``<name>.ckpt`` (the pickled
    model object), ``<name>.pt`` (its state dict) and ``config.json``. Only the
    latter two are loadable across package versions, because unpickling a
    ``.ckpt`` rebuilds the model class and e3nn's generated code and so depends
    on the exact versions that wrote it.
    """
    return ValueError(
        f"{path} looks like a pickled model object. ATOMICA loads models from a "
        f"config JSON plus a weights state dict instead, which is portable "
        f"across PyTorch, CUDA and e3nn versions:\n"
        f"    {config_flag} <config.json> {weights_flag} <weights.pt>\n"
        f"Training writes config.json and a .pt next to every .ckpt, and "
        f"https://huggingface.co/ada-f/ATOMICA publishes models in this form."
    )
