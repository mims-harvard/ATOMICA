# Source https://github.com/THUNLP-MT/GET

from typing import Tuple, List, Optional

from data.dataset import Block, VOCAB
from data.tokenizer.tokenize_3d import tokenize_3d


def atom_blocks_to_frag_blocks(
        blocks: List[Block],
        smiles: Optional[str]=None,
        bonds: Optional[List[Tuple[int, int, int]]]=None,
        fragmentation_method: Optional[str]=None
    ) -> List[Block]:

    smis, idxs = tokenize_3d(
        [block.units[0].element for block in blocks],
        [block.units[0].coordinate for block in blocks],
        smiles=smiles, bonds=bonds, fragmentation_method=fragmentation_method)

    new_blocks = []
    VOCAB.load_tokenizer(fragmentation_method)
    for smi, group_idx in zip(smis, idxs):
        atoms = [blocks[i].units[0] for i in group_idx]
        block = Block(
            symbol=VOCAB.abrv_to_symbol(smi),
            units=atoms)
        assert block.symbol != VOCAB.UNK
        new_blocks.append(block)
    return new_blocks