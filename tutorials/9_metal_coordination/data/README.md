# Data

Two files, 29 MB in total.

| file | rows | what it is |
|---|---|---|
| `metal_pockets.parquet` | 26,234 | the ATOMICA interaction graphs for the metal sites the probe uses |
| `metal_coordination_labels.parquet` | 71,967 | one row per metal site in the protein-ion splits, with its MetalPDB match |

## `metal_pockets.parquet`

A subset of ATOMICA's pretraining protein-ion splits, which are released in full at
[Harvard Dataverse](https://doi.org/10.7910/DVN/4DUBJX). It holds the sites with a verified
MetalPDB label, grouped by PDB entry into 20,159 train, 2,421 validation and 3,654 test. Columns
are the standard ATOMICA graph columns plus `id` and `split`, so
`atomica.data.dataset.PDBDataset` reads it directly.

Each pocket has two segments: segment 0 holds the protein residues, segment 1 holds a global node
and exactly one metal block. **The pockets contain amino acids only**, no waters and no cofactors,
which is why the coordination number has two versions.

## `metal_coordination_labels.parquet`

| column | meaning |
|---|---|
| `id` | joins to `metal_pockets.parquet` |
| `split`, `probe_split` | the source split, and which probe split the site lands in |
| `pdb_code` | the PDB entry; splits and bootstrap resamples are grouped on this |
| `element` | the metal |
| `status` | how the MetalPDB match resolved |
| `cn_full`, `cn_protein`, `cn_missing` | coordination number over all deposited donors, over the protein donors in the model input, and the difference |
| `n_donors_water` | water donors in the deposited sphere |
| `geometry_raw`, `geometry_base`, `geometry_vacancy` | the FindGeo assignment and its vacancy qualifier |

Class labels are derived from these raw fields by `metal_tasks.derive_label_columns`, so the class
vocabulary lives in readable code rather than being frozen into the file.

### Coverage

A metal site id names a BioLiP assembly and a ligand chain, not a metal atom, and nothing links it
to a MetalPDB record directly. Chain letters do not survive assembly generation and neither do
absolute coordinates, so sites were matched on a distance fingerprint: for every donor MetalPDB
lists, the pocket must contain that residue by name and number with an atom at the reported
metal-donor distance to within 0.05 A, and all of a candidate's donors must be found. Residue
numbers and interatomic distances are both invariant to assembly generation. The matching was
checked against MetalPDB's own donor-to-donor distances, a signal the matcher never uses, and
agrees for 99.1% of matched sites.

Of the 7,200 held-out test sites:

| `status` | sites | used |
|---|---|---|
| `matched`, a uniquely identified metal | 3,654 | yes |
| `entry_not_in_metalpdb` | 2,388 | no |
| `ambiguous_agree`, several symmetry copies fit and all agreed on the label | 489 | no |
| `no_full_match` | 330 | no |
| `ambiguous_conflict` | 262 | no |
| `no_endogenous_donors` | 77 | no |

So 3,654 test sites over 1,562 PDB entries carry a verified coordination number, and 2,313 of
those over 1,162 entries also carry a FindGeo polyhedron; the other 1,341 are ones FindGeo left
unassigned. Quoting 3,654 for the geometry task would overstate it by 58%.

## Citing

- MetalPDB, the source of the coordination numbers and the geometry assignments: Putignano,
  Rosato, Banci, Andreini, *Nucleic Acids Research* 46(D1):D459-D464, 2018.
  <https://metalpdb.cerm.unifi.it/>. What is here is a derived annotation table keyed by ATOMICA
  ids, not a redistribution of MetalPDB; check its terms of use before reusing the labels.
- FindGeo, which produced the polyhedron assignments MetalPDB distributes: Andreini, Cavallaro,
  Lorenzini, *Bioinformatics* 28(12):1658-1660, 2012.
