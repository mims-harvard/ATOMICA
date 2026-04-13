% 549 words

\xhdr{Cross-modality interface comparison for orthosteric PPI inhibitors}

Orthosteric inhibitors of PPIs include small molecules that bind at a protein-protein interface and competitively block partner binding. 
%
Because orthosteric inhibitors engage the same surface features as the native partner, we compare protein-inhibitor and PPI embeddings in the \name shared latent space and test whether embedding similarity localizes to the inhibited PPI region.

We use the curated 2P2Idb database \cite{basse20122p2idb}, which contains matched PPI structures and their protein-inhibitor PDB structures. After quality filtering, we examined 18 protein-peptide complexes with 268 protein-inhibitor complexes and six protein-protein complexes with 187 protein-inhibitor complexes (Methods~\ref{method:ppi-inhibitors}). Using the \name embedding notation in Fig.~\ref{fig:fig1}c, for each protein-peptide complex and their matched protein-inhibitor ligand complex, we embed distances between peptide and inhibitor blocks $\text{dist}_{\text{ATOMICA}}(\mathbf{h}_\text{peptide}^\text{block}, \mathbf{h}_\text{inhibitor}^\text{block})$ and the spatial distance between the aligned coordinates, $\mathbf{x}$, between the blocks $\text{dist}_{\mathbb{R}^3}(\mathbf{x}_\text{peptide}^\text{block}, \mathbf{x}_\text{inhibitor}^\text{block})$. Across 268 matched pairs, each system contained a median of 60 inhibitor-peptide block pairs (IQR: 40-79). We quantify localization with Fold Change@10: the proportion of inhibitor-peptide block pairs with $\text{dist}_{\mathbb{R}^3} < 4$ \AA{} in the top-10 pairs with the lowest $\text{dist}_{\text{ATOMICA}}$ relative to the reference proportion among all inhibitor-peptide block pairs in the matched pair (additional results for Fold Change values are in Fig.~S4 and Fig.~S5). We observe overrepresentation above reference in 14/18 protein-peptide complexes (78\%), spanning 161/268 matched protein-inhibitor complexes (Fig.~\ref{fig:fig4}d).  Consistently, $\text{dist}_{\text{ATOMICA}}$ correlates with $\text{dist}_{\mathbb{R}^3}$ across matched structures. Nine out of 18 exhibit a significant positive correlation (FDR q $< 0.05$, Table S3), exceeding the number of protein-peptide complexes expected by chance (binomial test p=$6.28\times10^{-8}$).

In Fig.~\ref{fig:fig4}b, we explore the menin and mixed lineage leukemia (MLL) protein-peptide interaction, which plays a central role in acute leukemias \cite{krivtsov2019menin}, and is inhibited by the ligand MIV-7 (PDB ligand code 2SE) \cite{he2014high}.
%
We compare all 144 inhibitor-peptide block pairs and rank them by \name embedding similarity. The 10 most similar pairs are closer in the aligned 3D structures than the 10 least similar pairs, indicating that blocks with similar embeddings tend to localize to corresponding regions of the native interface.
%Comparing the top-10 most similar inhibitor-peptide block pairs to the bottom-10 pairs in a total of 144 inhibitor-peptide block pairs, we observe that highly similar embedding pairs are closer in aligned 3D space, showing that inhibitor and peptide blocks with similar \name embeddings tend to occupy spatially corresponding regions of the native interface.

For protein-protein (A-B) complexes, we compare the inhibitor interface embedding $\mathbf{h}_{\text{inhibitor}}^\text{interface}$ to embeddings $\mathbf{h}_{\text{protein B}}^{\text{interface}, i}$ of 1,000 local surface patches sampled on protein B. We rank these patches by $\text{dist}_\text{ATOMICA}(\mathbf{h}_{\text{inhibitor}}^\text{interface}, \mathbf{h}_{\text{protein B}}^{\text{interface}, i})$ and test whether the most similar patches are located near the native A-B binding site on protein B.
%
%For protein-protein (A-B) complexes, we compare the inhibitor interface embedding $\mathbf{h}_{\text{inhibitor}}^\text{interface}$ to \name embeddings of a sampled interface $i$ on protein B, $\mathbf{h}_{\text{protein B}}^{\text{interface}\, i}$. In total, we sample 1,000 interfaces on protein B, and retrieve the 10 interfaces with the lowest $\text{dist}_\text{ATOMICA}(\mathbf{h}_{\text{inhibitor}}^\text{interface}, \mathbf{h}_{\text{protein B}}^{\text{interface}\, i})$.
%
%We evaluate retrieval with Fold Change@10 using positives defined by proximity of the interface on B to the native A-B binding site with a threshold of $\text{dist}_{\mathbb{R}^3}<12$~\AA{} (threshold set to the 25th percentile of sampled distances). This is compared against the proportion of uniformly sampled interfaces within 12 \AA{} to the interface.
We evaluate retrieval using Fold Change@10, where positives are sampled surface patches on protein B whose centers lie within 12~\AA{} of the native A-B binding site on B. We set $\text{dist}_{\mathbb{R}^3}<12$~\AA{} threshold to the 25th percentile of sampled distances and compare the fraction of positives among the top 10 retrieved patches to the fraction among all sampled patches.
%
% A fold change of greater than 1 is observed in all six protein-protein complexes, covering 186/187 matched protein-inhibitor complexes (Fig.~\ref{fig:fig4}f), and significant positive correlations between similarity and distance to the native site in 5/6 complexes after correction (FDR q $<0.05$; Table S4). These results show that $\text{dist}_\text{ATOMICA}$ is associated with spatial proximity to the native protein-protein binding site across distinct protein-protein interactions.
Fold Change@10 is greater than 1 in all six protein-protein complexes, covering 186 of 187 matched protein-inhibitor complexes (Fig.~\ref{fig:fig4}f). In addition, after multiple-testing correction, 5 of 6 complexes show a significant positive correlation between embedding similarity and proximity to the native site (FDR q $< 0.05$; Table S4). These results indicate that lower $\text{dist}_\text{ATOMICA}$ is associated with closer proximity to the native protein-protein binding site across distinct protein-protein interactions.
 
%For the KRAS-SOS1 (Fig.~\ref{fig:fig4}g,h) and HRAS-SOS1 (Fig.~\ref{fig:fig4}i,j) complexes, we examine the top 10 and bottom 10 sampled SOS1 interfaces (out of 1{,}000) ranked by similarity to the inhibitor embedding. For a covalent inhibitor of KRAS \cite{zhang2022chemoselective} and a small-molecule inhibitor of HRAS \cite{kessler2020drugging}, the most and least similar retrieved interfaces (top 1 and bottom 1) illustrates their spatial proximity to the native RAS-SOS1 interface, highlighting the localization of high-similarity regions of the inhibitor embedding to the PPI interface.

For the KRAS-SOS1 (Fig.~\ref{fig:fig4}g,h) and HRAS-SOS1 (Fig.~\ref{fig:fig4}i,j) complexes, we rank 1{,}000 sampled surface patches on SOS1 by their similarity to the inhibitor embedding and compare the top 10 and bottom 10 patches. For a covalent KRAS inhibitor \cite{zhang2022chemoselective} and a small-molecule HRAS inhibitor \cite{kessler2020drugging}, the top-ranked patch is closer to the native RAS-SOS1 interface, whereas the bottom-ranked patch is farther away. These examples show that inhibitor embedding similarity is highest near the native protein-protein interface.


\section{Cross-modality comparison for orthosteric PPI inhibitors}\label{method:ppi-inhibitors}

\subsection{Dataset}
We utilized the 2P2IDB database, which contains experimentally determined structures of protein-protein interactions and their inhibitors from the Protein Data Bank (PDB). For PPI structures, they were processed to keep only the target (A) and partner (B) chain. For protein--inhibitor structures, they were processed to keep only the target (A) and inhibitor (I) ligand.


\subsection{Protein-peptide inhibitor analyses}
For protein--peptide complexes where peptide B contains $\leq 30$ residues, we performed block-level comparisons between inhibitor-bound structures and peptide B to assess whether \name embeddings capture structural similarities at the interface.

\subsubsection{Structure alignment}
We aligned inhibitor target chain structures to PPI target chain structures. Initial sequence alignment using BLOSUM62 substitution matrix with gap opening penalty of $-10$ and gap extension penalty of $-1$. Superposition of matched C$\alpha$ atom coordinates using Kabsch algorithm. Iterative outlier rejection was then applied and residue pairs with RMSD $>$ 2.0 \AA\ were removed. The alignment was refined over 5 cycles until convergence.

\subsubsection{\name embedding computation}
We computed \name embeddings for both inhibitors and peptide B using the pretrained \name model. For the inhibitor embedding, we embedded the inhibitor bound to the target protein pocket and extracted the block embeddings of the inhibitor, $\mathbf{h}_\text{inhibitor}^\text{block}$. For the protein--peptide embedding, we embedded the peptide bound to the target protein pocket and extracted the block embeddings of the peptide, $\mathbf{h}_\text{peptide}^\text{block}$.

\subsubsection{Block pairwise comparison}
For each inhibitor-PPI pair, we performed the following block-level comparison. Applied the rotation matrix $R$ and translation vector $t$ from the alignment step to transform inhibitor block coordinates into the PPI reference frame. Block coordinates, $\mathbf{x}$, are the average of the atom coordinates within each block. We filtered blocks to remove singleton blocks (size = 1) and global-type blocks, retaining only blocks with $>$1 atoms. Pairwise embedding distances, $\text{dist}_{\text{ATOMICA}}(\mathbf{h}_\text{peptide}^\text{block}, \mathbf{h}_\text{inhibitor}^\text{block})$, between all inhibitor blocks and peptide B blocks were calculated as the cosine distance between block embedding vectors.
%
Pairwise spatial distance was given by the Euclidean distance between block center coordinates which we define as $\text{dist}_{\mathbb{R}^3}(\mathbf{x}_\text{peptide}^\text{block},\mathbf{x}_\text{inhibitor}^\text{block})$ (\AA). We filter the pairs of inhibitor--PPI pairs to only keep those with at least 10 pairwise blocks for comparison, this results in 18 out of 31 protein-peptide complexes being suitable leaving 268 out of 1848 matched inhibitor structures. The final number of inhibitor structures matched to each PPI complex is available in Table S1.

\subsubsection{Statistical analyses}
\xhdr{Spearman's rank correlation}
For each protein-peptide complex and their matched protein-inhibitor complexes, we computed the Spearman's rank correlation coefficient between $\text{dist}_{\text{ATOMICA}}$ and the spatial distance $\text{dist}_{\mathbb{R}^3}$ for all inhibitor--peptide blocks for all inhibitors that match the protein-peptide complex. We tested for statistical significance using a two-tailed test ($p < 0.05$). Multiple testing correction was performed using the Benjamini-Hochberg false discovery rate (FDR) procedure with $q < 0.05$.

\xhdr{Fold Change analysis}
We evaluated whether embedding-based retrieval could identify spatially proximal blocks. For each inhibitor block, we ranked peptide B blocks by embedding distance, $\text{dist}_{\text{ATOMICA}}$, and selected the top-$k$ = 10 lowest distances as retrieved inhibitor-peptide block pairs. Inhibitor-peptide block pairs with $\text{dist}_{\mathbb{R}^3}$ within a geometric threshold of 4.0 \AA\ were considered spatially close. We computed \textit{Precision@10} as the fraction of top-10 retrieved blocks within 4.0 \AA\ and \textit{Fold Change@10}, which is Precision@10 divided by the baseline rate (overall fraction of blocks within the threshold  $\text{dist}_{\mathbb{R}^3} < 4$ \AA{}).

\subsection{Protein-protein inhibitor analyses}
For protein--protein complexes where protein B has $>$30 residues, we employed a surface sampling approach to identify interface regions and assess whether ATOMICA embeddings can distinguish interface patches that are geometrically similar to inhibitor binding sites. We keep PPI complexes that have an A--B binary binding structure, as a result we discard INTEGRASE/LEDGF and TNFA trimers from our analyses. A total of 187 inhibitor structures across six PPIs are evaluated. The final number of inhibitor structures per PPI complex is available in Table S2.

\subsubsection{Surface point sampling}
We generated molecular surface representations and sampled points uniformly across the surface of each protein B. We use the MSMS tool \cite{sanner1996reduced} to compute solvent-accessible surfaces with the parameters: vertex density = 3.0 vertices per \AA$^2$, probe radius = 1.5 \AA. For each protein B chain, we extracted atomic coordinates and generated a triangular mesh with vertices, faces, and surface normals. We then sampled 1,000 points per protein B surface using area-weighted triangle sampling. For each sampled triangle, we uniformly sampled a point using barycentric coordinates. This was achieved with generated random values $r_1, r_2 \sim U(0,1)$, computed barycentric coordinates: $u = 1 - \sqrt{r_1}$, $v = \sqrt{r_1}(1 - r_2)$, $w = \sqrt{r_1} r_2$, and point position: $\mathbf{p} = u\mathbf{v}_0 + v\mathbf{v}_1 + w\mathbf{v}_2$ where $\mathbf{v}_0, \mathbf{v}_1, \mathbf{v}_2$ are triangle vertices.


\subsubsection{Interface patch definition}
For each sampled surface point, we defined a local interface patch based on spatial proximity to protein blocks. We computed the Euclidean distance from the surface point to all protein B block centers. Selected blocks were within a radius of 16.0 \AA\ from the surface point, and we discarded points with $<$8 nearby blocks. For each interface patch center on protein B, we computed the distance to the nearest C$\alpha$ atom on protein A.

\subsubsection{\name embedding computation}

\xhdr{Inhibitor embeddings}
For each protein--inhibitor complex ($A$--$I$), we apply \name to the inhibitor bound to the target protein pocket to obtain contextualized inhibitor interface embeddings, $\mathbf{h}_\text{inhibitor}^\text{interface}$.

\xhdr{Interface patch embeddings}
For each protein--protein ($A$--$B$) complex, we sample local surface interface patches on the protein $B$. Each patch is defined as the set of blocks within a 16~\AA{} radius of a surface point on $B$. We apply \name to each sampled patch to obtain an interface embedding, $\mathbf{h}_\text{protein B}^\text{interface}$, that captures the local structural context. These patch embeddings serve as retrieval candidates.

\subsubsection{Retrieval analysis}

We formulate retrieval as follows: given an inhibitor embedding (query), we rank sampled interface patches on protein $B$ (candidates) by embedding similarity and evaluate whether high-similarity patches localize to the native $A$--$B$ binding site.

To limit information leakage from the target interface, embeddings from protein $A$ are not used in forming retrieval candidates or computing similarity scores. Specifically, (i) the inhibitor embedding is constructed solely from ligand blocks at the interface, and (ii) candidate embeddings are computed exclusively from patches on protein $B$. Thus, retrieval compares a ligand-only query to protein-only interface patches, avoiding trivial localization through direct encoding of the target interface geometry.

We compute the following distance metrics: embedding distance  $\text{dist}_\text{ATOMICA} (\mathbf{h}^\text{interface}_\text{inhibitor}, \mathbf{h}^\text{interface}_\text{protein B})$, given by the cosine distance between inhibitor embedding and interface patch embedding, and spatial distance $ \text{dist}_{\mathbb{R}^3}(\mathbf{x}^\text{interface}_\text{inhibitor}, \mathbf{x}^\text{interface}_\text{protein B})$, given by the Euclidean distance from the interface patch center to the nearest C$\alpha$ atom on protein A.

\xhdr{Retrieval procedure} For each inhibitor, we ranked all interface patches by $\text{dist}_\text{ATOMICA}$ (ascending). The top-$k$ = 10 (lowest $\text{dist}_\text{ATOMICA}$) retrieved patches were selected. A geometric threshold of patches within 12.0 \AA\ of the interface protein A was considered ``close''. A threshold of 12.0 \AA\ was selected as it represents the 25th percentile of distances of sampled patches from the PPI interface. We computed \textit{Precision@10}: the fraction of top-10 patches within the geometric threshold and \textit{Enrichment@10}: Precision@10 divided by the baseline rate. The baseline rate is given by the proportion of all patches on protein B that are within 12.0 \AA{} from the A--B interface.

\subsubsection{Statistical analyses}
\xhdr{Spearman Correlation} For each PPI family, we aggregated all inhibitor-patch pairs and computed the Spearman rank correlation coefficient between embedding distances $\text{dist}_\text{ATOMICA}$ and geometric distances $\text{dist}_{\mathbb{R}^3}$. Statistical significance was assessed using a two-tailed test ($p < 0.05$), with FDR correction (Benjamini-Hochberg, $q < 0.05$) applied across families.