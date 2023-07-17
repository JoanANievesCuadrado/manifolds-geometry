# manifolds-geometry

This is the official repository associated to the paper [**The geometry of normal tissue and cancer
manifolds**](https://doi.org/10.1101/2021.08.20.457160)

We show that both the normal states and their respective tumors span disjoint manifolds in gene expression space. We compute a set of geometric properties of these manifolds.

Two examples of gene expression data (ESCA and LIHC) from TCGA are contained in .../databases_external. They are to be processed with preprocessing_tcga_data.py The results are mean value vectors for tissues, contained in .../databases_generated.

On the other hand, protein expression data from HPA are to be processed with preprocessing_hpa_data.py and the results are also written in .../databases_generated.

The scripts to generate figures and tables are provided in the root repository. They use the results of .../databases_generated. For consistency, the figures and tables themselves are reproduced in .../Figures and Tables.


