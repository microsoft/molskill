This directory contains several datasets used throughout the study. In summary:

* `production_public.csv` contains information in regards to the molecule pairs presented to the participants of the study during the production runs (a bit over 5000 pairs). The binary label indicates whether the `smiles_j` compound was chosen (1) or not (0).
* `pre_r{1, 2}.csv` contains responses for the first and second preliminary rounds of the study. Ratings from different participants are labeled on different columns.
* `other_dbs/*.csv` contain data in regards to the analyses run for Figure 5 in the study. Specifically, NIBR-filtered compounds for ChEMBL, the FDA-approved DrugBank, and the GDB extracted sets are present here.
* `assets/chembl_population_mean_std.csv` contains population level statistics that are used during default model training/evaluation for normalization of the descriptors used. 