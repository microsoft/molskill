# MolSkill 🤹⌬

[![ci](https://github.com/microsoft/molskill/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/microsoft/molskill/actions/workflows/ci.yml)
[![Anaconda-Server Badge](https://anaconda.org/msr-ai4science/molskill/badges/platforms.svg)](https://anaconda.org/msr-ai4science/molskill)
[![Anaconda-Server Badge](https://anaconda.org/msr-ai4science/molskill/badges/version.svg)](https://anaconda.org/msr-ai4science/molskill)

This repo contains associated code for the paper _Extracting medicinal chemistry intuition via preference machine learning_ as available on [Nature Communications](https://www.nature.com/articles/s41467-023-42242-1).

## Installation

We recommend that you make a fresh conda environment (currently we only support Python 3.9-3.10 Linux builds) and install the provided conda package for convenience:

```bash
conda install molskill=*=py3{x}* -c msr-ai4science -c conda-forge
```

Please substitute `{x}` above by either `9` or `10` depending on your python version. Additionally, you can also use the provided `environment.yml` file for manual installation.

A CUDA-enabled GPU is not required for usage, but strongly recommended for speed if you plan on scoring a large amount of compounds. 


## Usage

This work mainly exposes the `MolSkillScorer` class under the `molskill.scorer` module. We interface with RDKit to provide predictions accordingly. The user only has to provide a list of molecular strings that they wish to score. 

```python
from molskill.scorer import MolSkillScorer

smiles_strs = ["CCO", "O=C(Oc1ccccc1C(=O)O)C"] 

scorer = MolSkillScorer()
scores = scorer.score(smiles_strs)
```

We provide and use by default a pre-trained model on all the data that was collected during the original study. If a user wants to train custom models, please check the `train.py` script also included under this repository.

**Note**: The default model and featurizer does not support non-organic elements or molecules with multiple fragments. Furthermore we suggest that you pass the NIBR filters on your compounds before running them through default scorer. We recommend doing this to avoid out-of-distribution biases, as the provided models have never seen a violating molecule during training time. The filters are nowadays available on the RDKit - a guide on how to apply those is provided [here](https://github.com/rdkit/rdkit/tree/master/Contrib/NIBRSubstructureFilters).


## Citing

If you find this work or parts thereof useful, please consider citing the following BibTeX entry:

```
 @article{choung2023,
          place={Cambridge},
          title={Learning chemical intuition from humans in the loop},
          DOI={10.26434/chemrxiv-2023-knwnv},
          journal={ChemRxiv},
          publisher={Cambridge Open Engage},
          author={Choung, Oh-Hyeon and Vianello, Riccardo and Segler, Marwin and Stiefl, Nikolaus and Jiménez-Luna, José},
          year={2023}}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.Any use of third-party trademarks or logos are subject to those third-party's policies.
