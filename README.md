# Face generation for various tasks

This repository is based on [Stable diffusion](https://github.com/CompVis/stable-diffusion). We are generating real face images fro tasks such as eyes close-open, glasses on-off and hat on-off detection. 


## Requirements
[Mamba](https://mamba.readthedocs.io/en/latest/mamba-installation.html#mamba-install) package manager is recommended over conda. After installation just replace conda in every command with mamba.
A suitable [conda](https://conda.io/) environment named `face-gen` can be created and activated with:

```
conda env create -f environment.yaml
conda activate face-gen
```

## Face generation Script
Tune the parameters in params.py to generate images according to your machine's capacity.

```
python face_gen.py
```

## Images generated
Generated images can be found in face_gen_data folder.

## Reference

```
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```


