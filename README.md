# Face generation for various tasks

This repository is based on [Stable diffusion](https://github.com/CompVis/stable-diffusion). We are generating real face images fro tasks such as eyes close-open, glasses on-off and hat on-off detection. 

  
## Requirements
A suitable [conda](https://conda.io/) environment named `face-gen` can be created and activated with:

```
conda env create -f environment.yaml
conda activate face-gen
```

You can also update an existing [latent diffusion](https://github.com/CompVis/latent-diffusion) environment by running

```
conda install pytorch torchvision -c pytorch
pip install transformers==4.19.2 diffusers invisible-watermark
pip install -e .
``` 
PS: This environment file is incomplete at the moment. Please follow the instructions above and then install the additional libraries yourself. Update to the environment file coming soon.

## Face generation Script
Tune the parameters in params.py to generate images according to your machine's capacity.

```
python demo.py
```

## Images generated
Generated images can be found in specs_data folder.

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


