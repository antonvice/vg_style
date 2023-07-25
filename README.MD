# Van Gogh Style Transfer
This project fine-tunes a Stable Diffusion model on paintings by Van Gogh and provides a style transfer implementation to turn images into Van Gogh style paintings.

# Usage
To train the model:

```bash

python train.py --data path/to/vangogh/images --steps 1000
This will download the Van Gogh paintings dataset and fine-tune the model.
```
To run style transfer:

```bash

python style_transfer.py --input input.jpg --output output.jpg
```
Pass the input image and where to save the output stylistic image.

# Model
The core model is a Stable Diffusion pipeline pre-trained on Launchpad images. This is then fine-tuned on the Van Gogh paintings to adapt the style.

The training uses a DataCollatorForWholeWordMask and AdamW optimizer with a cosine decay learning rate schedule.

# Installation
The required libraries are:

diffusers
datasets
torch
torchvision
pandas

### Install with:
```bash
pip install -r requirements.txt
```
A GPU is recommended for efficient training and inference.

# References
Kaggle Van Gogh Paintings Dataset
Stable Diffusion
Style Transfer with Stable Diffusion
