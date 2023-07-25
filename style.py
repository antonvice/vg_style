import diffusers
from diffusers import StableDiffusionPipeline
from datasets import load_dataset
import torch
from torchvision import transforms
from PIL import Image
from data_prep import DLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset
import os
from transformers import DataCollatorForWholeWordMask


# Load the Images
# dataset = DLoader(csv_file)
# dataset._filter()
# image_paths = dataset._get_imgs(img_folder)

class ImageDataset(Dataset):
    def __init__(self, img_folder, transform=None):
        self.img_folder = img_folder
        self.image_paths = [os.path.join(img_folder, filename) for filename in os.listdir(img_folder)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        # Resize the image to a smaller size (e.g., 512x512) to avoid the DecompressionBombWarning
        resize_transform = transforms.Resize((512, 512))
        image = resize_transform(image)

        if self.transform:
            image = self.transform(image)

        return image

class VanGoghStyleTransfer:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.model_id = model_id
        self.pipe = None

    def _preprocess_images(self, images):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])

        images = [image for image in images]
        return torch.stack(images)

    def train(self, img_folder, num_train_steps=100, batch_size=4):
        # Load model
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id)

        # Load the dataset using ImageDataset
        dataset = ImageDataset(img_folder, transform=transforms.ToTensor())

        # Preprocess the dataset
        images = [image for image in dataset]
        images = self._preprocess_images(images)
        
        # Create dataset collator
        collate_fn = DataCollatorForWholeWordMask(
            tokenizer=self.pipe.tokenizer,
            mlm=False 
        )

        # Fine-tune model
        train_dataloader = torch.utils.data.DataLoader(images, batch_size=batch_size, collate_fn=collate_fn)
        optimizer = torch.optim.AdamW(self.pipe.adam_params(), lr=5e-6)

        # Cosine Annealing Warm Restarts scheduler
        T_0 = 10  # Initial restarts period (can be adjusted)
        T_mult = 2  # Factor by which the restart period increases (can be adjusted)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)

        self.pipe.train()

        for step, batch in enumerate(train_dataloader):
            print(f"Training step {step}")
            loss = self.pipe(**batch).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Save fine-tuned model and tokenizer
        self.pipe.save_pretrained("model/vangogh_model")
        self.pipe.tokenizer.save_pretrained("model/vangogh_model")

    def style_transfer(self, input_image_path, output_image_path):
        self.load_model()  # Ensure the model is loaded

        input_image = Image.open(input_image_path)
        resize_transform = transforms.Resize((512, 512))
        input_image_resized = resize_transform(input_image)
        input_tensor = self._preprocess_images([input_image_resized])[0].unsqueeze(0)

        self.pipe.eval()
        with torch.no_grad():
            generated_image = self.pipe(input_tensor).generated_images[0]

        generated_image = (generated_image + 1.0) / 2.0  # De-normalize to [0, 1] range
        generated_image = transforms.ToPILImage()(generated_image)
        generated_image.save(output_image_path)
        return generated_image  # Return the generated image for display


if __name__ == "__main__":
    van_gogh_st = VanGoghStyleTransfer()
    van_gogh_st.train('imgs', num_train_steps=100, batch_size=4)
    van_gogh_st.style_transfer('input_image.jpeg', 'output_image.jpg')