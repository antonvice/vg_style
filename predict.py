from cog import BasePredictor, Input, Path
from PIL import Image
import torch
import torchvision.transforms as T

from style import VanGoghStyleTransfer
# Preprocess 

def preprocess(image_in):
  input_img = Image.open(image_in)
  
  transform = T.Compose([
     T.Resize(256),
     T.ToTensor(),
     T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
  ])
  
  input_tensor = transform(input_img).unsqueeze(0)
  return input_tensor

# Postprocess

def postprocess(output_tensor):
  output_img = (output_tensor + 1) / 2
  output_img = T.ToPILImage()(output_img[0])
  return output_img

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = torch.load("./weights.pth")

    # The arguments and types the model takes as input
    def predict(self,
          image: Path = Input(description="Grayscale input image")
    ) -> Path:
        """Run a single prediction on the model"""
        processed_image = preprocess(image)
        output = self.model(processed_image)
        return postprocess(output)