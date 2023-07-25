from cog import BasePredictor, Input, Path
import torch
from style import VanGoghStyleTransfer
class Predictor(cog.Predictor):
    def setup(self):
        self.model = VanGoghStyleTransfer()
        self.model.train('vgdb_2016.csv', 'imgs', num_train_steps=100, batch_size=4)

    @cog.input("input_image", type=cog.Path)
    @cog.output("output_image", type=cog.Path)
    def predict(self, input_image, output_image):
        self.model.style_transfer(input_image, output_image)
        return output_image
    
