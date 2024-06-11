from PIL import Image
import torch
import torchvision.transforms as transforms

import utils

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.4


class VisionModel:
    def __init__(self):
        self.model = torch.load("model.bin", map_location=torch.device('cpu')).to(DEVICE)
        self.model.eval()
        self.skin_disease_labels = utils.get_skin_disease_labels()

    @staticmethod
    def load_image(image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(Image.open(image_path)).unsqueeze(0).to(DEVICE)
        return image

    def predict(self, image_path):
        image = self.load_image(image_path)
        with torch.no_grad():
            model_output = torch.sigmoid(self.model(image)).cpu().numpy()
            binary_output = (model_output > THRESHOLD).astype(int).squeeze()

            labels = []
            for index in range(len(binary_output)):
                if binary_output[index] == 1:
                    labels.append(self.skin_disease_labels[index])
            return labels


if __name__ == "__main__":
    img_path = "images/eczema.png"
    vision_model = VisionModel()
    skin_disease_labels = vision_model.predict(img_path)
    print(skin_disease_labels)
