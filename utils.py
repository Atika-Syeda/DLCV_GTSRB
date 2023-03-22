import PIL.Image as Image
from torchvision import transforms

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize([128, 128]),
    transforms.ToTensor()
])

