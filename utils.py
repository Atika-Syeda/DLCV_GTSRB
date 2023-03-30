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

data_rotate_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize([128, 128]),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

data_hflip_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize([128, 128]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

data_vflip_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize([128, 128]),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])

data_jitter_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize([128, 128]),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor()
])

