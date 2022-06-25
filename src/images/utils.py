import torch
import torchvision.utils

def get_sample(x_loader, y_loader):
    def get_8_images(loader):
        xs = []
        for x in loader:
            xs.append(x)
            if (x.shape[0] * len(xs) >= 8):
                break
        return torch.cat(xs)[:8]

    x = get_8_images(x_loader)
    y = get_8_images(y_loader)

    return x, y

def load_sample(path, device='cpu'):
    images = torch.load(path)
    
    x = images['x'].to(device)
    y = images['y'].to(device)

    return x, y

def save_sample(path, x, y):
    images = {
        'x': x.to('cpu'),
        'y': y.to('cpu')
    }    
    torch.save(images, path)

def normalize(tensor):
    return tensor * 0.5 + 0.5

def save_image(path, image):
    torchvision.utils.save_image(normalize(image), path)

def save_image_grid(path, x, y):
    torchvision.utils.save_image(
        torchvision.utils.make_grid(normalize(torch.cat((x, y))), nrow=4), 
        path)