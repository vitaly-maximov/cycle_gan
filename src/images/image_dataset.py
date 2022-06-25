import torchvision.transforms as tt
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, files, batch_size=1, device='cpu'):
        super().__init__()
        
        prepair = tt.Compose([
            tt.Resize(150),
            tt.CenterCrop(150)])
        
        self._images = []
        for file in files:
            image = Image.open(file)
            image.load()
            self._images.append(prepair(image))
        
        self._transform = tt.Compose([
            tt.RandomCrop(128),
            tt.RandomHorizontalFlip(p=0.5),
            tt.ToTensor(),
            tt.Normalize(0.5, 0.5)])
        
        self._device = device
        self._loader = DataLoader(self, batch_size=batch_size, shuffle=True)
    
    @staticmethod
    def to_test_tensor(file):
        image = Image.open(file)
        image.load()

        transform = tt.Compose([
            tt.Resize(128),
            tt.CenterCrop(128),
            tt.ToTensor(),
            tt.Normalize(0.5, 0.5)
        ])

        return transform(image).unsqueeze(0)

    def loader(self):
        return self._loader
    
    def __len__(self):
        return len(self._images)
    
    def __getitem__(self, index):
        x = self._images[index]
        x = self._transform(x)
        
        # 8-bit
        if (x.shape[0] == 1):
            x = x.repeat(3, 1, 1)
        
        return x.to(self._device)