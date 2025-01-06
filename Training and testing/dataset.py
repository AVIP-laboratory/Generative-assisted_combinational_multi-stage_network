from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms



def transformss(image):
    transform = transforms.Compose([
                        transforms.ToTensor(),
    ])
    data = transform(image)
    return data

# Costum dataset 생성
class add_nosie_data(Dataset):
    def __init__(self, path2img, direction='b2a', transform= transformss):
        super().__init__()
        self.direction = direction
        self.path2a = join(path2img)
        self.img_filenames = [x for x in listdir(self.path2a)]
        self.transform = transform

    def __getitem__(self, index):
        a = Image.open(join(self.path2a, self.img_filenames[index])).convert('RGB')

        if self.transform:
            a = self.transform(a)
        return a

    def __len__(self):
        return len(self.img_filenames)