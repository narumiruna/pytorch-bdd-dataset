import glob
import os

from torch.utils import data
from torchvision.datasets.folder import pil_loader

from utils import load_json


class BDDDataset(data.Dataset):

    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.samples = None

        self.prepare()

    def prepare(self):
        self.samples = []

        if self.train:
            label_paths = glob.glob(
                os.path.join(self.root, 'labels/train/*.json'))
            image_dir = os.path.join(self.root, 'images/100k/train')
        else:
            label_paths = glob.glob(
                os.path.join(self.root, 'labels/val/*.json'))
            image_dir = os.path.join(self.root, 'images/100k/val')

        for label_path in label_paths:
            image_path = os.path.join(
                image_dir,
                os.path.basename(label_path).replace('.json', '.jpg'))

            if os.path.exists(image_path):
                self.samples.append((image_path, label_path))
            else:
                raise FileNotFoundError

    def __getitem__(self, index):
        # TODO: handle label dict

        image_path, label_path = self.samples[index]

        image = pil_loader(image_path)
        label = load_json(label_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.samples)


def main():
    from torchvision import transforms
    transform = transforms.Compose(
        [transforms.Resize(64), transforms.ToTensor()])
    loader = data.DataLoader(
        BDDDataset('data/bdd100k', transform=transform),
        batch_size=2,
        shuffle=True)

    for i, (x, y) in enumerate(loader):
        print(x.size())
        print(y)
        break


if __name__ == '__main__':
    main()
