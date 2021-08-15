import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

class JPGs(Dataset):
    def __init__(self, data, directory, transform=None):
        self.data = data
        self.directory = directory
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = os.path.join(self.directory, self.data[idx])
        image = cv2.imread(path, cv2.COLOR_BGR2RGB)
        return image

def main(path):
    imlist=[f for f in os.listdir(path) if f.endswith('_rgb.jpg')]
    jpg_dataset = JPGs(data=imlist, directory=path, transform=None)
    im_loader = DataLoader(jpg_dataset)

    sum = torch.tensor([0.0, 0.0, 0.0])
    sum_sq = torch.tensor([0.0, 0.0, 0.0])

    for images in im_loader:
        sum += images.sum(axis = [0, 1, 2])
        sum_sq += (images ** 2).sum(axis = [0, 1, 2])

    val_counter=jpg_dataset.__len__()*jpg_dataset.__getitem__(0).shape[0]*jpg_dataset.__getitem__(0).shape[1]
    mean=sum/val_counter
    std=torch.abs(torch.sqrt(torch.abs((sum_sq/val_counter)-(mean**2)))-mean)

    print("Mean: ",mean," std: ", str(std))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('No RGB Image-directory specified')
        exit(-1)
    path = sys.argv[1]
    main(path)


