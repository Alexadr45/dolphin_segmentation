from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class DolphinDataset(Dataset):
    def __init__(self, images_directory, masks_directory, masks_id, transforms=None):
        self.images_directory = images_directory
        self.masks_directory  = masks_directory
        self.masks_id = masks_id
        self.transforms = transforms

    def __len__(self):
        return len(os.listdir(self.masks_directory))

    def __getitem__(self, idx):
        mask_name = os.listdir(self.masks_directory)[idx]
        mask_path = os.path.join(self.masks_directory, mask_name)
        mask  = np.asarray(Image.open(mask_path).convert('L'), dtype=np.float32) / 256
        image_filename = self.masks_id[mask_name]
        img_path = os.path.join(self.images_directory, image_filename)
        image = np.asarray(Image.open(img_path).convert("RGB"))

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask  = transformed["mask"]
                
        return image, mask
