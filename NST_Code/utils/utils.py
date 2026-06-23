from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

# ye dataset banae k liye use kr rahe hai
class ImageFolderDataset(Dataset):
    def __init__(self, root, transform = None):
        super(ImageFolderDataset, self).__init__()
        self.root = root
        self.transform = transform  # jo bhi root file unko transform kr denge
        self.files = list(os.listdir(root))  # root file ko read kr rahe hai
        self.files = [p for p in self.files if p.endswith(('.jpg', '.png', '.jpeg'))]  # yaha fileter lga denge taki image k alwala koi aur file nii hone chahiye


#yah length retun krta hai
    def __len__(self):
        return len(self.files)

# getItem index value leta hai
    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.files[idx]) # image ki path bna rahe hai
        image = Image.open(image_path).convert('RGB') # yah aconvert krk read kr rahe hai 3-r,g,b nii to b&w =2,leta hai error deta hai

        if self.transform:
            image = self.transform(image)

        return image

# now data set ready ho chuka upar ab use krgeneg
# get_transform - mtlb crop,short,reshape,resize krna image etc. hm kr rahe hai below method me
def get_transform(size, crop, final_size): #  ye sare input user se lega
    transform_list = []
    if size > 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.Resize(final_size))

    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)
        
# it will normalize the features
def adaptive_instance_normalization(content_feat, style_feat):
    # input is [batch size, channels, h, w] for normalization
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat) # style mean
    content_mean, content_std = calc_mean_std(content_feat) # conetnt mean
    # now normalize calculate
    normalized_content_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_content_feat * style_std.expand(size) + style_mean.expand(size)

def calc_mean_std(feat, eps=1e-5):
    # input for calculate mean std->[batch size, channels, h, w]=4
    size = feat.size()
    assert (len(size) == 4)  #4 is 4 dimensions
    batch_size, channels = size[:2]
    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    feat_var = feat.view(batch_size, channels, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)
    return feat_mean, feat_std