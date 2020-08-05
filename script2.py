# DF-Net Imports
from collections import defaultdict
import cv2
from itertools import islice
from multiprocessing.pool import ThreadPool as Pool
from pathlib import Path
import torch.nn.functional as F
import tqdm

# Imports
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pytorch_metric_learning import losses, samplers
import random
from skimage import io, transform
from sklearn.manifold import TSNE
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data.sampler import Sampler
from torchvision import datasets, models, transforms

def euclidean_dist(inputs_):
    # Compute pairwise distance, replace by the official when merged
    n = inputs_.size(0)
    dist = torch.pow(inputs_, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs_, inputs_.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

class BatchAllLoss(nn.Module):
    def __init__(self, device, margin=0.2):
        super(BatchAllLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)
        self.device = device

    def forward(self, inputs, targets):
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist_mat = euclidean_dist(inputs)
        # split the positive and negative pairs
        eyes_ = Variable(torch.eye(n, n)).to(self.device)
        pos_mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        neg_mask = eyes_.eq(eyes_) ^ pos_mask


        pos_mask = pos_mask ^ eyes_.eq(1)
        pos_dist = torch.masked_select(dist_mat, pos_mask)
        neg_dist = torch.masked_select(dist_mat, neg_mask)

        num_instances = len(pos_dist)//n + 1
        num_neg_instances = n - num_instances
        pos_dist = pos_dist.reshape(len(pos_dist)//(num_instances-1), num_instances-1)
        neg_dist = neg_dist.reshape(len(neg_dist)//(num_neg_instances), num_neg_instances)

        loss = list()
        for i, pos_pair in enumerate(pos_dist):
            neg_dist_ = neg_dist[i].repeat(num_instances - 1, 1)
            pos_dist_ = pos_pair.repeat(num_neg_instances, 1)
            pos_dist_ = pos_dist_.t()
            pos_dist_ = pos_dist_.reshape(num_neg_instances * (num_instances - 1))
            neg_dist_ = neg_dist_.reshape(num_neg_instances * (num_instances - 1))

            y = neg_dist_.data.new()
            y.resize_as_(neg_dist_.data)
            y.fill_(1)
            y = Variable(y)
            loss.append(self.ranking_loss(neg_dist_, pos_dist_, y))
        loss = torch.mean(torch.stack([loss_ for loss_ in loss]))
        return loss
    
    def __str__(self):
        return "Batch All, margin = {}".format(self.margin)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            # unorm
            # t.mul_(s).add_(m)

            # norm
            t.sub_(m).div_(s)
        return tensor

def map_features(outputs, labels, out_file):
    # create array of column for each feature output
    feat_cols = ['feature' + str(i) for i in range(outputs.shape[1])]
    # make dataframe of outputs -> labels
    df = pd.DataFrame(outputs, columns=feat_cols)
    df['y'] = labels
    df['labels'] = df['y'].apply(lambda i: str(i))
    # clear outputs and labelsj
    outputs, labels = None, None
    # creates an array of random indices from size of outputs
    np.random.seed(42)
    rndperm = np.random.permutation(df.shape[0])
    num_examples = 3000
    df_subset = df.loc[rndperm[:num_examples], :].copy()
    data_subset = df_subset[feat_cols].values
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=300)
    tsne_results = tsne.fit_transform(data_subset)
    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(16, 10))
    plt.scatter(
        x=df_subset["tsne-2d-one"],
        y=df_subset["tsne-2d-two"],
        c=df_subset["y"],
        s=3
    )
    plt.savefig(out_file, bbox_inches='tight', pad_inches=0)
    plt.close()



# Configurations
# Device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Num epochs
num_epochs = 50

# Model
model = models.resnet50(pretrained=True)

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Batch size
batch_size = 32

# # Data set
train_path = '/lab/vislab/DATA/CUB/images/train/'
test_path = '/lab/vislab/DATA/CUB/images/test/'
# train_path = '/lab/vislab/DATA/just/infilling/samples/places2/mini/'
# test_path = '/lab/vislab/DATA/just/infilling/samples/places2/mini/'



# Loss function
criterion = losses.TripletMarginLoss(margin=0.2,triplets_per_anchor="all") # so we are already doing batchall
# criterion = BatchAllLoss(device, margin=0.2)
# criterion = nn.CrossEntropyLoss()

# criterion = torch.nn.CosineEmbeddingLoss()

class RandomMask(object):
    """Add random occlusions to image.

    Args:
        mask: (Image.Image) - Image to use to occlude.
    """

    def __init__(self, mask):
        assert isinstance(mask, Image.Image)
        self.mask = mask

    def __call__(self, sample):
        self.mask = self.mask.resize((128, 128))
        theta = randint(0, 45)
        self.mask = self.mask.rotate(angle=theta)
        seed_x = randint(1, 10)
        seed_y = randint(1, 10)
        sample.paste(self.mask, (20 * seed_x, 20 * seed_y))

        return sample


# mask = Image.open('./samples/places2/mask/mask_01.png')

transformations = transforms.Compose([
    transforms.RandomResizedCrop((256, 256)),
    # RandomMask(mask),
    transforms.ToTensor(),
    # transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ])
])

dataset = datasets.ImageFolder(train_path, transformations)
test_dataset = datasets.ImageFolder(test_path, transformations)

train_sampler = samplers.MPerClassSampler(dataset.targets, 8, len(dataset))

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, sampler=None, num_workers=4, drop_last=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, sampler=None, num_workers=4, drop_last=True)

# DF-Net Utils
def resize_like(x, target, mode='bilinear'):
    return F.interpolate(x, target.shape[-2:], mode=mode, align_corners=False)


def list2nparray(lst, dtype=None):
    """fast conversion from nested list to ndarray by pre-allocating space"""
    if isinstance(lst, np.ndarray):
        return lst
    assert isinstance(lst, (list, tuple)), 'bad type: {}'.format(type(lst))
    assert lst, 'attempt to convert empty list to np array'
    if isinstance(lst[0], np.ndarray):
        dim1 = lst[0].shape
        assert all(i.shape == dim1 for i in lst)
        if dtype is None:
            dtype = lst[0].dtype
            assert all(i.dtype == dtype for i in lst), 'bad dtype: {} {}'.format(dtype, set(i.dtype for i in lst))
    elif isinstance(lst[0], (int, float, complex, np.number)):
        return np.array(lst, dtype=dtype)
    else:
        dim1 = list2nparray(lst[0])
        if dtype is None:
            dtype = dim1.dtype
        dim1 = dim1.shape
    shape = [len(lst)] + list(dim1)
    rst = np.empty(shape, dtype=dtype)
    for idx, i in enumerate(lst):
        rst[idx] = i
    return rst


def get_img_list(path):
    return sorted(list(Path(path).glob('*.png'))) + sorted(list(Path(path).glob('*.jpg'))) + sorted(
        list(Path(path).glob('*.jpeg')))


def gen_miss(img, mask, output):
    imgs = get_img_list(img)
    masks = get_img_list(mask)
    print('Total images:', len(imgs), len(masks))

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    for i, (img, mask) in tqdm.tqdm(enumerate(zip(imgs, masks))):
        path = out.joinpath('miss_%04d.png' % (i + 1))
        img = cv2.imread(str(img), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img.shape[:2][::-1])
        mask = mask[..., np.newaxis]
        miss = img * (mask > 127) + 255 * (mask <= 127)

        cv2.imwrite(str(path), miss)


def merge_imgs(dirs, output, row=1, gap=2, res=512):
    image_list = [get_img_list(path) for path in dirs]
    img_count = [len(image) for image in image_list]
    print('Total images:', img_count)
    assert min(img_count) > 0, 'Please check the path of empty folder.'

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_img = len(dirs)
    row = row
    column = (n_img - 1) // row + 1
    print('Row:', row)
    print('Column:', column)

    for i, unit in tqdm.tqdm(enumerate(zip(*image_list))):
        name = output_dir.joinpath('merge_%04d.png' % i)
        merge = np.ones([
            res * row + (row + 1) * gap, res * column + (column + 1) * gap, 3], np.uint8) * 255
        for j, img in enumerate(unit):
            r = j // column
            c = j - r * column
            img = cv2.imread(str(img), cv2.IMREAD_COLOR)
            if img.shape[:2] != (res, res):
                img = cv2.resize(img, (res, res))
            start_h, start_w = (r + 1) * gap + r * res, (c + 1) * gap + c * res
            merge[start_h: start_h + res, start_w: start_w + res] = img
        cv2.imwrite(str(name), merge)



# DF-Net Model
def get_norm(name, out_channels):
    if name == 'batch':
        norm = nn.BatchNorm2d(out_channels)
    elif name == 'instance':
        norm = nn.InstanceNorm2d(out_channels)
    else:
        norm = None
    return norm


def get_activation(name):
    if name == 'relu':
        activation = nn.ReLU()
    elif name == 'elu':
        activation == nn.ELU()
    elif name == 'leaky_relu':
        activation = nn.LeakyReLU(negative_slope=0.2)
    elif name == 'tanh':
        activation = nn.Tanh()
    elif name == 'sigmoid':
        activation = nn.Sigmoid()
    else:
        activation = None
    return activation


class Conv2dSame(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        padding = self.conv_same_pad(kernel_size, stride)
        if type(padding) is not tuple:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding)
        else:
            self.conv = nn.Sequential(
                nn.ConstantPad2d(padding * 2, 0),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0)
            )

    def conv_same_pad(self, ksize, stride):
        if (ksize - stride) % 2 == 0:
            return (ksize - stride) // 2
        else:
            left = (ksize - stride) // 2
            right = left + 1
            return left, right

    def forward(self, x):
        return self.conv(x)


class ConvTranspose2dSame(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        padding, output_padding = self.deconv_same_pad(kernel_size, stride)
        self.trans_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride,
            padding, output_padding)

    def deconv_same_pad(self, ksize, stride):
        pad = (ksize - stride + 1) // 2
        outpad = 2 * pad + stride - ksize
        return pad, outpad

    def forward(self, x):
        return self.trans_conv(x)


class UpBlock(nn.Module):

    def __init__(self, mode='nearest', scale=2, channel=None, kernel_size=4):
        super().__init__()

        self.mode = mode
        if mode == 'deconv':
            self.up = ConvTranspose2dSame(
                channel, channel, kernel_size, stride=scale)
        else:
            def upsample(x):
                return F.interpolate(x, scale_factor=scale, mode=mode)

            self.up = upsample

    def forward(self, x):
        return self.up(x)


class EncodeBlock(nn.Module):

    def __init__(
            self, in_channels, out_channels, kernel_size, stride,
            normalization=None, activation=None):
        super().__init__()

        self.c_in = in_channels
        self.c_out = out_channels

        layers = []
        layers.append(
            Conv2dSame(self.c_in, self.c_out, kernel_size, stride))
        if normalization:
            layers.append(get_norm(normalization, self.c_out))
        if activation:
            layers.append(get_activation(activation))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class DecodeBlock(nn.Module):

    def __init__(
            self, c_from_up, c_from_down, c_out, mode='nearest',
            kernel_size=4, scale=2, normalization='batch', activation='relu'):
        super().__init__()

        self.c_from_up = c_from_up
        self.c_from_down = c_from_down
        self.c_in = c_from_up + c_from_down
        self.c_out = c_out

        self.up = UpBlock(mode, scale, c_from_up, kernel_size=scale)

        layers = []
        layers.append(
            Conv2dSame(self.c_in, self.c_out, kernel_size, stride=1))
        if normalization:
            layers.append(get_norm(normalization, self.c_out))
        if activation:
            layers.append(get_activation(activation))
        self.decode = nn.Sequential(*layers)

    def forward(self, x, concat=None):
        out = self.up(x)
        if self.c_from_down > 0:
            out = torch.cat([out, concat], dim=1)
        out = self.decode(out)
        return out


class BlendBlock(nn.Module):

    def __init__(
            self, c_in, c_out, ksize_mid=3, norm='batch', act='leaky_relu'):
        super().__init__()
        c_mid = max(c_in // 2, 32)
        self.blend = nn.Sequential(
            Conv2dSame(c_in, c_mid, 1, 1),
            get_norm(norm, c_mid),
            get_activation(act),
            Conv2dSame(c_mid, c_out, ksize_mid, 1),
            get_norm(norm, c_out),
            get_activation(act),
            Conv2dSame(c_out, c_out, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.blend(x)


class FusionBlock(nn.Module):
    def __init__(self, c_feat, c_alpha=1):
        super().__init__()
        c_img = 3
        self.map2img = nn.Sequential(
            Conv2dSame(c_feat, c_img, 1, 1),
            nn.Sigmoid())
        self.blend = BlendBlock(c_img * 2, c_alpha)

    def forward(self, img_miss, feat_de):
        img_miss = resize_like(img_miss, feat_de)
        raw = self.map2img(feat_de)
        alpha = self.blend(torch.cat([img_miss, raw], dim=1))
        result = alpha * raw + (1 - alpha) * img_miss
        return result, alpha, raw


class DFNet(nn.Module):
    def __init__(
            self, c_img=3, c_mask=1, c_alpha=3,
            mode='nearest', norm='batch', act_en='relu', act_de='leaky_relu',
            en_ksize=[7, 5, 5, 3, 3, 3, 3, 3], de_ksize=[3] * 8,
            blend_layers=[0, 1, 2, 3, 4, 5]):
        super().__init__()

        c_init = c_img + c_mask

        self.n_en = len(en_ksize)
        self.n_de = len(de_ksize)
        assert self.n_en == self.n_de, (
            'The number layer of Encoder and Decoder must be equal.')
        assert self.n_en >= 1, (
            'The number layer of Encoder and Decoder must be greater than 1.')

        assert 0 in blend_layers, 'Layer 0 must be blended.'

        self.en = []
        c_in = c_init
        self.en.append(
            EncodeBlock(c_in, 64, en_ksize[0], 2, None, None))
        for k_en in en_ksize[1:]:
            c_in = self.en[-1].c_out
            c_out = min(c_in * 2, 512)
            self.en.append(EncodeBlock(
                c_in, c_out, k_en, stride=2,
                normalization=norm, activation=act_en))

        # register parameters
        for i, en in enumerate(self.en):
            self.__setattr__('en_{}'.format(i), en)

        self.de = []
        self.fuse = []
        for i, k_de in enumerate(de_ksize):

            c_from_up = self.en[-1].c_out if i == 0 else self.de[-1].c_out
            c_out = c_from_down = self.en[-i - 1].c_in
            layer_idx = self.n_de - i - 1

            self.de.append(DecodeBlock(
                c_from_up, c_from_down, c_out, mode, k_de, scale=2,
                normalization=norm, activation=act_de))
            if layer_idx in blend_layers:
                self.fuse.append(FusionBlock(c_out, c_alpha))
            else:
                self.fuse.append(None)

        # register parameters
        for i, de in enumerate(self.de[::-1]):
            self.__setattr__('de_{}'.format(i), de)
        for i, fuse in enumerate(self.fuse[::-1]):
            if fuse:
                self.__setattr__('fuse_{}'.format(i), fuse)

    def forward(self, img_miss, mask):

        out = torch.cat([img_miss, mask], dim=1)

        out_en = [out]
        for encode in self.en:
            out = encode(out)
            out_en.append(out)

        results = []
        alphas = []
        raws = []
        for i, (decode, fuse) in enumerate(zip(self.de, self.fuse)):
            out = decode(out, out_en[-i - 2])
            if fuse:
                result, alpha, raw = fuse(img_miss, out)
                results.append(result)
                alphas.append(alpha)
                raws.append(raw)

        return results[::-1], alphas[::-1], raws[::-1]


# Inpainter
class Inpainter:

    def __init__(self, model_path, input_size, batch_size):
        self.model_path = model_path
        self._input_size = input_size
        self.batch_size = batch_size
        self.init_model(model_path)


    @property
    def input_size(self):
        if self._input_size > 0:
            return (self._input_size, self._input_size)
        elif 'celeba' in self.model_path:
            return (256, 256)
        else:
            return (256, 256)


    def init_model(self, path):
        if torch.cuda.is_available():
            self.device = torch.device('cuda:1')
            print('Using gpu.')
        else:
            self.device = torch.device('cpu')
            print('Using cpu.')

        self.model = DFNet().to(self.device)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint) # loading in that pre-trained model
        self.model.eval()

        print('Model %s loaded.' % path)


    def to_numpy(self, tensor):
        tensor = tensor.mul(255).byte().data.cpu().numpy()
        tensor = np.transpose(tensor, [0, 2, 3, 1])
        return tensor


    def inpaint(self, imgs, masks):
        print("inpainting images ...")
#
        imgs = self.to_numpy(imgs)
        for i in range(imgs.shape[0]):
            imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
            # imgs[i] = np.transpose(imgs[i], [2,0,1])

        # print("IMGS: AFTER BGR. shape: ", imgs.shape)
        imgs = np.transpose(imgs, [0,3,1,2])
        imgs = torch.from_numpy(imgs)#

        imgs = imgs.to(self.device)
        masks = masks.to(self.device)

        imgs = imgs.float().div(255)
        masks = masks.float().div(255)

        # print(imgs.shape)
        # print(masks.shape)

        imgs_miss = imgs * masks

        # print(imgs_miss.shape)
        # plt.imshow(transforms.ToPILImage()(imgs[0].cpu()))
        # plt.savefig("imgs.png")
        # plt.imshow(transforms.ToPILImage()(masks[0].cpu()))
        # plt.savefig("masks.png")
        # plt.imshow(transforms.ToPILImage()(imgs_miss[0].cpu()))
        # plt.savefig("imgs_miss.png")

        result, alpha, raw = self.model(imgs_miss, masks)
        result, alpha, raw = result[0], alpha[0], raw[0]
        result = imgs * masks + result * (1 - masks)
        # plt.imshow(transforms.ToPILImage()(result[0].cpu()))
        # plt.savefig("inpainted.png")
        # 12/0
        return result


# Main Trainer
def train_model():
    """Generic function to train model"""
    lines = []
    start_time = datetime.now()
    
    # DF-Net Tester Instantiate
    pretrained_model_path = './model/model_places2.pth'
    inpainter = Inpainter(pretrained_model_path, 256, 32)

    # Epochs
    correct = 0
    incorrect = 0
    num_batches = 1
    loss_values = []
    train_values = []

    location = [(a,b) for a in range(255) for b in range(255)]
    small_mask = Image.new('L', (4, 4), 0)
    masks = []
    for _ in range(32):
        base = Image.new('L',(256,256),255)
        r = random.choice(location)
        base.paste(small_mask, r)
        location.pop(location.index(r))
        base = np.ascontiguousarray(np.expand_dims(base, 0)).astype(np.uint8)
        masks.append(base)
    masks = np.array(masks)
    masks = torch.from_numpy(masks)

    for epoch in range(num_epochs):
        print("epoch num:", epoch)

        for phase in ['train', 'valid']:

            correct = 0
            incorrect = 0

            running_outputs = torch.FloatTensor().cpu()
            running_labels = torch.LongTensor().cpu()
            running_loss = 0.0

            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            # Batches
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                
                inpainted_img_batch = inpainter.inpaint(inputs, masks)

                # unorm = UnNormalize([ 104/255, 107/255, 128/255 ],[ 1/255, 1/255, 1/255 ])
                # inpainted_img_batch = unorm(inpainted_img_batch)
                # inputs = unorm(inputs)

                # plt.imshow(transforms.ToPILImage()(inpainted_img_batch[0].cpu()))
                # plt.savefig("inpainted.png")
                # plt.imshow(transforms.ToPILImage()(inputs[0].cpu()))
                # plt.savefig("input.png")
                # 12/0

                # inpainted_img_batch, labels = inpainted_img_batch.to(device, dtype=torch.float), labels.to(device)                    
                output = model(inpainted_img_batch)

                loss = criterion(output, labels)
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                if phase == 'valid':
                    running_outputs = torch.cat((running_outputs, output.cpu().detach()), 0)
                    running_labels = torch.cat((running_labels, labels.cpu().detach()), 0)
                    running_loss += loss.item()
                    num_batches += 1

            if phase == 'valid':
                # Accuracy
                running_outputs = running_outputs.to(device)
                running_labels = running_labels.to(device)

                for idx, emb in enumerate(running_outputs):
                    pairwise = torch.nn.PairwiseDistance(p=2).to(device)
                    dist = pairwise(emb, running_outputs)
                    closest = torch.topk(dist, 2, largest=False).indices[1]
                    if running_labels[idx] == running_labels[closest]:
                        correct += 1
                    else:
                        incorrect += 1

                print(running_loss / num_batches)
                print("Correct", correct)
                print("Incorrect", incorrect)
                if correct + incorrect != 0:
                    accuracy = correct / (correct + incorrect)
                    train_values.append(accuracy)
                else:
                    accuracy = "Can't divide by 0."
                print("Accuracy: ", accuracy)

                loss_values.append(running_loss / num_batches)

            time_elapsed = datetime.now() - start_time
            print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

    
    plt.plot(loss_values)
    plt.title("Train Loss Function")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    plt.savefig('train_loss.png')
    # map_features(running_outputs.cpu(), running_labels.cpu(), 'features.png')
    torch.save(model.state_dict(), 'saved_model.pth')
    print("Finished training model.")
    return model, running_loss

def test_model():
    """Generic function to train model"""
    lines = []
    start_time = datetime.now()
    PATH = 'saved_model.pth'
    net = models.resnet50(pretrained=True)
    net.load_state_dict(torch.load(PATH))
    net.to(device)
    net.eval()
    net.train(False)
    
    # DF-Net Tester Instantiate
    pretrained_model_path = './model/model_places2.pth'
    inpainter = Inpainter(pretrained_model_path, 256, 32)

    plt.figure(figsize=(16, 10))
    plt.title("Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    for i in range(2,5):

        # Epochs
        correct = 0
        incorrect = 0
        num_batches = 1
        loss_values = []
        train_values = []

        mask_path = "./samples/places2/mask_0{}/".format(i)


        for epoch in range(num_epochs):
            print("epoch num:", epoch)
            location = [(a,b) for a in range(250) for b in range(250)]
            small_mask = Image.new('L', (2, 2), 'black')
            masks = []
            for _ in range(32):
                base = Image.new('L',(256,256),'white')
                r = random.choice(location)
                base.paste(small_mask, r)
                location.pop(location.index(r))
                base = np.ascontiguousarray(np.expand_dims(base, 0)).astype(np.uint8)
                masks.append(base)
            masks = np.array(masks)
            masks = torch.from_numpy(masks)


            correct = 0
            incorrect = 0

            running_outputs = torch.FloatTensor().cpu()
            running_labels = torch.LongTensor().cpu()
            running_loss = 0.0

            # Batches
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(test_loader):
                    # optimizer.zero_grad()
                    
                    inpainted_img_batch = inpainter.inpaint(inputs, masks)
                    # unorm = UnNormalize([ 104/255, 107/255, 128/255 ],[ 1/255, 1/255, 1/255 ])
                    # inpainted_img_batch = unorm(inpainted_img_batch)
                    # inputs, labels = inputs.to(device), labels.to(device)
                    # output = model(inputs)

                    inpainted_img_batch, labels = inpainted_img_batch.to(device, dtype=torch.float), labels.to(device)

                    output = net(inpainted_img_batch)

                    loss = criterion(output, labels)

                    running_outputs = torch.cat((running_outputs, output.cpu().detach()), 0)
                    running_labels = torch.cat((running_labels, labels.cpu().detach()), 0)
                    running_loss += loss.item()
                    num_batches += 1

            # Accuracy
            running_outputs = running_outputs.to(device)
            running_labels = running_labels.to(device)

            for idx, emb in enumerate(running_outputs):
                pairwise = torch.nn.PairwiseDistance(p=2).to(device)
                dist = pairwise(emb, running_outputs)
                closest = torch.topk(dist, 2, largest=False).indices[1]
                if running_labels[idx] == running_labels[closest]:
                    correct += 1
                else:
                    incorrect += 1

            print(running_loss / num_batches)
            print("Correct", correct)
            print("Incorrect", incorrect)
            if correct + incorrect != 0:
                accuracy = correct / (correct + incorrect)
                train_values.append(accuracy)
            else:
                accuracy = "Can't divide by 0."
            print("Accuracy: ", accuracy)

            loss_values.append(running_loss / num_batches)

            time_elapsed = datetime.now() - start_time
            print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

        if i == 1: 
            plot1, = plt.plot(train_values)
        elif i == 2: 
            plot2, = plt.plot(train_values)
        elif i == 3: 
            plot3, = plt.plot(train_values)
        elif i == 4: 
            plot4, = plt.plot(train_values)
        elif i == 5: 
            plot5, = plt.plot(train_values)
        
    # pure control group :
    correct = 0
    incorrect = 0
    num_batches = 1
    loss_values = []
    train_values = []
    
    for epoch in range(num_epochs):
        print("epoch num:", epoch)

        correct = 0
        incorrect = 0

        running_outputs = torch.FloatTensor().cpu()
        running_labels = torch.LongTensor().cpu()
        running_loss = 0.0

        # Batches
        with torch.no_grad():
            for batch_idx, (tensor, labels) in enumerate(test_loader):
                # optimizer.zero_grad()
                
                tensor = tensor.mul(255).byte().data.cpu().numpy()
                tensor = np.transpose(tensor, [0, 2, 3, 1])
                inputs = tensor
                for i in range(inputs.shape[0]):
                    inputs[i] = cv2.cvtColor(inputs[i], cv2.COLOR_BGR2RGB)
                    # imgs[i] = np.transpose(imgs[i], [2,0,1])

                # print("IMGS: AFTER BGR. shape: ", imgs.shape)
                inputs = np.transpose(inputs, [0,3,1,2])
                inputs = torch.from_numpy(inputs)#

                inputs = inputs.float().div(255)
                # unorm = UnNormalize([ 104/255, 107/255, 128/255 ],[ 1/255, 1/255, 1/255 ])
                # inputs = unorm(inputs)

                inputs, labels = inputs.to(device), labels.to(device)
                output = net(inputs)
                loss = criterion(output, labels)

                running_outputs = torch.cat((running_outputs, output.cpu().detach()), 0)
                running_labels = torch.cat((running_labels, labels.cpu().detach()), 0)
                running_loss += loss.item()
                num_batches += 1

        # Accuracy
        running_outputs = running_outputs.to(device)
        running_labels = running_labels.to(device)

        for idx, emb in enumerate(running_outputs):
            pairwise = torch.nn.PairwiseDistance(p=2).to(device)
            dist = pairwise(emb, running_outputs)
            closest = torch.topk(dist, 2, largest=False).indices[1]
            if running_labels[idx] == running_labels[closest]:
                correct += 1
            else:
                incorrect += 1

        print(running_loss / num_batches)
        print("Correct", correct)
        print("Incorrect", incorrect)
        if correct + incorrect != 0:
            accuracy = correct / (correct + incorrect)
            train_values.append(accuracy)
        else:
            accuracy = "Can't divide by 0."
        print("Accuracy: ", accuracy)

        loss_values.append(running_loss / num_batches)

        time_elapsed = datetime.now() - start_time
        print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    plot5, = plt.plot(train_values)


    plt.legend([plot2,plot3,plot4,plot5],["i2","i3","i4","control"])
    plt.show()
    plt.savefig('test_accuracy.png')
    plt.close()
    plt.figure(figsize=(16, 10))
    plt.plot(loss_values)
    plt.title("Test Loss Function")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    plt.savefig('test_loss.png')
    return model, running_loss

# Run Script
model.to(device)
trained_model, loss = train_model()

test_results, loss = test_model()