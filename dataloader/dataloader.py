
# Dataset

# occlusion transform?
mask = Image.open('/lab/vislab/DATA/masks/mask.png')

transformations = transforms.Compose([
    transforms.Resize((256,256)),
    RandomMask(mask),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(train_path, transformations)

plt.imshow(transforms.ToPILImage()(dataset[3][0]), interpolation="bicubic")

train_set, test_set = torch.utils.data.random_split(dataset, [5000, 1033])

