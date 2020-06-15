class RandomMask(object):
    """Add random occlusions to image.

    Args:
        mask: (Image.Image) - Image to use to occlude.
    """

    def __init__(self, mask):
        assert isinstance(mask, Image.Image)
        self.mask = mask

    def __call__(self, sample):
        self.mask = self.mask.resize((64,64))
        theta = randint(0,45)
        self.mask = self.mask.rotate(angle=theta)
        seed_x = randint(1,10)
        seed_y = randint(1,10)
        sample.paste(self.mask, (20*seed_x, 20*seed_y))

        return sample

