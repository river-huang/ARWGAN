import itertools

import numpy as np
import torch
import torch.nn.functional as F
import utils
import torch.nn as nn

def rgb_to_ycbcr_jpeg(image):
    return utils.rgb_to_ycbcr(image)


# 2. Chroma subsampling
def downsampling_420(image):
    # input: batch x height x width x 3
    # output: tuple of length 3
    #   y:  batch x height x width
    #   cb: batch x height/2 x width/2
    #   cr: batch x height/2 x width/2
    y, cb, cr = torch.split(image, 1, dim=1)
    cb = F.avg_pool2d(cb, kernel_size=2, stride=2)
    cr = F.avg_pool2d(cr, kernel_size=2, stride=2)
    return y, cb, cr


# 3. Block splitting
def image_to_patches(image):
    # input: batch x h x w
    # output: batch x h*w/64 x h x w
    # input: batch x h x w
    # output: batch x h*w/64 x h x w
    k = 8
    b, c, h, w = image.size()
    image_reshaped = torch.reshape(image, [b, h // k, k, -1, k])
    image_transposed = image_reshaped.permute((0, 1, 3, 2, 4))
    return torch.reshape(image_transposed, [b, -1, k, k])


# 4. DCT
def dct_8x8(image):
    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
            (2 * y + 1) * v * np.pi / 16)
    tensor = torch.from_numpy(tensor)

    alpha = np.array([1. / np.sqrt(2)] + [1] * 7, dtype=np.float32)
    scale = np.outer(alpha, alpha) * 0.25
    scale = torch.from_numpy(scale)

    image = image - 128.
    result = scale * torch.tensordot(image, tensor, dims=2).reshape(image.size())
    return result


# 5. Quantizaztion
y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60,
                                        55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103,
                                        77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T
y_table = torch.from_numpy(y_table)
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66],
                            [24, 26, 56, 99], [47, 66, 99, 99]]).T
c_table = torch.from_numpy(c_table)

def y_quantize(image, rounding, factor=1):
    image = image / (y_table * factor)
    image = rounding(image)
    return image


def c_quantize(image, rounding, factor=1):
    image = image / (c_table * factor)
    image = rounding(image)
    return image


# -5. Dequantization
def y_dequantize(image, factor=1):
    return image * (y_table * factor)


def c_dequantize(image, factor=1):
    return image * (c_table * factor)


# -4. Inverse DC
def idct_8x8(image):
    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
            (2 * v + 1) * y * np.pi / 16)
    tensor = torch.from_numpy(tensor)

    alpha = np.array([1. / np.sqrt(2)] + [1] * 7, dtype=np.float32)
    alpha = np.outer(alpha, alpha)
    alpha = torch.from_numpy(alpha)

    image = image * alpha
    result = 0.25 * torch.tensordot(image, tensor, dims=2).reshape(image.size()) + 128
    return result


# -3. Block joining
def patches_to_image(patches, height, width):
    # input: batch x h*w/64 x h x w
    # output: batch x h x w
    k = 8
    batch_size = patches.size()[0]
    image_reshaped = torch.reshape(patches,
                                   [batch_size, height // k, width // k, k, k])
    image_transposed = image_reshaped.permute((0, 1, 3, 2, 4))
    return torch.reshape(image_transposed, [batch_size, height, width])


# -2. Chroma upsampling
def upsampling_420(y, cb, cr):
    # input:
    #   y:  batch x height x width
    #   cb: batch x height/2 x width/2
    #   cr: batch x height/2 x width/2
    # output:
    #   image: batch x height x width x 3
    def repeat(x, k=2):
        height, width = x.size()[1:3]
        x = torch.unsqueeze(x, 1)
        x = x.repeat([1, 1, k, k])
        x = torch.reshape(x, [-1, height * k, width * k])
        return x

    cb = repeat(cb)
    cr = repeat(cr)
    return torch.stack((y, cb, cr), dim=1)


# -1. YCbCr -> RGB
def ycbcr_to_rgb_jpeg(image):
    return utils.ycbcr_to_rgb(image)


def diff_round(x):
    return torch.round(x) + (x - torch.round(x)) ** 3


def quality_to_factor(quality):
    if quality < 50:
        return 50. / quality
    else:
        return 2. - quality * 0.02


def jpeg_compress_decompress(image, downsample_c=False, rounding=diff_round, factor=1.,device=True):
    image=(image+1)/2
    image*=255
    b, c, h, w = image.size()

    orig_height, orig_width = h, w
    if h % 16 != 0 or w % 16 != 0:
        # Round up to next multiple of 16
        h = ((h - 1) // 16 + 1) * 16
        w = ((w - 1) // 16 + 1) * 16

        vpad = h - orig_height
        wpad = w - orig_width
        top = vpad // 2
        bottom = vpad - top
        left = wpad // 2
        right = wpad - left

        # image = tf.pad(image, [[0, 0], [top, bottom], [left, right], [0, 0]], 'SYMMETRIC')
        image = F.pad(image, [left, right, top, bottom])

    # "Compression"
    image = rgb_to_ycbcr_jpeg(image)
    if downsample_c:
        y, cb, cr = downsampling_420(image)
    else:
        y, cb, cr = torch.split(image, 1, dim=1)
    components = {'y': y, 'cb': cb, 'cr': cr}

    for k in components.keys():
        comp = components[k]
        comp = image_to_patches(comp)
        comp = dct_8x8(comp)
        comp = c_quantize(comp, rounding, factor) if k in ('cb', 'cr') else y_quantize(comp, rounding, factor)
        components[k] = comp

    # "Decompression"
    for k in components.keys():
        comp = components[k]
        comp = c_dequantize(comp, factor) if k in ('cb', 'cr') else y_dequantize(comp, factor)
        comp = idct_8x8(comp)
        if k in ('cb', 'cr'):
            if downsample_c:
                comp = patches_to_image(comp, h // 2, w // 2)
            else:
                comp = patches_to_image(comp, h, w)
        else:
            comp = patches_to_image(comp, h, w)
        components[k] = comp

    y, cb, cr = components['y'], components['cb'], components['cr']
    if downsample_c:
        image = upsampling_420(y, cb, cr)
    else:
        image = torch.stack((y, cb, cr), dim=1)
    image = ycbcr_to_rgb_jpeg(image)

    # Crop to original size
    if orig_height != h or orig_width != w:
        # image = image[:, top:-bottom, left:-right]
        image = image[:, :, :-vpad, :-wpad]

    # Hack: RGB -> YUV -> RGB sometimes results in incorrect values
    #    min_value = tf.minimum(tf.reduce_min(image), 0.)
    #    max_value = tf.maximum(tf.reduce_max(image), 255.)
    #    value_range = max_value - min_value
    #    image = 255 * (image - min_value) / value_range
    image = torch.min(torch.tensor(255.), torch.max(torch.tensor(0.), image))
    image/=255
    return image
class Jpeg(nn.Module):
    def __init__(self,factor):
        super(Jpeg, self).__init__()
        self.factor=factor
        self.device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, noise_and_cover):
        encoded_image=noise_and_cover[0]
        encoded_image=encoded_image.cpu()
        jpeg_image=jpeg_compress_decompress(encoded_image,factor=quality_to_factor(50))
        noise_and_cover[0]=torch.FloatTensor(jpeg_image).to(self.device)
        return noise_and_cover
