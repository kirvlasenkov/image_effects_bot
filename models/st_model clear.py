from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import copy
from scipy import misc


class StyleTransferModel:

    def __init__(self):
        imsize = 256
        self.loader = transforms.Compose([
            transforms.Resize(imsize),  # normalize the image size
            transforms.CenterCrop(imsize),
            transforms.ToTensor()])  # turn it into a convenient format
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unloader = transforms.ToPILImage()  # tensor in the crater
        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()

    def image_loader(self, image_name):
        image = Image.open(image_name)
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    def get_style_model_and_losses(self, cnn, normalization_mean, normalization_std,
                                   style_img, content_img,
                                   content_layers=content_layers_default,
                                   style_layers=style_layers_default):
        cnn = copy.deepcopy(cnn)

        # normalization module
        normalization = Normalization(normalization_mean, normalization_std).to(self.device)

        # just in order to have an iterable access to or list of content/syle
        # losses
        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                # Redefining the relu level
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    def get_input_optimizer(self, input_img):
        # this line to show that input is a parameter that requires a gradient
        #adds the contents of the image tensor to the list of parameters changed by the optimizer
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

    def transfer_style(self, content_img_stream, style_img_stream):
        style_img = self.image_loader(style_img_stream)  # as well as here
        content_img = self.image_loader(content_img_stream)  # change the path to the one that you have.
        input_img = content_img.clone()
        output = self.run_style_transfer(self.cnn,
                                         self.cnn_normalization_mean,
                                         self.cnn_normalization_std,
                                         content_img, style_img, input_img)
        return self.imsave(output)

    def image_loader(self, image_name):
        image = Image.open(image_name)
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    def imsave(self, tensor, title="out.jpg"):
        image = tensor.cpu().clone()
        image = image.squeeze(0)  # function for drawing an image
        image = self.unloader(image)
        return image