import torch
import torch.nn as nn
import torchvision.models as models

class LPIPS(nn.Module):
    def __init__(self, net='vgg', use_dropout=True):
        """
        Learned Perceptual Image Patch Similarity (LPIPS) metric.
        
        Args:
            net (str): Backbone network ('vgg' is the default and best performer in the paper).
            use_dropout (bool): Whether to use dropout in the linear calibration layers 
                                (not strictly necessary for inference, but part of the training structure).
        """
        super(LPIPS, self).__init__()
        
        if net == 'vgg':
            # Load VGG16 pretrained on ImageNet
            vgg_pretrained_features = models.vgg16(pretrained=True).features
            
            # Divide VGG into 5 slices to extract features at different depths
            # As described in the paper, we use layers conv1-5
            self.slice1 = torch.nn.Sequential()
            self.slice2 = torch.nn.Sequential()
            self.slice3 = torch.nn.Sequential()
            self.slice4 = torch.nn.Sequential()
            self.slice5 = torch.nn.Sequential()
            
            # Mapping VGG layers to slices (Standard LPIPS slicing)
            for x in range(4):
                self.slice1.add_module(str(x), vgg_pretrained_features[x])
            for x in range(4, 9):
                self.slice2.add_module(str(x), vgg_pretrained_features[x])
            for x in range(9, 16):
                self.slice3.add_module(str(x), vgg_pretrained_features[x])
            for x in range(16, 23):
                self.slice4.add_module(str(x), vgg_pretrained_features[x])
            for x in range(23, 30):
                self.slice5.add_module(str(x), vgg_pretrained_features[x])
        else:
            raise NotImplementedError("Only VGG is implemented in this snippet.")

        # Freeze the backbone (we are not training the VGG weights)
        for param in self.parameters():
            param.requires_grad = False

        # Define the "channel scaling factors" (w in Eq. 1 / Fig. 3)
        # In the paper, these are learned (calibrated). 
        # Initializing them to 1.0 results in Cosine Distance (uncalibrated), 
        # which the paper notes still performs better than SSIM/PSNR.
        self.lin0 = NetLinLayer(64, use_dropout=use_dropout)
        self.lin1 = NetLinLayer(128, use_dropout=use_dropout)
        self.lin2 = NetLinLayer(256, use_dropout=use_dropout)
        self.lin3 = NetLinLayer(512, use_dropout=use_dropout)
        self.lin4 = NetLinLayer(512, use_dropout=use_dropout)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]

    def forward(self, x, x0):
        """
        Calculate LPIPS distance between two batches of images.
        Input should be normalized to [-1, 1].
        """
        # 1. Extract features from both images
        outs_x = self.get_features(x)
        outs_x0 = self.get_features(x0)

        diffs = []
        
        # 2. Iterate through layers
        for kk in range(len(self.lins)):
            # Normalize activations in the channel dimension (Eq 1: y_hat)
            feat_x = normalize_tensor(outs_x[kk])
            feat_x0 = normalize_tensor(outs_x0[kk])
            
            # Square difference
            diff = (feat_x - feat_x0) ** 2

            # 3. Apply linear scaling (w_l) and spatial average
            # The lin layer applies the 'w' vector from Fig 3
            res = self.lins[kk](diff) 
            
            # Average spatially (H, W) -> Scalar per image
            dist = res.mean([2, 3], keepdim=True)
            diffs.append(dist)

        # 4. Sum across layers
        val = sum(diffs)
        return val

    def get_features(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

class NetLinLayer(nn.Module):
    """
    A single linear layer (1x1 convolution) to act as the learned weights 'w'.
    """
    def __init__(self, chn_in, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if use_dropout else []
        
        # 1x1 Convolution maps channel inputs to a weighted sum (or simple scaling)
        # Note: In the official repo, they often use a 1x1 conv to map chn_in -> 1
        # representing the weighted L2 distance immediately. 
        # Here we follow the logic: scale channels -> sum.
        layers += [nn.Conv2d(chn_in, 1, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)
        
        # Initialize weights to 1.0 (Cosine Distance equivalent)
        # This aligns with the "uncalibrated" performance mentioned in the paper.
        # To get the "Linear" performance (LPIPS-Lin), these weights must be trained 
        # on the BAPPS dataset.
        torch.nn.init.constant_(self.model[-1].weight, 1.0/chn_in) 

    def forward(self, x):
        return self.model(x)

def normalize_tensor(x, eps=1e-10):
    """
    Unit-normalize tensor in the channel dimension (dim=1).
    Math: x = x / sqrt(sum(x^2))
    """
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)

# Example Usage
# if __name__ == "__main__":
#     # Initialize metric
#     lpips_metric = LPIPS(net='vgg')
#     lpips_metric.eval() # Set to evaluation mode

#     # Create two dummy image patches (Batch Size 1, 3 Channels, 64x64)
#     # Inputs should be normalized between [-1, 1]
#     img1 = torch.zeros(1, 3, 64, 64)
#     img2 = torch.ones(1, 3, 64, 64)

#     # Compute distance
#     distance = lpips_metric(img1, img2)
#     print(f"LPIPS Distance: {distance.item():.4f}")