import torch
import torch.nn as nn


def conv_block(in_c, out_c, down=True, use_batch=True, act="relu"):
    """
    Convolutional block for encoder/decoder architecture.
    
    Args:
        in_c (int): Number of input channels
        out_c (int): Number of output channels
        down (bool): If True, create downsampling block; if False, create upsampling block
        use_batch (bool): Whether to use batch normalization
        act (str): Activation function type ("relu", "leaky_relu", "tanh")
        
    Returns:
        nn.Sequential: Sequential container of layers
    """
    layers = []
    
    if down:
        # Downsampling: Conv with stride 2 for spatial reduction
        layers.append(nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=not use_batch))
    else:
        # Upsampling: ConvTranspose with stride 2 for spatial expansion
        layers.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=not use_batch))
    
    # Add batch norm if specified
    if use_batch:
        layers.append(nn.BatchNorm2d(out_c))
    
    # Add activation function
    if act == "relu":
        layers.append(nn.ReLU(inplace=True))
    elif act == "leaky_relu":
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    elif act == "tanh":
        layers.append(nn.Tanh())
    
    return nn.Sequential(*layers)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, init_features=64):
        """
        U-Net architecture for image-to-image translation.
        
        Args:
            in_channels (int): Number of input image channels
            init_features (int): Initial number of features (channels in first layer)
        """
        super(UNetGenerator, self).__init__()
        
        # Initial number of features
        features = init_features
        
        # Encoder (downsampling path)
        # layer 1: no batch norm
        self.encoder1 = conv_block(in_channels, features, down=True, use_batch=False, act="leaky_relu")  # 64
        self.encoder2 = conv_block(features, features * 2, down=True, use_batch=True, act="leaky_relu")  # 128
        self.encoder3 = conv_block(features * 2, features * 4, down=True, use_batch=True, act="leaky_relu")  # 256
        self.encoder4 = conv_block(features * 4, features * 8, down=True, use_batch=True, act="leaky_relu")  # 512
        self.encoder5 = conv_block(features * 8, features * 8, down=True, use_batch=True, act="leaky_relu")  # 512
        self.encoder6 = conv_block(features * 8, features * 8, down=True, use_batch=True, act="leaky_relu")  # 512
        self.encoder7 = conv_block(features * 8, features * 8, down=True, use_batch=True, act="leaky_relu")  # 512
        
        # Bottleneck
        self.bottleneck = conv_block(features * 8, features * 8, down=True, use_batch=False, act="leaky_relu")  # 512, 1x1 output for 256x256 input
        
        # Decoder (upsampling path with skip connections)
        # Decoder 1: Input from bottleneck (8F), Output 8F, Upsamples 1x1 -> 2x2
        self.decoder1 = conv_block(features * 8, features * 8, down=False, use_batch=True, act="relu")
        # Decoder 2: Input cat(dec1, enc7) (16F), Output 8F, Upsamples 2x2 -> 4x4
        self.decoder2 = conv_block(features * 16, features * 8, down=False, use_batch=True, act="relu")
        # Decoder 3: Input cat(dec2, enc6) (16F), Output 8F, Upsamples 4x4 -> 8x8
        self.decoder3 = conv_block(features * 16, features * 8, down=False, use_batch=True, act="relu")
        # Decoder 4: Input cat(dec3, enc5) (16F), Output 8F, Upsamples 8x8 -> 16x16
        self.decoder4 = conv_block(features * 16, features * 8, down=False, use_batch=True, act="relu")
        # Decoder 5: Input cat(dec4, enc4) (16F), Output 4F, Upsamples 16x16 -> 32x32
        self.decoder5 = conv_block(features * 16, features * 4, down=False, use_batch=True, act="relu")
        # Decoder 6: Input cat(dec5, enc3) (8F), Output 2F, Upsamples 32x32 -> 64x64
        self.decoder6 = conv_block(features * 8, features * 2, down=False, use_batch=True, act="relu")
        # Decoder 7: Input cat(dec6, enc2) (4F), Output F, Upsamples 64x64 -> 128x128
        self.decoder7 = conv_block(features * 4, features, down=False, use_batch=True, act="relu")
        # Decoder 8: Input cat(dec7, enc1) (2F), Output F, Upsamples 128x128 -> 256x256
        self.decoder8 = conv_block(features * 2, features, down=False, use_batch=True, act="relu")
        
        # Final layer: Takes output of decoder8 (F channels, 256x256)
        # Outputs 3 channels (RGB) with a 1x1 convolution
        self.final_conv = nn.Conv2d(features, 3, kernel_size=1)
        self.final_act = nn.Tanh()

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        enc6 = self.encoder6(enc5)
        enc7 = self.encoder7(enc6)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc7)
        
        # Decoder with skip connections
        dec1 = self.decoder1(bottleneck)
        dec1 = torch.cat([dec1, enc7], dim=1)
        
        dec2 = self.decoder2(dec1)
        dec2 = torch.cat([dec2, enc6], dim=1)
        
        dec3 = self.decoder3(dec2)
        dec3 = torch.cat([dec3, enc5], dim=1)
        
        dec4 = self.decoder4(dec3)
        dec4 = torch.cat([dec4, enc4], dim=1)
        
        dec5 = self.decoder5(dec4)
        dec5 = torch.cat([dec5, enc3], dim=1)
        
        dec6 = self.decoder6(dec5)
        dec6 = torch.cat([dec6, enc2], dim=1)
        
        dec7 = self.decoder7(dec6)
        dec7 = torch.cat([dec7, enc1], dim=1) # Output: 2*features channels, 128x128
        
        dec8 = self.decoder8(dec7) # Output: features channels, 256x256
        
        # Final convolution and activation
        output = self.final_conv(dec8) # Input: features channels, Output: 3 channels
        output = self.final_act(output)
        
        return output


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6, features=64):
        """
        PatchGAN discriminator for image-to-image translation.
        
        Args:
            in_channels (int): Number of input channels (concatenated input & target)
            features (int): Initial number of features
        """
        super(PatchDiscriminator, self).__init__()
        
        # Layer 1: no batch norm
        self.layer1 = conv_block(in_channels, features, down=True, use_batch=False, act="leaky_relu")  # 64, stride 2
        
        # Layer 2
        self.layer2 = conv_block(features, features * 2, down=True, use_batch=True, act="leaky_relu")  # 128, stride 2
        
        # Layer 3
        self.layer3 = conv_block(features * 2, features * 4, down=True, use_batch=True, act="leaky_relu")  # 256, stride 2
        
        # Layer 4: stride 1
        self.layer4 = nn.Sequential(
            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 512, stride 1
        
        # Layer 5: stride 1, output layer
        self.layer5 = nn.Sequential(
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )  # 1, stride 1

    def forward(self, x, y):
        # Concatenate input and target images along channel dimension
        combined = torch.cat([x, y], dim=1)
        
        # Forward pass through layers
        x = self.layer1(combined)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        return x  # Returns 30x30 patch map for 256x256 input 