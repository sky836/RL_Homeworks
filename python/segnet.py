import torch
import torch.nn as nn
import torch.nn.functional as F

class SegNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(SegNet, self).__init__()
        
        # Encoder (VGG16 based)
        self.enc1 = self._encoder_block(in_channels, 64, 2)
        self.enc2 = self._encoder_block(64, 128, 2)
        self.enc3 = self._encoder_block(128, 256, 3)
        self.enc4 = self._encoder_block(256, 512, 3)
        self.enc5 = self._encoder_block(512, 512, 3)
        
        # Decoder
        self.dec5 = self._decoder_block(512, 512, 3)
        self.dec4 = self._decoder_block(512, 256, 3)
        self.dec3 = self._decoder_block(256, 128, 3)
        self.dec2 = self._decoder_block(128, 64, 2)
        self.dec1 = self._decoder_block(64, num_classes, 2)
        
        # Initialize weights
        self._initialize_weights()
        
    def _encoder_block(self, in_channels, out_channels, num_layers):
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            else:
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
        return nn.Sequential(*layers)
    
    def _decoder_block(self, in_channels, out_channels, num_layers):
        layers = []
        layers.append(nn.MaxUnpool2d(kernel_size=2, stride=2))
        for i in range(num_layers):
            if i == num_layers - 1:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            else:
                layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            if i < num_layers - 1:
                layers.append(nn.BatchNorm2d(in_channels))
                layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder
        x1, idx1 = self.enc1(x)
        x2, idx2 = self.enc2(x1)
        x3, idx3 = self.enc3(x2)
        x4, idx4 = self.enc4(x3)
        x5, idx5 = self.enc5(x4)
        
        # Decoder
        x = self.dec5(x5, idx5)
        x = self.dec4(x, idx4)
        x = self.dec3(x, idx3)
        x = self.dec2(x, idx2)
        x = self.dec1(x, idx1)
        
        return x

def test_segnet():
    # Create a sample input
    batch_size = 2
    channels = 3
    height = 256
    width = 256
    x = torch.randn(batch_size, channels, height, width)
    
    # Create model
    model = SegNet(in_channels=channels, num_classes=2)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
if __name__ == "__main__":
    test_segnet() 