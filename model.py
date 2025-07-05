import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModule(nn.Module):
    """注意力模块"""
    def __init__(self, num_features, hidden_units, attention_dims=4):
        super(AttentionModule, self).__init__()
        
        # 第一个1x1卷积，步长为3 (保持原始设计)
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=1, stride=3, padding=0)
        self.prelu1 = nn.PReLU(num_parameters=num_features)
        
        # 连续的3x3卷积，步长为3 (保持原始设计)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=3, padding=0)
        self.prelu2 = nn.PReLU(num_parameters=num_features)
        
        self.conv3 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=3, padding=0)
        self.prelu3 = nn.PReLU(num_parameters=num_features)
        
        self.conv4 = nn.Conv2d(num_features, num_features, kernel_size=3, stride=3, padding=0)
        self.prelu4 = nn.PReLU(num_parameters=num_features)
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 最后两个1x1卷积
        self.conv5 = nn.Conv2d(num_features, hidden_units, kernel_size=1)
        self.prelu5 = nn.PReLU(num_parameters=hidden_units)
        
        self.conv6 = nn.Conv2d(hidden_units, attention_dims, kernel_size=1)
        
    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        x = self.prelu4(self.conv4(x))
        
        x = self.global_avg_pool(x)
        
        x = self.prelu5(self.conv5(x))
        x = torch.sigmoid(self.conv6(x))
        
        return x


class Generator(nn.Module):
    """生成器模块"""
    def __init__(self, input_channels, num_features, output_channels, num_blocks, res_skip, scale):
        super(Generator, self).__init__()
        
        self.num_blocks = num_blocks
        self.res_skip = res_skip
        self.output_channels = output_channels
        self.scale = scale
        
        # 为每个输出通道创建卷积层
        self.conv_blocks = nn.ModuleDict()
        self.attention_modules = nn.ModuleDict()
        
        for i in range(output_channels):
            # 每个通道的卷积块
            channel_convs = nn.ModuleList()
            for j in range(num_blocks):
                if j == 0:
                    conv = nn.Conv2d(input_channels, num_features, kernel_size=3, padding=1)
                else:
                    conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
                channel_convs.append(conv)
            
            self.conv_blocks[f'channel_{i}'] = channel_convs
            
            # 输出卷积
            self.conv_blocks[f'channel_{i}_out'] = nn.Conv2d(num_features, scale*scale, kernel_size=3, padding=1)
            
            # 注意力模块
            self.attention_modules[f'channel_{i}'] = AttentionModule(num_features, num_features, attention_dims=4)
        
        # PReLU激活函数
        self.prelu = nn.PReLU(num_parameters=num_features)
        
    def forward(self, x):
        outputs = []
        outputs_raw = []
        
        for i in range(self.output_channels):
            conv = None
            conv_skip = None
            
            # 卷积块
            for j in range(self.num_blocks):
                conv_layer = self.conv_blocks[f'channel_{i}'][j]
                
                if j == 0:
                    conv = self.prelu(conv_layer(x))
                else:
                    conv = self.prelu(conv_layer(conv))
                
                # 残差连接逻辑
                if j % self.res_skip == 0:
                    conv_skip = conv
                elif j % self.res_skip == self.res_skip - 1:
                    # 应用注意力
                    attention = self.attention_modules[f'channel_{i}'](conv)
                    conv = conv * attention
                    conv = conv + conv_skip
            
            # 输出卷积
            conv_out = self.conv_blocks[f'channel_{i}_out'](conv)
            conv_out = torch.tanh(conv_out)
            outputs_raw.append(conv_out)
            
            # 处理输出
            conv_out = conv_out * 0.58 + 0.5
            # depth_to_space操作
            conv_out = F.pixel_shuffle(conv_out, self.scale)
            outputs.append(conv_out)
        
        return outputs, outputs_raw


class MicroISP(nn.Module):
    """MicroISP主网络"""
    def __init__(self, 
                 num_features=4,
                 input_channels=4,
                 output_channels=3,
                 num_blocks=7,
                 res_skip=3,
                 scale=2):
        super(MicroISP, self).__init__()
        
        self.generator1 = Generator(input_channels, num_features, output_channels, 
                                  num_blocks, res_skip, scale)
        self.generator2 = Generator(scale*scale, num_features, output_channels, 
                                  num_blocks, res_skip, scale)
        
    def forward(self, x):
        # 第一个生成器
        outputs_1, outputs_raw = self.generator1(x)
        
        # 第二个生成器，输入是outputs_raw
        outputs_2_list = []
        for i in range(len(outputs_raw)):
            conv = None
            conv_skip = None
            
            # 使用第二个生成器处理每个raw输出
            input_tensor = outputs_raw[i]
            
            for j in range(self.generator2.num_blocks):
                conv_layer = self.generator2.conv_blocks[f'channel_{i}'][j]
                
                if j == 0:
                    conv = self.generator2.prelu(conv_layer(input_tensor))
                else:
                    conv = self.generator2.prelu(conv_layer(conv))
                
                if j % self.generator2.res_skip == 0:
                    conv_skip = conv
                elif j % self.generator2.res_skip == self.generator2.res_skip - 1:
                    attention = self.generator2.attention_modules[f'channel_{i}'](conv)
                    conv = conv * attention
                    conv = conv + conv_skip
            
            # 输出
            conv_out = self.generator2.conv_blocks[f'channel_{i}_out'](conv)
            conv_out = torch.tanh(conv_out) * 0.58 + 0.5
            conv_out = F.pixel_shuffle(conv_out, self.generator2.scale)
            outputs_2_list.append(conv_out)
        
        # 拼接所有输出通道
        result = torch.cat(outputs_2_list, dim=1)
        
        return result


def test_model_shapes():
    """测试模型的输入输出形状"""
    print("=== MicroISP PyTorch 模型形状测试 ===")
    
    # 创建模型
    model = MicroISP(
        num_features=4,
        input_channels=4,
        output_channels=3,
        num_blocks=7,
        res_skip=3,
        scale=2
    )
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 测试不同输入尺寸 (最小128)
    test_sizes = [(1, 4, 128, 128), (1, 4, 256, 256), (2, 4, 512, 512)]
    
    model.eval()
    with torch.no_grad():
        for batch_size, channels, height, width in test_sizes:
            print(f"\n测试输入形状: {(batch_size, channels, height, width)}")
            
            # 创建随机输入
            input_tensor = torch.randn(batch_size, channels, height, width)
            
            try:
                # 前向传播
                output = model(input_tensor)
                
                expected_output_shape = (batch_size, 3, height * 2, width * 2)
                print(f"输出形状: {tuple(output.shape)}")
                print(f"期望形状: {expected_output_shape}")
                print(f"形状匹配: {tuple(output.shape) == expected_output_shape}")
                
                # 检查输出值范围
                print(f"输出值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
                
            except Exception as e:
                print(f"错误: {e}")
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    test_model_shapes() 