import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """改进的通道注意力模块"""
    def __init__(self, num_features, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 使用更合理的reduction ratio
        self.fc = nn.Sequential(
            nn.Conv2d(num_features, num_features // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, 1, bias=False)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return torch.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)


class CBAM(nn.Module):
    """卷积块注意力模块 (Convolutional Block Attention Module)"""
    def __init__(self, num_features, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(num_features, reduction)
        self.spatial_attention = SpatialAttention()
        
    def forward(self, x):
        # 通道注意力
        x = x * self.channel_attention(x)
        # 空间注意力
        x = x * self.spatial_attention(x)
        return x


class ResidualBlock(nn.Module):
    """改进的残差块"""
    def __init__(self, num_features, use_attention=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.prelu = nn.PReLU(num_parameters=num_features)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = CBAM(num_features)
        
    def forward(self, x):
        residual = x
        out = self.prelu(self.conv1(x))
        out = self.conv2(out)
        
        if self.use_attention:
            out = self.attention(out)
        
        out = out + residual
        return self.prelu(out)


class ImprovedGenerator(nn.Module):
    """改进的生成器"""
    def __init__(self, input_channels, num_features, output_channels, num_blocks, scale):
        super(ImprovedGenerator, self).__init__()
        
        self.scale = scale
        self.output_channels = output_channels
        
        # 输入特征提取
        self.input_conv = nn.Conv2d(input_channels, num_features, 3, padding=1)
        self.input_prelu = nn.PReLU(num_parameters=num_features)
        
        # 残差块序列
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(num_features, use_attention=(i % 2 == 0))
            for i in range(num_blocks)
        ])
        
        # 特征融合
        self.feature_fusion = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.fusion_prelu = nn.PReLU(num_parameters=num_features)
        
        # 输出层 - 统一处理所有通道
        self.output_conv = nn.Conv2d(num_features, output_channels * scale * scale, 3, padding=1)
        
    def forward(self, x):
        # 输入特征提取
        feat = self.input_prelu(self.input_conv(x))
        
        # 残差块处理
        res_feat = self.residual_blocks(feat)
        
        # 特征融合
        fused = self.fusion_prelu(self.feature_fusion(res_feat))
        fused = fused + feat  # 全局残差连接
        
        # 输出生成
        output = self.output_conv(fused)
        output = torch.tanh(output) * 0.58 + 0.5
        
        # 上采样
        output = F.pixel_shuffle(output, self.scale)
        
        return output


class ImprovedMicroISP(nn.Module):
    """改进的MicroISP模型"""
    def __init__(self, 
                 num_features=32,  # 增加特征维度
                 input_channels=4,
                 output_channels=3,
                 num_blocks=8,     # 增加块数
                 scale=2):
        super(ImprovedMicroISP, self).__init__()
        
        # 第一阶段：粗处理
        self.coarse_generator = ImprovedGenerator(
            input_channels, num_features, output_channels, num_blocks, scale
        )
        
        # 第二阶段：细化处理
        self.refine_generator = ImprovedGenerator(
            output_channels, num_features // 2, output_channels, num_blocks // 2, 1
        )
        
        # 可选的色彩校正模块
        self.color_correction = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, 1),
            nn.PReLU(num_parameters=output_channels)
        )
        
    def forward(self, x):
        # 第一阶段：粗处理
        coarse_output = self.coarse_generator(x)
        
        # 第二阶段：细化处理
        refined_output = self.refine_generator(coarse_output)
        
        # 残差连接
        final_output = coarse_output + refined_output
        
        # 色彩校正
        final_output = self.color_correction(final_output)
        
        # 确保输出在合理范围内
        final_output = torch.clamp(final_output, 0, 1)
        
        return final_output


class LightweightMicroISP(nn.Module):
    """轻量级版本的MicroISP"""
    def __init__(self, 
                 num_features=16,
                 input_channels=4,
                 output_channels=3,
                 num_blocks=4,
                 scale=2):
        super(LightweightMicroISP, self).__init__()
        
        # 深度可分离卷积块
        self.depthwise_conv = nn.Conv2d(input_channels, input_channels, 3, padding=1, groups=input_channels)
        self.pointwise_conv = nn.Conv2d(input_channels, num_features, 1)
        self.prelu = nn.PReLU(num_parameters=num_features)
        
        # 轻量级残差块
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_features, num_features, 3, padding=1, groups=num_features),
                nn.Conv2d(num_features, num_features, 1),
                nn.PReLU(num_parameters=num_features)
            ) for _ in range(num_blocks)
        ])
        
        # 输出层
        self.output_conv = nn.Conv2d(num_features, output_channels * scale * scale, 3, padding=1)
        
    def forward(self, x):
        # 深度可分离卷积
        x = self.prelu(self.pointwise_conv(self.depthwise_conv(x)))
        
        # 残差块
        for block in self.blocks:
            residual = x
            x = block(x) + residual
        
        # 输出生成
        output = self.output_conv(x)
        output = torch.tanh(output) * 0.58 + 0.5
        output = F.pixel_shuffle(output, 2)
        
        return torch.clamp(output, 0, 1)


def compare_models():
    """比较不同模型的参数量和性能"""
    from model import MicroISP
    models = {
        'Original': lambda: MicroISP(num_features=4, input_channels=4, output_channels=3, num_blocks=7, res_skip=3, scale=2),
        'Improved': lambda: ImprovedMicroISP(num_features=32, input_channels=4, output_channels=3, num_blocks=8, scale=2),
        'Lightweight': lambda: LightweightMicroISP(num_features=16, input_channels=4, output_channels=3, num_blocks=4, scale=2)
    }
    
    print("=== 模型比较 ===")
    for name, model_fn in models.items():
        try:
            model = model_fn()
            total_params = sum(p.numel() for p in model.parameters())
            print(f"{name}: {total_params:,} 参数")
            
            # 测试前向传播
            with torch.no_grad():
                x = torch.randn(1, 4, 256, 256)
                output = model(x)
                print(f"  输出形状: {output.shape}")
                
        except Exception as e:
            print(f"{name}: 错误 - {e}")


if __name__ == "__main__":
    compare_models() 