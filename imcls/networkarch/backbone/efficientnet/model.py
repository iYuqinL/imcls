import torch
from torch import nn
from torch.nn import functional as F
from .effi_utils import (
    relu_fn,
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
)
from ....nn_module import ChannelAttention, SpatialAttention

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = relu_fn(self._bn0(self._expand_conv(inputs)))
        x = relu_fn(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)
        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon
        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
  
        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )
            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))
        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self.feat_dim = out_channels
        # Final linear layer
        # self._dropout = self._global_params.dropout_rate
        # self._fc = nn.Linear(out_channels, self._global_params.num_classes)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = relu_fn(self._bn0(self._conv_stem(inputs)))
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        # Head
        x = relu_fn(self._bn1(self._conv_head(x)))
        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        # Convolution layers
        x = self.extract_features(inputs)
        # Pooling and final linear layer
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(len(x.shape)-1)
        x = x.squeeze(len(x.shape)-1)
        # if self._dropout:
        #     x = F.dropout(x, p=self._dropout, training=self.training)
        # x = self._fc(x)
        return x

    def freeze_blocks(self, block_begin=0, block_end=0):
        if block_begin < 0 or block_end < 0:
            print("num_blocks should not be less than 0, if it is, then freeze no block")
            return
        self._conv_stem.eval()
        self._conv_stem.weight.requires_grad = False
        self._bn0.eval()
        self._bn0.weight.requires_grad = False
        self._bn0.bias.requires_grad = False
        block_cnt = 0
        paras_cnt = 0
        for i in range(block_begin, min(block_end, len(self._blocks))):
            self._blocks[i].eval()
            block_cnt += 1
            for name, p in self._blocks[i].named_parameters():
                if p.requires_grad is True:
                    p.requires_grad = False
                    paras_cnt += 1
        print("freezed %d blocks with %d paras" % (block_cnt, paras_cnt))
        return

    def unfreeze_blocks(self, block_begin=0, block_end=0):
        if block_end < 0 or block_begin < 0:
            print("num_blocks should not be less than 0, if it is, then unfreeze no block")
            return
        block_cnt = 0
        paras_cnt = 0
        for i in range(min(block_end, len(self._blocks))-1, block_begin-1, -1):
            self._blocks[i].train()
            block_cnt += 1
            for name, p in self._blocks[i].named_parameters():
                if p.requires_grad is False:
                    p.requires_grad = True
                    paras_cnt += 1
        if i == 0:
            self._conv_stem.train()
            self._conv_stem.weight.requires_grad = True
            self._bn0.train()
            self._bn0.weight.requires_grad = True
            self._bn0.bias.requires_grad = True
        print("unfreezed %d blocks with %d paras" % (block_cnt, paras_cnt))
        return

    def freeze_blocks_by_id(self, blocks_ids):
        block_cnt = 0
        paras_cnt = 0
        for idx in blocks_ids:
            self._blocks[idx].eval()
            block_cnt += 1
            for name, p in self._blocks[idx].named_parameters():
                if p.requires_grad is True:
                    p.requires_grad = False
                    paras_cnt += 1
        print("freezed %d blocks with %d paras" % (block_cnt, paras_cnt))
        return

    def unfreeze_blocks_by_id(self, blocks_ids):
        block_cnt = 0
        paras_cnt = 0
        for idx in blocks_ids:
            self._blocks[idx].train()
            block_cnt += 1
            for name, p in self._blocks[idx].named_parameters():
                if p.requires_grad is False:
                    p.requires_grad = True
                    paras_cnt += 1
        print("freezed %d blocks with %d paras" % (block_cnt, paras_cnt))
        return

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return EfficientNet(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000):
        model = EfficientNet.from_name(model_name, override_params={'num_classes': num_classes})
        load_pretrained_weights(model, model_name)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ['efficientnet_b'+str(i) for i in range(num_models)]
        if model_name.replace('-', '_') not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))


def efficientnet_b0(pretrained=False, **kwargs):
    if pretrained:
        return EfficientNet.from_pretrained("efficientnet-b0", **kwargs)
    return EfficientNet.from_name("efficientnet-b0", **kwargs)

def efficientnet_b1(pretrained=False, **kwargs):
    if pretrained:
        return EfficientNet.from_pretrained("efficientnet-b1", **kwargs)
    return EfficientNet.from_name("efficientnet-b1", **kwargs)

def efficientnet_b2(pretrained=False, **kwargs):
    if pretrained:
        return EfficientNet.from_pretrained("efficientnet-b2", **kwargs)
    return EfficientNet.from_name("efficientnet-b2", **kwargs)

def efficientnet_b3(pretrained=False, **kwargs):
    if pretrained:
        return EfficientNet.from_pretrained("efficientnet-b3", **kwargs)
    return EfficientNet.from_name("efficientnet-b3", **kwargs)

def efficientnet_b4(pretrained=False, **kwargs):
    if pretrained:
        return EfficientNet.from_pretrained("efficientnet-b4", **kwargs)
    return EfficientNet.from_name("efficientnet-b4", **kwargs)

def efficientnet_b5(pretrained=False, **kwargs):
    if pretrained:
        return EfficientNet.from_pretrained("efficientnet-b5", **kwargs)
    return EfficientNet.from_name("efficientnet-b5", **kwargs)

def efficientnet_b6(pretrained=False, **kwargs):
    if pretrained:
        return EfficientNet.from_pretrained("efficientnet-b6", **kwargs)
    return EfficientNet.from_name("efficientnet-b6", **kwargs)

def efficientnet_b7(pretrained=False, **kwargs):
    if pretrained:
        return EfficientNet.from_pretrained("efficientnet-b7", **kwargs)
    return EfficientNet.from_name("efficientnet-b7", **kwargs)

if __name__ == "__main__":
    ca = ChannelAttention(32)
    print("ChannelAttention")
    for var_name in ca.state_dict():
        print(var_name, "\t", ca.state_dict()[var_name].size())
    print("SpatialAttention:")
    sa = SpatialAttention()
    for var_name in sa.state_dict():
        print(var_name, "\t", sa.state_dict()[var_name].size())
    print("efficientnet:")
    effinet = EfficientNet.from_name('efficientnet-b1', override_params={'num_classes': 9}, ifcbam=True)
    effinet_state_dict = effinet.state_dict()
    effinet_state_dict.pop('_ca.fc1.weight')
    for var_name in effinet_state_dict:
        print(var_name, "\t", effinet.state_dict()[var_name].size())
