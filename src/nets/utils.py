import torch.nn as nn

lr = 0.0002
betas = (0.5, 0.999)

def conv_block(
    isize, osize, kernel, stride, padding, 
    activation=None, 
    normalize=True, 
    dropout=False, 
    transpose=False, 
    reflect=False):
    
    modules = []
    
    if (reflect):
        modules.append(nn.ReflectionPad2d(padding))
        padding = 0
    
    if (transpose):
        modules.append(nn.ConvTranspose2d(isize, osize, kernel_size=kernel, stride=stride, padding=padding, bias=normalize))
    else:
        modules.append(nn.Conv2d(isize, osize, kernel_size=kernel, stride=stride, padding=padding, bias=normalize))
    
    if (normalize):
        modules.append(nn.InstanceNorm2d(osize, affine=False, track_running_stats=False))
    
    if (dropout):
        modules.append(nn.Dropout(p=0.5, inplace=True))

    if (activation == 'leaky'):
        modules.append(nn.LeakyReLU(0.2, inplace=True))        
    elif (activation == 'relu'):
        modules.append(nn.ReLU(inplace=True))    
    elif (activation == 'tanh'):
        modules.append(nn.Tanh())
    
    return nn.Sequential(*modules)

def conv_block_4_2(isize, osize, **kwargs):
    return conv_block(isize, osize, 4, 2, 1, **kwargs)

def conv_block_3_1(isize, osize, **kwargs):
    return conv_block(isize, osize, 3, 1, 1, **kwargs)

def conv_transpose_block_4_2(isize, osize, **kwargs):
    return conv_block(isize, osize, 4, 2, 1, transpose=True, **kwargs)

def conv_block_7_1(isize, osize, **kwargs):
    return conv_block(isize, osize, 7, 1, 3, **kwargs)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)