import torch
import torch.nn as nn

from .discriminator import Discriminator
from .u_net_generator import UNetGenerator
from .res_net_generator import ResNetGenerator
from .mix_generator import MixGenerator

class CycleGan():
    def __init__(self, generator, lm=10, dropout=False, device='cpu'):
        def create_generator():
            if (generator == 'u-net'):
                return UNetGenerator(dropout=dropout)
            if (generator == 'res-net'):
                return ResNetGenerator(dropout=dropout)
            if (generator == 'mix'):
                return MixGenerator(dropout=dropout)
            raise(RuntimeError(f"Unknown generator '{generator}': use 'u-net', 'res-net' or 'mix'."))

        self._generator_x2y = create_generator().to(device)
        self._generator_y2x = create_generator().to(device)

        self._discriminator_x = Discriminator().to(device)
        self._discriminator_y = Discriminator().to(device)

        self._device = device
        self._lm = lm

    def x2y(self, x):
        return self._generate(self._generator_x2y, x)

    def y2x(self, y):
        return self._generate(self._generator_y2x, y)

    def step(self, x, y):
        self._train()

        loss_d_x = self._discriminator_step(self._discriminator_x, self._generator_y2x, y, x)
        loss_d_y = self._discriminator_step(self._discriminator_y, self._generator_x2y, x, y)
        
        loss_g_x2y = self._generator_step(self._generator_x2y, self._generator_y2x, self._discriminator_y, x, y)
        loss_g_y2x = self._generator_step(self._generator_y2x, self._generator_x2y, self._discriminator_x, y, x)

        return (loss_d_x, loss_d_y, loss_g_x2y, loss_g_y2x)
    
    def save(self, path):
        checkpoint = {
            "g_x2y": self._generator_x2y.state_dict(),
            "g_x2y_optimizer": self._generator_x2y.optimizer().state_dict(),
            "g_y2x": self._generator_y2x.state_dict(),
            "g_y2x_optimizer": self._generator_y2x.optimizer().state_dict(),
            "d_x": self._discriminator_x.state_dict(),
            "d_x_optimizer": self._discriminator_x.optimizer().state_dict(),
            "d_y": self._discriminator_y.state_dict(),
            "d_y_optimizer": self._discriminator_y.optimizer().state_dict()
        }    
        torch.save(checkpoint, path)
    
    def load(self, path):
        checkpoint = torch.load(path)

        self._generator_x2y.load_state_dict(checkpoint["g_x2y"])
        self._generator_x2y.optimizer().load_state_dict(checkpoint["g_x2y_optimizer"])
        self._generator_y2x.load_state_dict(checkpoint["g_y2x"])
        self._generator_y2x.optimizer().load_state_dict(checkpoint["g_y2x_optimizer"])
        self._discriminator_x.load_state_dict(checkpoint["d_x"])
        self._discriminator_x.optimizer().load_state_dict(checkpoint["d_x_optimizer"])
        self._discriminator_y.load_state_dict(checkpoint["d_y"])
        self._discriminator_y.optimizer().load_state_dict(checkpoint["d_y_optimizer"])
    
    def _generate(self, generator, source):
        with torch.no_grad():
            generator.eval()
            return generator(source)
    
    def _train(self):
        self._generator_x2y.train()
        self._generator_y2x.train()
        self._discriminator_x.train()
        self._discriminator_y.train()

    def _discriminator_step(self, discriminator, forward, image, target):
        discriminator.zero_grad()
        
        real = discriminator(target)
        fake = discriminator(forward(image).detach())
        
        loss = self._discriminator_loss(fake, real)
        loss.backward()
        
        discriminator.optimizer().step()
        
        del real, fake
        torch.cuda.empty_cache()
        
        return loss.item()

    def _generator_step(self, forward, backward, discriminator, image, target):
        forward.zero_grad()
        
        same = forward(target)
        identity_loss = self._similarity_loss(same, target)
        
        fake = forward(image)
        generation_loss = self._generator_loss(discriminator(fake))
        
        cycle = backward(fake)
        cycle_loss = self._similarity_loss(cycle, image)
        
        loss = generation_loss + self._lm * (0.5 * identity_loss + cycle_loss)
        loss.backward()
        
        forward.optimizer().step()
        
        del same, fake, cycle
        torch.cuda.empty_cache()
        
        return loss.item()

    def _discriminator_loss(self, fake, real):
        criterion = nn.functional.binary_cross_entropy
        
        one = torch.rand(real.shape, device=self._device) * 0.3 + 0.7
        zero = torch.rand(real.shape, device=self._device) * 0.3
        
        return 0.5 * (criterion(real, one) + criterion(fake, zero))
    
    def _generator_loss(self, fake):
        criterion = nn.functional.binary_cross_entropy
        return criterion(fake, torch.ones(fake.shape).to(self._device))
    
    def _similarity_loss(self, fake, real):
        criterion = nn.functional.l1_loss
        return criterion(fake, real)