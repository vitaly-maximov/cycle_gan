from nets.cycle_gan import CycleGan

def parse_bool(item):
    return (item.lower() == 'true')

def cycle_gan(config, device):
    return CycleGan(config['generator'], config['lambda'], parse_bool(config['dropout']), device)

def last_model(models_dir):
    models = models_dir.rglob('*.pt')
    models = filter(lambda x: x.name[:-3].isdigit(), models)
    models = sorted(models, key=lambda x: int(x.name[:-3]))
    return models[-1] if models else None