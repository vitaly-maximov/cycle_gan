import sys
import json
import torch
from pathlib import Path
from tqdm import tqdm

import utils
import images.utils
from images.image_dataset import ImageDataset

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        with open(sys.argv[1]) as json_file:
            config = json.load(json_file)
    except:
        print("Usage: python train.py config.json")
        return
    
    config_dir = Path(sys.argv[1]).parents[0]

    def get_files(key):
        return Path(config_dir, config[key]).rglob(config['extension'])

    x_loader = ImageDataset(get_files('x-path'), config['batch-size'], device).loader()
    y_loader = ImageDataset(get_files('y-path'), config['batch-size'], device).loader()

    # use same sample
    output_images_dir = Path(config_dir, config['output-images'])
    output_images_dir.mkdir(exist_ok=True, parents=True)

    sample_path = Path(output_images_dir, 'sample.bin')
    if sample_path.exists():
        x_sample, y_sample = images.utils.load_sample(sample_path, device)
    else:
        x_sample, y_sample = images.utils.get_sample(x_loader, y_loader)

        images.utils.save_sample(sample_path, x_sample, y_sample)
        images.utils.save_image_grid(Path(output_images_dir, 'sample.jpg'), x_sample, y_sample)
    
    # use last model
    output_models_dir = Path(config_dir, config['output-models'])
    output_models_dir.mkdir(exist_ok=True, parents=True)

    cycleGan = utils.cycle_gan(config, device)

    from_epoch = 1
    if utils.parse_bool(config['continue']):
        model = utils.last_model(output_models_dir)
        if model is not None:
            from_epoch = int(model.name[:-3]) + 1
            cycleGan.load(model)

    # train
    for epoch in range(from_epoch, config['epochs'] + 1):
        print('Epoch', epoch)

        for x, y in tqdm(zip(x_loader, y_loader), total=len(x_loader)):
            if (x.shape == y.shape):
                cycleGan.step(x, y)
            
            del x, y
            torch.cuda.empty_cache()
        
        # checkpoint
        model_path = Path(output_models_dir, f'{epoch}.pt')
        cycleGan.save(model_path)

        ancient_path = Path(output_models_dir, f'{epoch - config["preserve"]}.pt')
        if ancient_path.exists():
            ancient_path.unlink()

        # epoch images
        x_gen, y_gen = cycleGan.y2x(y_sample), cycleGan.x2y(x_sample)
        images.utils.save_image_grid(Path(output_images_dir, f'{epoch}.jpg'), y_gen, x_gen)

        del x_gen, y_gen
        torch.cuda.empty_cache()

if (__name__ == "__main__"):
    main()