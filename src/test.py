import sys
import json
import torch
from pathlib import Path

import utils
from images.utils import save_image
from images.image_dataset import ImageDataset

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        with open(sys.argv[1]) as json_file:
            config = json.load(json_file)

        input_tensor = ImageDataset.to_test_tensor(sys.argv[2]).to(device)

        output_file = Path(sys.argv[3])
        output_file.parents[0].mkdir(exist_ok=True, parents=True)

        y2x = False
        if (len(sys.argv) > 4):
            y2x = (sys.argv[4] == '--y2x')
    except:
        print("Usage: python test.py config.json input.jpg output.jpg [--y2x]")
        return
    
    cycleGan = utils.cycle_gan(config, device)

    config_dir = Path(sys.argv[1]).parents[0]
    models_dir = Path(config_dir, config['output-models'])

    model = utils.last_model(models_dir)
    if model is None:
        print("No trained models found")
        return

    cycleGan.load(model)

    f = cycleGan.y2x if y2x else cycleGan.x2y
    output_tensor = f(input_tensor)

    save_image(output_file, output_tensor)

if (__name__ == "__main__"):
    main()