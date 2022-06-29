import sys
import json
from pathlib import Path

import utils

def main():
    try:
        with open(sys.argv[1]) as json_file:
            config = json.load(json_file)

        output_file = Path(sys.argv[2])
        output_file.parents[0].mkdir(exist_ok=True, parents=True)

        y2x = False
        if (len(sys.argv) > 3):
            y2x = (sys.argv[3] == '--y2x')
    except:
        print("Usage: python export.py config.json model.onnx [--y2x]")
        return
    
    cycleGan = utils.cycle_gan(config, 'cpu')

    config_dir = Path(sys.argv[1]).parents[0]
    models_dir = Path(config_dir, config['output-models'])

    model = utils.last_model(models_dir)
    if model is None:
        print("No trained models found")
        return

    cycleGan.load(model)
    
    f = cycleGan.export_y2x if y2x else cycleGan.export_x2y
    f(output_file)

if (__name__ == "__main__"):
    main()