# Cycle GAN project
A study project to make PyTorch implementation for unpaired image-to-image translation based on:<br>
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

A notebook was created to go through the original article step by step:<br>
https://github.com/vitaly-maximov/cycle_gan/blob/main/CycleGAN.ipynb

An attempt to reproduce horse2zebra results was made:
![image](https://user-images.githubusercontent.com/2083367/176541933-e9304a35-933a-4fac-8c6e-89ed4f208f46.png)

More images at:
https://github.com/vitaly-maximov/cycle_gan/tree/main/horse2zebra/best-epochs

Also a dataset with Rerih images and random pencil drawings of mountains was prepared to make a drawing look like a Rerih masterpiece:
![image](https://user-images.githubusercontent.com/2083367/176544829-323bcfda-d82e-4e3a-a627-0226e821211b.png)

There is a live demo: https://vitaly-maximov.github.io/cycle_gan/ <br>
![image](https://user-images.githubusercontent.com/2083367/176633246-9f801d49-2a11-4e62-9345-dcc9b5a1692a.png)

## Getting Started
### Installation
<ul>
  <li>Clone this repo:<br>
    <pre><code>git clone https://github.com/vitaly-maximov/cycle_gan.git
cd cycle_gan</code></pre>
  </li>
  <li>Create virtual environment with all dependencies:<br>
    <pre><code>python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt</code></pre>
  </li>
  <li>
    Download <a href="https://drive.google.com/drive/folders/1wTombzPmc7qz3QR-q1oyeTMxuHiieXAn?usp=sharing">datasets and models</a>
  </li>
</ul>

### CycleGAN training
<ul>
  <li>Use "src/train.py" script with a configuration (<a href="https://github.com/vitaly-maximov/cycle_gan/blob/main/horse2zebra/u-net.json">example.json</a>):<br>
    <pre><code>python ./src/train.py config.json</code></pre>
  </li>
  <li>Configuration json contains:
    <ul>
      <li><b>"x-path"</b>: a path to x-images</li>
      <li><b>"y-path"</b>: a path to y-images</li>
      <li><b>"extension"</b>: extension of the images (e.g. '*.jpg')</li>
      <li><b>"batch-size"</b>: size of a batch</li>
      <li><b>"generator"</b>: type of generator ('u-net', 'res-net' or 'mix')</li>      
      <li><b>"lambda"</b>: a weight of cycle loss (see the <a href="https://arxiv.org/pdf/1703.10593.pdf">article</a>)</li>
      <li><b>"dropout"</b>: to add dropout modules</li>
      <li><b>"output-images"</b>: a path to store sample images for each epoch</li>
      <li><b>"output-models"</b>: a path to store models for each epoch</li>
      <li><b>"continue"</b>: to continue training from the last model</li>
      <li><b>"preserve"</b>: how many models should be preserved during training</li>
      <li><b>"epochs"</b>: how many epochs to train</li>
    </ul>
  </li>
  <li>To ignore sporadic cuda errors (e.g. out of memory) and continue training, "repeat.bat" script can be used:<br>
    <pre><code>../scripts/repeat.bat "python ..\src\train.py u-net.json" 10</code></pre>
  </li>
  <li>There are configurations and scripts to run training for horse2zebra and mountain2rerih <a href="https://drive.google.com/drive/folders/1HfbSZFPPZAZdgMvkxDnl7T-VBj426eA-?usp=sharing">datasets</a> (make sure local paths are correct):<br>
    <pre><code>./u-net.bat</code></pre>    
  </li>
</ul>

### Inference
<ul>
  <li>Use "src/test.py" script with a configuration and paths to input and output images, e.g.:
    <pre><code>python ../src/test.py u-net.json horse.jpg zebra.jpg</code></pre>
  </li>
  <li>There is an option <b>--y2x</b> to make y2x transformation</li>
  <li>Script uses last trained model from <b>"output-models"</b> path of the configuration</li>
  <li>There are pretrained horse2zebra and mountain2rerih <a href="https://drive.google.com/drive/folders/1ZIsAoyIbJQ9olDfey6LFwUpsAXYD6ExE?usp=sharing">models</a></li>
</ul>

### Onnx
<ul>
  <li>Use "src/export.py" script with a configuration and output path:
    <pre><code>python ../src/export.py u-net.json model.onnx</code></pre>
  </li>
  <li>Onnx model was used in a live <a href="https://vitaly-maximov.github.io/cycle_gan/">demo</a></li>
</ul>
