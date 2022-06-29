let onnx_session = null;

async function load_model_async() {
    const message = document.getElementById('message');

    message.innerHTML = "Loading model...";

    onnx_session = await ort.InferenceSession.create(
        "./model.onnx", 
        { executionProviders: ['wasm'] });

    message.innerHTML = "Draw or paste an image of a mountain:";
}

async function process(args) {
    const status = document.getElementById('status');
    const output = document.getElementById('output');
    
    status.innerHTML = "Processing...";

    const width = 128;
    const height = 128;

    var bitmap = await createImageBitmap(args.image.asBlob('jpg', 100));
    var canvas = document.createElement('canvas');

    canvas.width = width;
    canvas.height = height;

    var context = canvas.getContext('2d');
    context.drawImage(bitmap, 0, 0, width, height);

    const inputData = context.getImageData(0, 0, width, height).data;

    const inputTensor = new ort.Tensor('float32', new Float32Array(inputData), [width, height, 4]);
    const feeds = {}
    feeds[onnx_session.inputNames[0]] = inputTensor;

    const result = await onnx_session.run(feeds);
    const buffer = new Uint8ClampedArray(result[onnx_session.outputNames[0]].data)
    
    const outputData = context.createImageData(width, height);
    outputData.data.set(buffer);

    context.putImageData(outputData, 0, 0);
    output.src = canvas.toDataURL();
       
    status.innerHTML = "";
}

document.addEventListener("DOMContentLoaded", async () => {
    await load_model_async();

    document.getElementById('arrow').style.visibility = 'visible';

    let painterro = Painterro({
        id: 'painterro',
        activeColor: '#000000',
        activeFillColor: '#ffffff',
        activeFillColorAlpha: 1,
        defaultPrimitiveShadowOn: false,
        defaultSize: '256x256',
        defaultTool: 'brush',
        how_to_paste_actions: 'replace_all',
        hiddenTools: [
           'crop', 'arrow', 'rect', 'ellipse', 'text', 'rotate', 'resize', 'save', 'open', 
           'close', 'zoomin', 'zoomout', 'select', 'pixelize', 'undo', 'redo', 'settings'],
        onChange: process});

    painterro.show("mountain.jpg");
});