from openvino.tools.pot import IEEngine
from openvino.tools.pot import load_model, save_model
from openvino.tools.pot import compress_model_weights
from openvino.tools.pot import create_pipeline

import os
import os.path as osp
from imageloader import ImageLoader

# mdl_pre_dir = r'C:/Users/43597/data1/task/deploy_tools/openvino/open_model_zoo/models/public/alexnet'
# mdl_name = r'alexnet'

# mdl_pre_dir = r'C:/Users/43597/data1/task/deploy_tools/openvino/open_model_zoo/models/public/repvgg-a0'
# mdl_name = r'repvgg-a0'

mdl_pre_dir = r'C:/Users/43597/data1/task/deploy_tools/openvino/open_model_zoo/models/public/ssdlite_mobilenet_v2'
mdl_name = r'ssdlite_mobilenet_v2'

# Model config specifies the model name and paths to model .xml and .bin file
model_config = {
    "model_name": "model",
    "model": osp.join(mdl_pre_dir, f'FP32/{mdl_name}.xml'),
    "weights": osp.join(mdl_pre_dir, f'FP32/{mdl_name}.bin'),
}

# Engine config
engine_config = {"device": "CPU"}

algorithms = [
    {
        "name": "DefaultQuantization",
        "params": {
            "target_device": "ANY",
            "stat_subset_size": 300
        },
    }
]

# Step 1: Implement and create user's data loader
data_loader = ImageLoader("./")

# Step 2: Load model
model = load_model(model_config=model_config)

# Step 3: Initialize the engine for metric calculation and statistics collection.
engine = IEEngine(config=engine_config, data_loader=data_loader)

# Step 4: Create a pipeline of compression algorithms and run it.
pipeline = create_pipeline(algorithms, engine)
compressed_model = pipeline.run(model=model)

# Step 5 (Optional): Compress model weights to quantized precision
#                     to reduce the size of the final .bin file.
compress_model_weights(compressed_model)

# Step 6: Save the compressed model to the desired path.
# Set save_path to the directory where the model should be saved
compressed_model_paths = save_model(
    model=compressed_model,
    save_path=osp.join(mdl_pre_dir, 'MY_INT8'),
    model_name=mdl_name,
)

print(f'quant done!')