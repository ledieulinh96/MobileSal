import numpy as np
from models.model_remove_idr import MobileSal
import torch, os
from time import time
from tqdm import tqdm
import logging

try:
    from torch2trt import torch2trt, TRTModule
    trt_installed = 1
except ModuleNotFoundError:
    print("please install torch2trt for TensorRT speed test!")
    trt_installed = 0
print("loaded all packages")

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

model = MobileSal().to(device).eval()
current_dir = os.getcwd()

log_file= f'speed.log'
destination_path = os.path.join(current_dir,"snapshots", log_file)

# Configure logging
logging.basicConfig(filename=destination_path, filemode='a', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# If you still need to use print and want it logged:
import sys

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        if message != '\n':
            self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
sys.stdout = Logger(log_file)



# We use multi-scale pretrained model as the model weights
#model.load_state_dict(torch.load("pretrained/mobilesal_ms.pth"))
model.load_state_dict(torch.load("pretrained/model_60.pth"))
batch_size = 10
x = torch.randn(batch_size,3,320,320).to(device)
y = torch.randn(batch_size,1,320,320).to(device)
total_params = sum([np.prod(p.size()) for p in model.parameters()])
#depthpred_params = sum([np.prod(p.size()) for p in model.idr.parameters()])
#print('Total network parameters (excluding idr): ' + str(total_params - depthpred_params))
print('Total network parameters (including idr): ' + str(total_params))


######################################
#### PyTorch Test [BatchSize 16] #####
######################################
for i in tqdm(range(50)):
    # warm up
    p = model(x,y)
    p = p + 1

total_t = 0
for i in tqdm(range(100)):
    start = time()
    p = model(x,y)
    p = p + 1 # replace torch.cuda.synchronize()``
    total_t += time() - start

print("FPS", 100 / total_t * batch_size)
print("PyTorch batchsize=20 speed test completed, expected 450FPS for RTX 2080Ti!")

torch.cuda.empty_cache()
sys.stdout.close()
if not trt_installed:
    exit()

######################################
#### TensorRT Test [Batch Size=1] ####
######################################
x = torch.randn(1,3,320,320).cuda()
y = torch.randn(1,1,320,320).cuda()

save_path = "pretrained/model_60.pth"
if os.path.exists(save_path):
    print('loading TensorRT model', save_path)
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(save_path))
else:
    print("converting model to TensorRT format!")
    model_trt = torch2trt(model, [x,y], fp16_mode=False)
    torch.save(model_trt.state_dict(), save_path)

torch.cuda.empty_cache()

with torch.no_grad():
    for i in tqdm(range(50)):
        p = model_trt(x,y)
        p = p + 1

total_t = 0
with torch.no_grad():
    for i in tqdm(range(2000)):
        start = time()
        p = model_trt(x,y)
        p = p + 1 # replace torch.cuda.synchronize()
        total_t += time() - start
print(2000 / total_t)
print("TensorRT batchsize=1 speed test completed, expected 420FPS for RTX 2080Ti!")


######################################
##### TensorRT Test [BS=1, FP16] #####
######################################
save_path = "pretrained/mobilesal_temp_fp16.pth"
if os.path.exists(save_path):
    print('loading TensorRT model', save_path)
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(save_path))
else:
    print("converting model to TensorRT format!")
    model_trt = torch2trt(model, [x,y], fp16_mode=True)
    torch.save(model_trt.state_dict(), save_path)
print("Completed!!")

for i in tqdm(range(50)):
    p = model_trt(x,y)
    p = p + 1

total_t = 0
for i in tqdm(range(2000)):
    start = time()
    p = model_trt(x,y)
    p = p + 1 # replace torch.cuda.synchronize()
    total_t += time() - start
print(2000 / total_t)
print("TensorRT batchsize=1 fp16 speed test completed, expected 800FPS for RTX 2080Ti!")
