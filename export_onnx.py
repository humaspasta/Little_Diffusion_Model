import os
import torch 
from model import basic_model

patterns = ['moons' , 'circles']

for pattern in patterns:
    path = os.path.join('.' , 'weights', pattern)
    save_path = os.path.join('.' , 'Onnx_files', f'{pattern}_model.onnx')
    os.makedirs(os.path.join('.' , 'Onnx_files') , exist_ok=True)

    model = basic_model()
    weights = torch.load(path)
    model.load_state_dict(weights)
    model.eval()
    dummy_input = torch.randn(1,3)

    torch.onnx.export(model , dummy_input, save_path)
    print(f"Exported {pattern}: {os.path.getsize(save_path) / 1024:.1f} KB") 



    
