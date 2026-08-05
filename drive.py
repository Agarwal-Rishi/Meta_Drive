import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.utils as utils


env = MetaDriveEnv(
    dict(
        map=settings.number_of_road_blocks,
        start_seed=settings.start_seed,
        num_scenarios=settings.number_of_scenarios,
        traffic_density=settings.traffic_density,
        random_spawn_lane_index=settings.random_starting_lane,
        out_of_road_done=settings.end_when_out_of_road,
        crash_vehicle_done=settings.end_when_vehicle_crashes,
        crash_object_done=settings.end_when_object_crashes,
    )
)

def convert_image(image):
    image_tensor = torch.tensor(image, dtype=torch.float32)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    return image_tensor  # (1, 3, 112, 112)

device = get_device()
model = model()
model.to(device)

model.eval()

state = env.reset()
env.render()

def update(dt):
    global state

    image_tensor = convert_image(state)

    with torch.no_grad():
        predicted_action = model(image_tensor)
    action = predicted_action.squeeze(0).cpu().numpy()
    next_state, reward, done, info = env.step(action)
    env.render()

    if done:
        state = env.reset()
        env.render()
    else:
        state = next_state

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)
try:    
    pyglet.app.run()
finally:
    env.close()