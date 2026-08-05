import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import config as settings
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.envs.metadrive_env import MetaDriveEnv
from model import DrivingModel, get_device


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
        use_render=settings.show_driving_window,
        image_observation=True,
        norm_pixel=True,
        stack_size=1,
        sensors=dict(
            rgb_camera=(RGBCamera, 112, 112),
        ),
        vehicle_config=dict(
            image_source="rgb_camera",
        ),
    )
)

def convert_image(image):
    image_tensor = torch.tensor(np.array(image), dtype=torch.float32)
    if image_tensor.ndim == 4:          # (N, H, W, C)
        return image_tensor.permute(0, 3, 1, 2)  # (N, C, H, W)
    # single image (H, W, C)
    return image_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

device = get_device()
model = DrivingModel().to(device)

saved_weights = torch.load(
    settings.model_file,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(saved_weights)

model.eval()

state, information = env.reset()
env.render()

def update():
    global state

    frame = state["image"][..., -1]  # (112, 112, 3)
    image_tensor = convert_image(frame)

    with torch.no_grad():
        predicted_action = model(image_tensor)
    action = predicted_action.squeeze(0).cpu().numpy()
    next_state, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
        state, information = env.reset()
        env.render()
    else:
        state = next_state

try:
    while True:
        update()
finally:
    env.close()
