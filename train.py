import torch
import numpy
import config as settings
import torch.nn as nn
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.expert_policy import ExpertPolicy
from torch.utils.data import TensorDataset, DataLoader


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_image(image):
    image_tensor = torch.tensor(numpy.array(image), dtype=torch.float32)
    if image_tensor.ndim == 4:          # (N, H, W, C)
        return image_tensor.permute(0, 3, 1, 2)  # (N, C, H, W)
    # single image (H, W, C)
    return image_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),   # 112 → 56
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 56 → 28
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 28 → 14
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# 14 → 7
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),

            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 2),
        )

    def forward(self, x):
        return self.features(x)

model = model()
model.to(get_device())

if __name__ == "__main__":
    device = get_device()

    env = MetaDriveEnv(
        dict(
            map=3,
            start_seed=42,
            num_scenarios=1,
            traffic_density=0.0,
            need_inverse_traffic=False,
            random_spawn_lane_index=False,
            map_config=dict(
                type="block_sequence",
                config="CCCC",
                lane_num=2,
                lane_width=3.5,
            ),
            out_of_road_done=True,
            crash_vehicle_done=True,
            crash_object_done=True,
            horizon=2000,
            use_render=True,
            manual_control=False,
            agent_policy=ExpertPolicy,
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

    images = []
    actions = []

    current_image = env.reset()
    env.render()
    recorded_steps = 0

    while recorded_steps < settings.data_collection_steps:
        # ExpertPolicy drives; dummy step action is ignored
        next_image, reward, terminated, truncated, info = env.step([0.0, 0.0])
        env.render()

        current_action = [env.agent.steering, env.agent.throttle_brake]
        frame = next_image["image"][..., -1]  # (112, 112, 3)
        images.append(frame)
        actions.append(current_action)
        recorded_steps += 1

        if terminated or truncated:
            current_image = env.reset()
            env.render()
        else:
            current_image = next_image

    env.close()

    # initialize the train val test splits
    image_train_end = int(0.75 * len(images))
    action_train_end = int(0.75 * len(actions))
    image_val_end = int(0.85 * len(images))
    action_val_end = int(0.85 * len(actions))
    image_test_end = len(images)
    action_test_end = len(actions)

    train_images = convert_image(images[:image_train_end])
    train_actions = torch.tensor(
        actions[:action_train_end],
        dtype=torch.float32,
    )
    val_images = convert_image(images[image_train_end:image_val_end])
    val_actions = torch.tensor(
        actions[action_train_end:action_val_end],
        dtype=torch.float32,
    )
    test_images = convert_image(images[image_val_end:image_test_end])
    test_actions = torch.tensor(
        actions[action_val_end:action_test_end],
        dtype=torch.float32,
    )

    train_dataset = TensorDataset(train_images, train_actions)
    val_dataset = TensorDataset(val_images, val_actions)
    test_dataset = TensorDataset(test_images, test_actions)

    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=settings.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=settings.batch_size, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(settings.epochs):
        for images, actions in train_loader:
            images = images.to(device)
            actions = actions.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
        
            print(f"Epoch {epoch}, Loss: {loss.item()}")
        #validation
        model.eval()
        with torch.no_grad():
            for images, actions in val_loader:
                images = images.to(device)
                actions = actions.to(device)
                outputs = model(images)
                loss = criterion(outputs, actions)
                print(f"Validation Loss: {loss.item()}")
                
    model.eval()
    with torch.no_grad():
        for images, actions in test_loader:
            images = images.to(device)
            actions = actions.to(device)
            outputs = model(images)
            loss = criterion(outputs, actions)
            print(f"Test Loss: {loss.item()}")

settings.model_folder.mkdir(parents=True, exist_ok=True)

torch.save(
    model.state_dict(),
    settings.model_file,
)

print(f"Model saved to {settings.model_file}")



