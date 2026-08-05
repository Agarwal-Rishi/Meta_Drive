import torch
import numpy
import config as settings
import torch.nn as nn
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.envs.metadrive_env import MetaDriveEnv
from torch.utils.data import TensorDataset, DataLoader

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_image(image):
    image_tensor = torch.tensor(image, dtype=torch.float32)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    return image_tensor  # (1, 3, 112, 112)

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
            nn.Tanh(),
        )

    def forward(self, x):
        return self.features(x)

model = model()
model.to(get_device())

if __name__ == "__main__":
    device = get_device()
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
            horizon=settings.maximum_episode_steps,
            use_render=settings.show_collection_window,
            manual_control=settings.use_manual_control,
            controller=settings.manual_controller,
            image_observation=True,
            sensors=dict(rgb_camera=(RGBCamera, 112, 112)),
            vehicle_config=dict(image_source="rgb_camera"),
        )
    )

    images = []
    actions = []

    current_image = env.reset()
    env.render()
    key_handler = key.KeyStateHandler()
    env.unwrapped.window.push_handlers(key_handler)
    recorded_steps = 0

    def update(dt):
        global current_image, recorded_steps

        steering = 0.0
        throttle = 0.0

        if key_handler[key.UP]:
            throttle = 1.0
        if key_handler[key.DOWN]:
            throttle = -1.0
        if key_handler[key.LEFT]:
            steering = -1.0
        if key_handler[key.RIGHT]:
            steering = 1.0

        current_action = [steering, throttle]

        next_image, reward, done, info = env.step(current_action)

        images.append(next_image)
        actions.append(current_action)

        recorded_steps += 1

        if done:
            env.reset()
        else:
            current_image = next_image
        env.render()

        if recorded_steps >= settings.data_collection_steps:
            env.close()

    pyglet.clock.schedule_interval(
        update,
        1.0 / env.unwrapped.frame_rate,
    )

    # initialize the train val test splits
    image_train_end = int(0.75 * len(images))
    action_train_end = int(0.75 * len(actions))
    image_val_end = int(0.85 * len(images))
    action_val_end = int(0.85 * len(actions))
    image_test_end = len(images)
    action_test_end = len(actions)

    train_images = convert_image(images[:image_train_end])
    train_actions = torch.tensor(
        actions[:train_end],
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

    model.save(settings.model_path)
    print(f"Model saved to {settings.model_path}")

    model.load(settings.model_path)
    print(f"Model loaded from {settings.model_path}")

    model.eval()
    with torch.no_grad():
        for images, actions in test_loader:
            images = images.to(device)
