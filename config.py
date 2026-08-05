from pathlib import Path


# File and folder locations

project_folder = Path(__file__).resolve().parent

data_folder = project_folder / "data"
model_folder = project_folder / "models"

demonstration_file = data_folder / "demonstrations.npz"
model_file = model_folder / "driving_model.pytorch"


# Track settings

number_of_road_blocks = 3
start_seed = 0
number_of_scenarios = 1
traffic_density = 0.0
random_starting_lane = False


# Episode-ending settings

end_when_out_of_road = True
end_when_vehicle_crashes = True
end_when_object_crashes = True
maximum_episode_steps = 2000


# Data-collection settings

show_collection_window = True
use_manual_control = True
manual_controller = "keyboard"


# Artificial-intelligence driving settings

show_driving_window = True


# Supervised-learning settings

learning_rate = 0.001
batch_size = 64
number_of_training_epochs = 30
validation_fraction = 0.20
random_seed = 42

data_collection_steps = 5000