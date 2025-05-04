import torch
import viplanner_wrapper
import hydra
from omegaconf import DictConfig

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def viplanner_eval(cfg: DictConfig):
    # Access configuration parameters
    model_path = cfg.viplanner.model_path
    data_path = cfg.viplanner.data_path
    camera_cfg_path = cfg.viplanner.camera_cfg_path
    device = cfg.viplanner.device

    viplanner = viplanner_wrapper.VIPlannerAlgo(model_dir=model_path, device=device)
    hook = viplanner.net.decoder.fc2.register_forward_hook(ablate_neuron_fc2())


    # Load and process images from training data. Need to reshape to add batch dimension in front
    depth_image, sem_image = viplanner_wrapper.preprocess_training_images(data_path, 8, device)

    # setup goal, also needs to have batch dimension in front
    goals = torch.tensor([137, 111.0, 1.0], device=device).repeat(1, 1)
    goals = viplanner_wrapper.transform_goal(camera_cfg_path, goals, 8, device)

    _, paths, fear = viplanner.plan_dual(
        depth_image, sem_image, goals
    )
    print(f"paths {paths}, fear {fear}")

def ablate_neuron_fc2(module, input, output):
    output = output.clone()
    output[:, 100] = 0  # Zero out neuron 100
    return output

if __name__ == '__main__':
    viplanner_eval()