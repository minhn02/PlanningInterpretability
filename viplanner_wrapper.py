import os
import torch
import torchvision.transforms as transforms
import numpy as np
from viplanner.viplanner.config import TrainCfg
from viplanner.viplanner.plannernet import AutoEncoder, DualAutoEncoder
from viplanner.viplanner.traj_cost_opt.traj_opt import TrajOpt


class VIPlannerAlgo:
    def __init__(self, model_dir: str, fear_threshold: float = 0.5, device: str = "cuda"):
        """Apply VIPlanner Algorithm

        Args:
            model_dir (str): Directory that includes model.pt and model.yaml
        """
        super().__init__()

        assert os.path.exists(model_dir), "Model directory does not exist"
        assert os.path.isfile(os.path.join(model_dir, "model.pt")), "Model file does not exist"
        assert os.path.isfile(os.path.join(model_dir, "model.yaml")), "Model config file does not exist"

        # Parameters
        self.fear_threshold = fear_threshold
        self.device = device

        # Load model
        self.train_config: TrainCfg = None
        self.load_model(model_dir)

        # Get transforms for images
        self.transform = transforms.Resize(self.train_config.img_input_size, antialias=None)

        # Initialize trajectory optimizer
        self.traj_generate = TrajOpt()

        # Visualization colors and sizes
        self.color_fear = (1.0, 0.4, 0.1)  # red
        self.color_path = (0.4, 1.0, 0.1)  # green
        self.size = 5.0

    def load_model(self, model_dir: str):
        """Load the model and its configuration."""
        # Load training configuration
        self.train_config: TrainCfg = TrainCfg.from_yaml(os.path.join(model_dir, "model.yaml"))
        print(
            f"Model loaded using sem: {self.train_config.sem}, rgb: {self.train_config.rgb}, "
            f"knodes: {self.train_config.knodes}, in_channel: {self.train_config.in_channel}"
        )

        if isinstance(self.train_config.data_cfg, list):
            self.max_goal_distance = self.train_config.data_cfg[0].max_goal_distance
            self.max_depth = self.train_config.data_cfg[0].max_depth
        else:
            self.max_goal_distance = self.train_config.data_cfg.max_goal_distance
            self.max_depth = self.train_config.data_cfg.max_depth

        # Initialize the appropriate model
        if self.train_config.sem:
            self.net = DualAutoEncoder(self.train_config)
        else:
            self.net = AutoEncoder(self.train_config.in_channel, self.train_config.knodes)

        # Load model weights
        try:
            model_state_dict, _ = torch.load(os.path.join(model_dir, "model.pt"), map_location=self.device)
        except ValueError:
            model_state_dict = torch.load(os.path.join(model_dir, "model.pt"), map_location=self.device)
        self.net.load_state_dict(model_state_dict)

        # Set model to evaluation mode
        self.net.eval()

        # Move model to the appropriate device
        if self.device.lower() == "cpu":
            print("CUDA not available, VIPlanner will run on CPU")
            self.cuda_avail = False
        else:
            self.net = self.net.to(self.device)
            self.cuda_avail = True

    ###
    # Transformations
    ###

    def goal_transformer(self, goal: torch.Tensor, cam_pos: torch.Tensor, cam_quat: torch.Tensor) -> torch.Tensor:
        """Transform goal into camera frame."""
        goal_cam_frame = goal - cam_pos
        goal_cam_frame[:, 2] = 0  # trained with z difference of 0
        goal_cam_frame = self.quat_apply(self.quat_inv(cam_quat), goal_cam_frame)
        return goal_cam_frame

    def path_transformer(
        self, path_cam_frame: torch.Tensor, cam_pos: torch.Tensor, cam_quat: torch.Tensor
    ) -> torch.Tensor:
        """Transform path from camera frame to world frame."""
        return self.quat_apply(
            cam_quat.unsqueeze(1).repeat(1, path_cam_frame.shape[1], 1), path_cam_frame
        ) + cam_pos.unsqueeze(1)

    def input_transformer(self, image: torch.Tensor) -> torch.Tensor:
        """Transform input images."""
        image = self.transform(image)
        image[image > self.max_depth] = 0.0
        image[~torch.isfinite(image)] = 0  # set all inf or nan values to 0
        return image

    ###
    # Quaternion Utilities
    ###

    @staticmethod
    def quat_inv(q: torch.Tensor) -> torch.Tensor:
        """Invert a quaternion."""
        q_conj = q.clone()
        q_conj[..., :3] *= -1  # Negate the vector part
        return q_conj

    @staticmethod
    def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Apply a quaternion rotation to a vector."""
        q_xyz = q[..., :3]
        q_w = q[..., 3:4]

        # Compute cross products
        t = 2 * torch.cross(q_xyz, v, dim=-1)
        v_rotated = v + q_w * t + torch.cross(q_xyz, t, dim=-1)
        return v_rotated

    ###
    # Planning
    ###

    def plan(self, image: torch.Tensor, goal_robot_frame: torch.Tensor) -> tuple:
        """Plan a trajectory using a single input image."""
        with torch.no_grad():
            keypoints, fear = self.net(self.input_transformer(image), goal_robot_frame)
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)

        return keypoints, traj, fear

    def plan_dual(self, dep_image: torch.Tensor, sem_image: torch.Tensor, goal_robot_frame: torch.Tensor) -> tuple:
        """Plan a trajectory using depth and semantic images."""
        # Transform input
        sem_image = self.transform(sem_image) / 255
        with torch.no_grad():
            keypoints, fear = self.net(self.input_transformer(dep_image), sem_image, goal_robot_frame)
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)

        return keypoints, traj, fear

    ###
    # Debugging
    ###

    def debug_draw(self, paths: torch.Tensor, fear: torch.Tensor, goal: torch.Tensor):
        """Debugging utility to print paths and fear levels."""
        for idx, curr_path in enumerate(paths):
            if fear[idx] > self.fear_threshold:
                print(f"[FEAR] Path {idx}: {curr_path.cpu().numpy()}, Goal: {goal.cpu().numpy()}")
            else:
                print(f"[SAFE] Path {idx}: {curr_path.cpu().numpy()}, Goal: {goal.cpu().numpy()}")