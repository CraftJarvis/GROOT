import av
import hydra
import argparse
from hydra import compose, initialize
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, Optional
from jarvis.arm.models import ConditionedAgent, MixedAgent
from jarvis.stark_tech.env_interface import MinecraftWrapper

POLICY_CONFIG_DIR = "arm/configs/policy"

def write(frames, path):
    container = av.open(path, mode='w', format='mp4')
    stream = container.add_stream('h264', rate=20)
    stream.width = 640 
    stream.height = 360
    stream.pix_fmt = 'yuv420p'
    for frame in frames:
        frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

def get_config_from_yaml(config_name: str):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    config_path = Path(POLICY_CONFIG_DIR) / f"{config_name}.yaml"
    initialize(config_path=str(config_path.parent), version_base='1.3')
    config = compose(config_name=config_path.stem)
    return config

def run(
    env: str, 
    task_config: Dict,
    policy_configs: str, 
    resolution: Optional[Tuple[int, int]] = None,
):
    frames = []
    agent = MixedAgent(
        obs_space=MinecraftWrapper.get_obs_space(),
        action_space=MinecraftWrapper.get_action_space(),
        policy_configs=policy_configs
    ).cuda()
    agent.eval()
    state = agent.initial_state()
    env = MinecraftWrapper(env, prev_action_obs=True)
    if resolution is not None:
        env.resize_resolution = resolution
    obs, info = env.reset()
    idx = 0
    bar = tqdm()
    terminated, truncated = False, False
    while (not terminated and not truncated):
        bar.set_description(f"Frame {idx}")
        obs['obs_conf'] = task_config
        action, state = agent.get_action(obs, state, first=None, input_shape="*")
        obs, reward, terminated, trauncated, info = env.step(action)
        frames.append(info['pov'])
        idx += 1
    env.close()
    write(frames, "video.mp4")

def run_test_vpt_single():
    task_config = {
        "A": {
            "ref_video": None, 
            "scale": "1.0", 
        }
    }
    policy_configs = {
        'A': get_config_from_yaml("vpt_native"),
    }
    run("collect_grass", task_config, policy_configs, resolution=(128, 128))

def run_test_groot_single():
    task_config = {
        "A": {
            "ref_video": "reference_videos/explore_mine.mp4", 
            "scale": "1.0", 
        }
    }
    policy_configs = {
        'A': get_config_from_yaml("groot_eff_1x"),
    }
    run("explore_mine", task_config, policy_configs, resolution=(224, 224))

def run_test_groot_subtract():
    task_config = {
        "A": {
            "ref_video": "reference_videos/build_golems.mp4", 
            "scale": "2.0", 
        }, 
        "B": {
            "ref_video": "reference_videos/baseline.mp4", 
            "scale": "-1.0", 
        }
    }
    policy_configs = {
        'A': get_config_from_yaml("groot_eff_1x"),
        'B': get_config_from_yaml("groot_eff_1x"),
    }
    run("build_golems", task_config, policy_configs, resolution=(224, 224))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default='run_test_vpt_single')
    args = parser.parse_args()
    if args.test == 'run_test_vpt_single':
        run_test_vpt_single()
    elif args.test == 'run_test_groot_single':
        run_test_groot_single()
    elif args.test == 'run_test_groot_subtract':
        run_test_groot_subtract()
    