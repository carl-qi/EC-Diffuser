import os
from isaac_panda_push_env import IsaacPandaPush
from isaac_env_wrappers import IsaacPandaPushGoalSB3Wrapper
from dlp_utils import check_config, load_pretrained_rep_model
import yaml
from pathlib import Path
import time
import pickle
import wandb
from moviepy.editor import ImageSequenceClip
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import cv2 as cv
from collections import deque
from dlp_utils import extract_dlp_features_with_bg

def setup_isaac_env(args):
    config_dir = args.env_config_dir

    # load config files
    config = yaml.safe_load(Path(f'{config_dir}/Config.yaml').read_text())
    isaac_env_cfg = yaml.safe_load(Path(f'{config_dir}/IsaacPandaPushConfig.yaml').read_text())

    # overwrite env_config/ based on args
    if args.planning_only:
        isaac_env_cfg['env']['numObjects'] = args.num_entity
        isaac_env_cfg["env"]["episodeLength"] = args.max_episode_length
        if args.push_t:
            isaac_env_cfg['env']['numColors'] = args.push_t_num_color
            isaac_env_cfg['env']['numGoalObjects'] = args.push_t_num_color
        isaac_env_cfg['env']['numObjects'] = args.num_entity
    else:
        if args.push_t:
            isaac_env_cfg['env']['numColors'] = args.num_entity
            isaac_env_cfg['env']['numGoalObjects'] = args.num_entity
    if not args.multiview:
        config['Model']['numViews'] = 1

    check_config(config, isaac_env_cfg, None)
    envs = IsaacPandaPush(
        cfg=isaac_env_cfg,
        rl_device=f"cuda:{config['cudaDevice']}",
        sim_device=f"cuda:{config['cudaDevice']}",
        graphics_device_id=config['cudaDevice'],
        headless=True,
        virtual_screen_capture=False,
        force_render=False,
    )

    latent_rep_model = load_pretrained_rep_model(dir_path=config['Model']['latentRepPath'], model_type=config['Model']['obsMode'])
    env = IsaacPandaPushGoalSB3Wrapper(
        env=envs,
        obs_mode=config['Model']['obsMode'],
        n_views=config['Model']['numViews'],
        latent_rep_model=latent_rep_model,
        latent_classifier=None,
        reward_cfg=config['Reward']['GT'],
        smorl=(config['Model']['method'] == 'SMORL'),
    )
    return env



def evaluate_policy(policy, env, args, logger, num_eval_episodes=100, exe_steps=1, stat_save_path=None):
    with torch.no_grad():
        policy.diffusion_model.eval()
        num_eval_episodes = (num_eval_episodes // env.num_envs) * env.num_envs
        # prepare stats variables
        total_returns = 0
        total_latent_rep_returns = 0
        total_avg_obj_dist = 0
        total_max_obj_dist = 0
        total_goal_success_frac = 0
        total_goals_reached = 0
        save_stat_dict = {
            "success": [],
            "success_frac": [],
            "avg_obj_dist": [],
            "max_obj_dist": [],
            "avg_return": [],
        }
        img_list = []
        goal_img = None
        ori_dist_list = []
        rollout = []

        print(f"\nEvaluating policy on {num_eval_episodes} random goals...")
        start_time = time.time()
        for i in tqdm(range(num_eval_episodes // env.num_envs)):
            # prepare rollout stats variables
            ret = np.zeros(env.num_envs)
            latent_rep_ret = np.zeros(env.num_envs)
            avg_obj_dist = np.ones(env.num_envs)
            max_obj_dist = np.ones(env.num_envs)
            goal_success_frac = np.zeros(env.num_envs)
            goals_reached = np.zeros(env.num_envs)
            obs = env.reset()
            t = 0
            while t < env.horizon:
                observation = obs['achieved_goal'][:, :, :].reshape(env.num_envs, -1)
                goal = obs['desired_goal'][:, :, :].reshape(env.num_envs, -1)
                conditions = {0: observation, args.horizon-1: goal}
                action_0, samples = policy(conditions, batch_size=1, verbose=args.verbose)

                for step in range(exe_steps):
                    actions = samples.actions[:, step]
                    new_obs, rewards, dones, infos = env.step(actions)
                    # gather stats
                    ret += rewards
                    avg_obj_dist = np.array([infos[i]["avg_obj_dist"] for i in range(len(infos))])
                    max_obj_dist = np.array([infos[i]["max_obj_dist"] for i in range(len(infos))])
                    goal_success_frac = np.array([infos[i]["goal_success_frac"] for i in range(len(infos))])
                    # save orientation distances
                    if env.push_t and t == env.horizon - 1:
                        ori_dist_list.extend(np.array([info["ori_dist"] for info in infos]))
                    
                    # save episode media and goals
                    if i == 0:
                        rollout.append(observation[0])
                        img_list.append(np.moveaxis(infos[0]["image"][0], 0, -1))
                        if t == 0:
                            goal_img = np.moveaxis(infos[0]["goal_image"][0], 0, -1)
                            front_bg = None
                            side_bg = None
                            if env.obs_mode == 'dlp':
                                front_img = infos[0]["image"][0]
                                _, front_bg = extract_dlp_features_with_bg(front_img, env.latent_rep_model, env.device)
                                side_img = infos[0]["image"][1]
                                _, side_bg = extract_dlp_features_with_bg(side_img, env.latent_rep_model, env.device)
                        if t == env.horizon - 1:
                            if goal_success_frac[0] == 1:
                                eval_vid_success = True
                                print("Visualized eval episode was a success")
                            else:
                                eval_vid_success = False
                                print("Visualized eval episode was a failure")
                        if t % 10 == 0: print(args.savepath, flush=True)
                        logger.log(t, samples, None, rollout, goal[0], front_bg=front_bg, side_bg=side_bg)
                    
                    # update last_obs
                    obs = new_obs
                    t += 1
                    if t >= env.horizon:
                        break
            total_returns += np.sum(ret)
            total_latent_rep_returns += np.sum(latent_rep_ret)
            total_avg_obj_dist += np.sum(avg_obj_dist)
            total_max_obj_dist += np.sum(max_obj_dist)
            total_goal_success_frac += np.sum(goal_success_frac)
            goals_reached[goal_success_frac == 1] = 1
            total_goals_reached += np.sum(goals_reached)
            save_stat_dict["success"].extend(goals_reached)
            save_stat_dict["success_frac"].extend(goal_success_frac)
            save_stat_dict["max_obj_dist"].extend(max_obj_dist)
            save_stat_dict["avg_obj_dist"].extend(avg_obj_dist)
            save_stat_dict["avg_return"].extend(ret / env.horizon)

        print(f"Evaluation completed in {time.time() - start_time:5.2f}s")
        if env.push_t:
            save_stat_dict["avg_ori_dist"] = ori_dist_list

        if stat_save_path is not None:
            with open(stat_save_path, 'wb') as file:
                pickle.dump(save_stat_dict, file)
            print(f"Saved eval stats to {stat_save_path}\n")

        # compute overall stats
        mean_return = total_returns / num_eval_episodes
        mean_latent_rep_return = total_latent_rep_returns / num_eval_episodes
        mean_avg_obj_dist = total_avg_obj_dist / num_eval_episodes
        mean_max_obj_dist = total_max_obj_dist / num_eval_episodes
        mean_success_frac = total_goal_success_frac / num_eval_episodes
        succes_rate = (total_goals_reached / num_eval_episodes) * 100

        std_success_rate = np.std(save_stat_dict["success"])
        std_return = np.std(save_stat_dict["avg_return"])
        std_success_frac = np.std(save_stat_dict["success_frac"])
        std_avg_obj_dist = np.std(save_stat_dict["avg_obj_dist"])
        std_max_obj_dist = np.std(save_stat_dict["max_obj_dist"])

        print(f"Goal success rate: {succes_rate:3.3f}%")
        print(f"Goal success fraction: {mean_success_frac:3.3f}")
        print(f"Max object-goal distance: {mean_max_obj_dist:3.3f}")
        print(f"Avg. object-goal distance: {mean_avg_obj_dist:3.3f}")
        print(f"Avg. reward: {mean_return / env.horizon:3.3f}")

        eval_stat_dict = {
            "succes_rate": succes_rate,
            "mean_success_frac": mean_success_frac,
            "mean_avg_obj_dist": mean_avg_obj_dist,
            "mean_max_obj_dist": mean_max_obj_dist,
            "mean_return": mean_return,
            "mean_latent_rep_return": mean_latent_rep_return,
            "std_success_rate": std_success_rate,
            "std_return": std_return,
            "std_success_frac": std_success_frac,
            "std_avg_obj_dist": std_avg_obj_dist,
            "std_max_obj_dist": std_max_obj_dist,
            "img_list": img_list,
            "goal_img": goal_img,
            "eval_vid_success": eval_vid_success,
            "entity_success_array": np.array(save_stat_dict["success_frac"]) * args.num_entity,
            "ori_dist_array": np.concatenate(ori_dist_list) if env.push_t else None,
            "mean_ori_dist":  np.mean(save_stat_dict['avg_ori_dist']) if env.push_t else None,
            "std_ori_dist": np.std(save_stat_dict["avg_ori_dist"]) if env.push_t else None,
        }
        policy.diffusion_model.train()
        return eval_stat_dict

def wandb_log_eval_stats(env, eval_stat_dict, args):
    # log stats
    wandb.log({"eval_goal_achievement_%": eval_stat_dict["succes_rate"]}, commit=False)
    wandb.log({"mean_success_frac": eval_stat_dict["mean_success_frac"]}, commit=False)
    wandb.log({"mean_avg_obj_dist": eval_stat_dict["mean_avg_obj_dist"]}, commit=False)
    wandb.log({"mean_max_obj_dist": eval_stat_dict["mean_max_obj_dist"]}, commit=False)
    wandb.log({"eval_mean_reward": eval_stat_dict["mean_return"]}, commit=False)
    wandb.log({"std_reward": eval_stat_dict["std_return"]}, commit=False)
    wandb.log({"std_success_rate": eval_stat_dict["std_success_rate"]}, commit=False)
    wandb.log({"std_success_frac": eval_stat_dict["std_success_frac"]}, commit=False)
    wandb.log({"std_avg_obj_dist": eval_stat_dict["std_avg_obj_dist"]}, commit=False)
    wandb.log({"std_max_obj_dist": eval_stat_dict["std_max_obj_dist"]}, commit=False)
    if env.push_t:
        wandb.log({"mean_ori_dist": eval_stat_dict["mean_ori_dist"]}, commit=False)
        wandb.log({"std_ori_dist": eval_stat_dict["std_ori_dist"]}, commit=False)
    # log episode video
    vid_save_dir = os.path.join(args.savepath, 'eval_episode_video.gif')
    ## add number to img_list and visualize with goal image side by side
    img_list = eval_stat_dict["img_list"]
    images = []
    for t, img in enumerate(img_list):
        img = img.copy()
        cv.putText(img, f'Timestep: {t}', (10, 10), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        images.append(np.concatenate([img, eval_stat_dict["goal_img"]], axis=1))
    clip = ImageSequenceClip(images, fps=15)
    clip.write_gif(vid_save_dir, fps=15)
    if hasattr(args, 'vis_traj_wandb') and args.vis_traj_wandb:
        # log goal image
        wandb.log({f"Eval Goal Image": wandb.Image(eval_stat_dict["goal_img"])}, commit=False)
        vid_caption = "Success" if eval_stat_dict["eval_vid_success"] else "Failure"
        vid = wandb.Video(data_or_path=vid_save_dir, caption=vid_caption, fps=15)
        wandb.log({f"Eval Episode Video": vid}, commit=False)
        # log orientation distance distribution plot
    if env.push_t:
        hist_fig = plt.figure(1, figsize=(5, 5), clear=True)
        plt.hist(eval_stat_dict["ori_dist_array"], bins=np.linspace(0, np.pi, num=50), edgecolor='black')
        wandb.log({f"Distribution of Orientation Distance from Goal": wandb.Image(hist_fig)}, commit=False)
    if "entity_success_array" in eval_stat_dict:
        hist_fig = plt.figure(1, figsize=(5, 5), clear=True)
        # plt.hist(eval_stat_dict["entity_success_array"], edgecolor='black')
        plt.hist([int(i*args.num_entity) for i in eval_stat_dict["entity_success_array"]], bins=np.arange(start=-0.5, stop=7, step=1), edgecolor='black')
        wandb.log({f"Distribution of Goal Success Fraction": wandb.Image(hist_fig)}, commit=False)