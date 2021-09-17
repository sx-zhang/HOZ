from __future__ import division

import time
import torch

from datasets.constants import AI2THOR_TARGET_CLASSES, AI2THOR_TARGET_CLASSES_19_TYPES

import setproctitle

from datasets.data import num_to_name
from models.model_io import ModelOptions

from agents.random_agent import RandomNavigationAgent

import random

from .train_util import (
    compute_loss,
    new_episode,
    run_episode,
    transfer_gradient_from_player_to_shared,
    end_episode,
    reset_player,
)


def a3c_train(
        rank,
        args,
        create_shared_model,
        shared_model,
        initialize_agent,
        optimizer,
        res_queue,
        end_flag,
        scenes,
):
    setproctitle.setproctitle('Training Agent: {}'.format(rank))

    targets = AI2THOR_TARGET_CLASSES[args.num_category]

    random.seed(args.seed + rank)
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]

    torch.cuda.set_device(gpu_id)
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)

    player = initialize_agent(create_shared_model, args, rank, scenes, targets, gpu_id=gpu_id)
    compute_grad = not isinstance(player, RandomNavigationAgent)

    model_options = ModelOptions()

    episode_num = 0

    while not end_flag.value:

        total_reward = 0
        player.eps_len = 0
        player.episode.episode_times = episode_num
        new_episode(args, player)
        player_start_time = time.time()

        while not player.done:
            player.sync_with_shared(shared_model)
            total_reward = run_episode(player, args, total_reward, model_options, True)
            loss = compute_loss(args, player, gpu_id, model_options)
            if compute_grad and loss['total_loss'] != 0:
                player.model.zero_grad()
                loss['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(player.model.parameters(), 100.0)
                transfer_gradient_from_player_to_shared(player, shared_model, gpu_id)
                optimizer.step()
            if not player.done:
                reset_player(player)

        for k in loss:
            loss[k] = loss[k].item()

        end_episode(
            player,
            res_queue,
            title=num_to_name(int(player.episode.scene[9:])),
            total_time=time.time() - player_start_time,
            total_reward=total_reward,
        )
        reset_player(player)

        episode_num = (episode_num + 1) % len(args.scene_types)

    player.exit()
