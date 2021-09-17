from __future__ import print_function, division

import os
import random
import ctypes
import setproctitle
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from utils import command_parser

from utils.class_finder import model_class, agent_class, optimizer_class
from utils.model_util import ScalarMeanTracker
from utils.data_utils import check_data, loading_scene_list
from main_eval import main_eval
from full_eval import full_eval

from runners import a3c_train, a3c_val


os.environ["OMP_NUM_THREADS"] = "1"


def main():
    setproctitle.setproctitle("Train/Test Manager")
    args = command_parser.parse_arguments()

    print('Training started from: {}'.format(
        time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    )

    args.learned_loss = False
    args.num_steps = 50
    target = a3c_val if args.eval else a3c_train

    scenes = loading_scene_list(args)

    create_shared_model = model_class(args.model)
    init_agent = agent_class(args.agent_type)
    optimizer_type = optimizer_class(args.optimizer)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.eval:
        args.test_or_val = 'test'
        main_eval(args, create_shared_model, init_agent)
        return

    start_time = time.time()
    local_start_time_str = time.strftime(
        '%Y_%m_%d_%H_%M_%S', time.localtime(start_time)
    )

    tb_log_dir = args.log_dir + '/' + args.title + '_' + args.phase + '_' + local_start_time_str
    log_writer = SummaryWriter(log_dir=tb_log_dir)

    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method("spawn")

    shared_model = create_shared_model(args)

    train_total_ep = 0
    n_frames = 0

    if args.continue_training is not None:
        orgin_state = shared_model.state_dict()
        saved_state = torch.load(
            args.continue_training, map_location=lambda storage, loc: storage
        )
        orgin_state.update(saved_state)
        shared_model.load_state_dict(orgin_state)
        train_total_ep = int(args.continue_training.split('_')[-7])
        n_frames = int(args.continue_training.split('_')[-8])

    if args.fine_tuning is not None:
        saved_state = torch.load(
            args.fine_tuning, map_location=lambda storage, loc: storage
        )
        model_dict = shared_model.state_dict()
        pretrained_dict = {k: v for k, v in saved_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(pretrained_dict)
        shared_model.load_state_dict(model_dict)

    if args.update_meta_network:
        for layer, parameters in shared_model.named_parameters():
            if not layer.startswith('meta'):
                parameters.requires_grad = False

    shared_model.share_memory()
    optimizer = optimizer_type(
        [v for k, v in shared_model.named_parameters() if v.requires_grad], lr=args.lr
    )
    optimizer.share_memory()
    print(shared_model)

    processes = []

    end_flag = mp.Value(ctypes.c_bool, False)
    train_res_queue = mp.Queue()

    for rank in range(0, args.workers):
        p = mp.Process(
            target=target,
            args=(
                rank,
                args,
                create_shared_model,
                shared_model,
                init_agent,
                optimizer,
                train_res_queue,
                end_flag,
                scenes,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)

    print("Train agents created.")

    train_thin = args.train_thin
    train_scalars = ScalarMeanTracker()

    try:
        while train_total_ep < args.max_ep:

            train_result = train_res_queue.get()
            train_scalars.add_scalars(train_result)
            train_total_ep += 1
            n_frames += train_result['ep_length']

            if (train_total_ep % train_thin) == 0:
                log_writer.add_scalar('n_frames', n_frames, train_total_ep)
                tracked_means = train_scalars.pop_and_reset()
                for k in tracked_means:
                    log_writer.add_scalar(
                        k + '/train', tracked_means[k], train_total_ep
                    )

            if (train_total_ep % args.ep_save_freq) == 0:

                print('{}: {}'.format(train_total_ep, n_frames))
                if not os.path.exists(args.save_model_dir):
                    os.makedirs(args.save_model_dir)
                state_to_save = shared_model.state_dict()
                save_path = os.path.join(
                    args.save_model_dir,
                    '{0}_{1}_{2}_{3}.dat'.format(
                        args.title, n_frames, train_total_ep, local_start_time_str
                    ),
                )
                torch.save(state_to_save, save_path)

    finally:
        log_writer.close()
        end_flag.value = True
        for p in processes:
            time.sleep(0.1)
            p.join()

    if args.test_after_train:
        full_eval()


if __name__ == "__main__":
    main()
