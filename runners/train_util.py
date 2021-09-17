from __future__ import division

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import json


def save_actions(actions, episode, spl, scene_type, agent_start_state):
    if spl>0:
        success = True
    else:
        success = False
    start_state = {'x': agent_start_state.x,
                   'y': agent_start_state.y,
                   'z': agent_start_state.z,
                   'rotation': dict(x=0.0, y=agent_start_state.rotation, z=0.0),
                   'horizon': agent_start_state.horizon}

    save_dict = {'scene': episode._env.scene_name,
                 'scene_type': scene_type,
                 'target': episode.target_object,
                 'start_state': start_state,
                 'actions': actions,
                 'success': success,
                 'spl': spl}
    return save_dict


def write_json(file_dir, save_dict):
    with open(file_dir, "w") as f:
        json.dump(save_dict, f)


def update_test_model(args, player, target_action_prob, weight=1):
    action_loss = weight * F.cross_entropy(player.last_action_probs, torch.max(target_action_prob, 1)[1])
    inner_gradient = torch.autograd.grad(
        action_loss,
        [v.requires_grad_() for _, v in player.model.named_parameters()],
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    player.model.load_state_dict(SGD_step(player.model, inner_gradient, args.inner_lr))
    player.episode.model_update = True


def run_episode(player, args, total_reward, model_options, training, shared_model=None):
    num_steps = args.num_steps
    update_test_type = {
        'meta_learning': args.update_test_meta_learning
    }

    update_test = True in update_test_type.values()

    for _ in range(num_steps):
        player.action(model_options, training, update_test)

        if update_test_type['meta_learning']:
            current_state = str(player.episode.environment.controller.state)
            if current_state in player.episode.states:
                target_action_prob = player.episode.meta_predictions[-1]
                update_test_model(args, player, target_action_prob, 1)

        total_reward = total_reward + player.reward
        if player.done:
            break
    return total_reward


def new_episode(args, player):
    player.episode.new_episode(args, player.scenes, player.targets)
    player.reset_hidden()
    player.done = False


def a3c_loss(args, player, gpu_id, model_options):
    """ Borrowed from https://github.com/dgriff777/rl_a3c_pytorch. """
    R = torch.zeros(1, 1)
    if not player.done:
        _, output = player.eval_at_state(model_options)
        R = output.value.data

    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            R = R.cuda()

    player.values.append(Variable(R))
    policy_loss = 0
    value_loss = 0
    gae = torch.zeros(1, 1)
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            gae = gae.cuda()
    R = Variable(R)
    for i in reversed(range(len(player.rewards))):
        R = args.gamma * R + player.rewards[i]
        advantage = R - player.values[i]
        value_loss = value_loss + 0.5 * advantage.pow(2)

        delta_t = (
                player.rewards[i]
                + args.gamma * player.values[i + 1].data
                - player.values[i].data
        )

        gae = gae * args.gamma * args.tau + delta_t

        policy_loss = (
                policy_loss
                - player.log_probs[i] * Variable(gae)
                - args.beta * player.entropies[i]
        )

    return policy_loss, value_loss


def imitation_learning_loss(player):
    episode_loss = torch.tensor(0)
    with torch.cuda.device(player.gpu_id):
        episode_loss = episode_loss.cuda()
    for i in player.il_update_actions:
        step_optimal_action = torch.tensor(player.il_update_actions[i]).reshape([1]).long()
        with torch.cuda.device(player.gpu_id):
            step_optimal_action = step_optimal_action.cuda()
        step_loss = F.cross_entropy(player.probs[i], step_optimal_action)
        episode_loss = episode_loss + step_loss

    return episode_loss

def meta_learning_loss(player):
    episode_loss = torch.tensor(0)
    with torch.cuda.device(player.gpu_id):
        episode_loss = episode_loss.cuda()
    for i in player.meta_learning_actions:
        step_optimal_action = torch.tensor(player.meta_learning_actions[i]).reshape([1]).long()
        with torch.cuda.device(player.gpu_id):
            step_optimal_action = step_optimal_action.cuda()
        # step_loss = F.cross_entropy(player.meta_predictions[i], step_optimal_action)
        step_loss = F.cross_entropy(player.probs[i], step_optimal_action)
        episode_loss = episode_loss + step_loss

    return episode_loss

def duplicate_states_loss(player):
    episode_loss = torch.tensor(0)
    with torch.cuda.device(player.gpu_id):
        episode_loss = episode_loss.cuda()
    for i in player.duplicate_states_actions:
        step_optimal_action = torch.tensor(player.duplicate_states_actions[i]).reshape([1]).long()
        with torch.cuda.device(player.gpu_id):
            step_optimal_action = step_optimal_action.cuda()
        step_loss = F.cross_entropy(player.probs[i], step_optimal_action)
        episode_loss = episode_loss + step_loss

    return episode_loss


def compute_learned_loss(args, player, gpu_id, model_options):
    loss_hx = torch.cat((player.hidden[0], player.last_action_probs), dim=1)
    learned_loss = {
        "learned_loss": player.model.learned_loss(
            loss_hx, player.learned_input, model_options.params
        )
    }
    player.learned_input = None
    return learned_loss


def transfer_gradient_from_player_to_shared(player, shared_model, gpu_id):
    """ Transfer the gradient from the player's model to the shared model
        and step """
    for param, shared_param in zip(
            player.model.parameters(), shared_model.parameters()
    ):
        if shared_param.requires_grad:
            if param.grad is None:
                shared_param._grad = torch.zeros(shared_param.shape)
            elif gpu_id < 0:
                shared_param._grad = param.grad
            else:
                shared_param._grad = param.grad.cpu()


def transfer_gradient_to_shared(gradient, shared_model, gpu_id):
    """ Transfer the gradient from the player's model to the shared model
        and step """
    i = 0
    for name, param in shared_model.named_parameters():
        if param.requires_grad:
            if gradient[i] is None:
                param._grad = torch.zeros(param.shape)
            elif gpu_id < 0:
                param._grad = gradient[i]
            else:
                param._grad = gradient[i].cpu()

        i += 1


def get_params(shared_model, gpu_id):
    """ Copies the parameters from shared_model into theta. """
    theta = {}
    for name, param in shared_model.named_parameters():
        # Clone and detach.
        param_copied = param.clone().detach().requires_grad_(True)
        if gpu_id >= 0:
            # theta[name] = torch.tensor(
            #     param_copied,
            #     requires_grad=True,
            #     device=torch.device("cuda:{}".format(gpu_id)),
            # )
            # Changed for pythorch 0.4.1.
            theta[name] = param_copied.to(torch.device("cuda:{}".format(gpu_id)))
        else:
            theta[name] = param_copied
    return theta


def update_loss(sum_total_loss, total_loss):
    if sum_total_loss is None:
        return total_loss
    else:
        return sum_total_loss + total_loss


def reset_player(player):
    player.clear_actions()
    player.repackage_hidden()


def SGD_step(theta, grad, lr):
    theta_i = {}
    j = 0
    for name, param in theta.named_parameters():
        if grad[j] is not None and "exclude" not in name and "ll" not in name:
            theta_i[name] = param - lr * grad[j]
        else:
            theta_i[name] = param
        j += 1

    return theta_i


def get_scenes_to_use(player, scenes, args):
    if args.new_scene:
        return scenes
    return [player.episode.environment.scene_name]


def compute_loss(args, player, gpu_id, model_options):
    if not args.only_il_loss:
        loss = {'policy_loss': a3c_loss(args, player, gpu_id, model_options)[0],
                'value_loss': a3c_loss(args, player, gpu_id, model_options)[1]}
        loss['total_loss'] = loss['policy_loss'] + 0.5 * loss['value_loss']
        if args.imitation_learning:
            loss['il_loss'] = imitation_learning_loss(player)
            loss['total_loss'] = loss['total_loss'] + args.il_rate * loss['il_loss']
        elif args.memory_duplicate_learning:
            loss['memory_loss'] = duplicate_states_loss(player)
            loss['total_loss'] = loss['total_loss'] + args.memory_duplicate_rate * loss['memory_loss']
    elif args.update_meta_network and args.only_il_loss:
        meta_loss = meta_learning_loss(player)
        loss = {'meta_loss': meta_loss,
                'total_loss': meta_loss}

    return loss


def end_episode(
        player, res_queue, title=None, episode_num=0, include_obj_success=False, **kwargs
):
    results = {
        'done_count': player.episode.done_count,
        'ep_length': player.eps_len,
        'success': int(player.success),
        'tools': {
            'scene': player.episode.scene,
            'target': player.episode.task_data,
            'states': player.episode.states,
            'action_outputs': player.episode.action_outputs,
            'action_list': [int(item) for item in player.episode.actions_record],
            'detection_results': player.episode.detection_results,
            'success': player.success,
            'match_score': player.episode.match_score,
            'indices_topk': player.episode.indices_topk,
        }
    }

    results.update(**kwargs)
    res_queue.put(results)


def get_bucketed_metrics(spl, best_path_length, success,action_value):
    out = {}
    for i in [1, 5]:
        if best_path_length >= i:
            out["GreaterThan/{}/success".format(i)] = success
            out["GreaterThan/{}/spl".format(i)] = spl
            out["GreaterThan/{}/sae".format(i)] = action_value

    return out



def compute_spl(player, start_state):
    best = float("inf")
    for obj_id in player.episode.task_data:
        try:
            _, best_path_len, _ = player.environment.controller.shortest_path_to_target(
                start_state, obj_id, False
            )
            if best_path_len < best:
                best = best_path_len
        except:
            continue

    if not player.success:
        return 0, best

    if best < float("inf"):
        return best / float(player.eps_len), best

    return 0, best


def action_prob_detection(bbox):
    center_point = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

    left_prob = np.linalg.norm(center_point - np.array([0, 150]))
    right_prob = np.linalg.norm(center_point - np.array([300, 150]))
    up_prob = np.linalg.norm(center_point - np.array([150, 0]))
    down_prob = np.linalg.norm(center_point - np.array([150, 300]))
    forward_prob = np.linalg.norm(center_point - np.array([150, 150]))

    detection_prob = torch.tensor([forward_prob, left_prob, right_prob, up_prob, down_prob])

    return torch.argmin(detection_prob)
