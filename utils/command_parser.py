import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Visual Navigation')

    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        metavar='LR',
        help='learning rate (default: 0.0001)',
    )

    parser.add_argument(
        '--inner-lr',
        type=float,
        default=0.0001,
        metavar='ILR',
        help='learning rate (default: 0.01)',
    )

    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        metavar='G',
        help='discount factor for rewards (default: 0.99)',
    )

    parser.add_argument(
        '--tau',
        type=float,
        default=1.00,
        metavar='T',
        help='parameter for GAE (default: 1.00)',
    )

    parser.add_argument(
        '--beta',
        type=float,
        default=1e-2,
        help='entropy regularization term'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        metavar='S',
        help='random seed (default: 1)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=12,
        metavar='W',
        help='how many training processes to use (default: 4)',
    )

    parser.add_argument(
        '--phase',
        type=str,
        default='train',
        help='train, eval or debug three choices'
    )

    parser.add_argument(
        '--num-category',
        type=int,
        default=22,
        help='the number of categories'
    )

    parser.add_argument(
        '--max-episode-length',
        type=int,
        default=30,
        metavar='M',
        help='maximum length of an episode (default: 100)',
    )

    parser.add_argument(
        '--meta_train_episodes',
        type=int,
        default=1,
        help='how many meta-train episodes.',
    )

    parser.add_argument(
        '--meta_test_episodes',
        type=int,
        default=1,
        help='how many meta-test episodes'
    )

    parser.add_argument(
        '--shared-optimizer',
        default=True,
        metavar='SO',
        help='use an optimizer with shared statistics.',
    )

    parser.add_argument(
        '--load-model',
        type=str,
        default='',
        help='Path to load a saved model.'
    )

    parser.add_argument(
        '--continue-training',
        type=str,
        default=None,
        help='continue training based on given model'
    )

    parser.add_argument(
        '--fine-tuning',
        type=str,
        default=None,
        help='fine tune based on given model'
    )

    parser.add_argument(
        '--ep-save-freq',
        type=int,
        default=1e5,
        help='save model after this # of training episodes (default: 1e+4)',
    )

    parser.add_argument(
        '--optimizer',
        default='SharedAdam',
        metavar='OPT',
        help='shared optimizer choice of SharedAdam or SharedRMSprop',
    )

    parser.add_argument(
        '--save-model-dir',
        default='trained_models/',
        metavar='SMD',
        help='folder to save trained navigation',
    )

    parser.add_argument(
        '--log-dir',
        default='runs/',
        metavar='LG',
        help='folder to save logs'
    )

    parser.add_argument(
        '--gpu-ids',
        type=int,
        default=-1,
        nargs='+',
        help='GPUs to use [-1 CPU only] (default: -1)',
    )

    parser.add_argument(
        '--amsgrad',
        default=True,
        metavar='AM',
        help='Adam optimizer amsgrad parameter'
    )

    parser.add_argument(
        '--grid_size',
        type=float,
        default=0.25,
        metavar='GS',
        help='The grid size used to discretize AI2-THOR maps.',
    )

    parser.add_argument(
        '--docker_enabled',
        action='store_true',
        help='Whether or not to use docker.'
    )

    parser.add_argument(
        '--test-after-train',
        action='store_true',
        help='run test after training'
    )

    parser.add_argument(
        '--wander',
        action='store_true',
        help='agent will aim to wander'
    )

    parser.add_argument(
        '--wander-during-training',
        action='store_true',
        help='train wander episode during navigation'
    )

    parser.add_argument(
        '--x_display',
        type=str,
        default=None,
        help='The X display to target, if any.'
    )

    parser.add_argument(
        '--test_timeout',
        type=int,
        default=10,
        help='The length of time to wait in between test runs.',
    )

    parser.add_argument(
        '--verbose',
        type=bool,
        default=False,
        help='If true, output will contain more information.',
    )

    parser.add_argument(
        '--max-ep',
        type=float,
        default=8000000,
        help='maximum # of episodes'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='BaseModel',
        help='Model to use.'
    )

    parser.add_argument(
        '--train_thin',
        type=int,
        default=1000,
        help='How often to print'
    )

    parser.add_argument(
        '--num-memory-block',
        type=int,
        default=2,
        help='the number of memory blocks'
    )

    parser.add_argument(
        '--local_executable_path',
        type=str,
        default=None,
        help='a path to the local thor build.',
    )

    parser.add_argument(
        '--hindsight_replay',
        type=bool,
        default=False,
        help='whether or not to use hindsight replay.',
    )

    parser.add_argument(
        '--enable_test_agent',
        action='store_true',
        help='Whether or not to have a test agent.',
    )

    parser.add_argument(
        '--title',
        type=str,
        default='a3c',
        help='Info for logging.'
    )

    parser.add_argument(
        '--train_scenes',
        type=str,
        default='[1-20]',
        help='scenes for training.'
    )

    parser.add_argument(
        '--val_scenes',
        type=str,
        default='[21-30]',
        help='old validation scenes before formal split.',
    )

    parser.add_argument(
        '--possible_targets',
        type=str,
        default='FULL_OBJECT_CLASS_LIST',
        help='all possible objects.',
    )

    # if none use all dest objects
    parser.add_argument(
        '--train_targets',
        type=str,
        default=None,
        help='specific objects for this experiment from the object list.',
    )

    parser.add_argument(
        '--action_space',
        type=int,
        default=6,
        help='space of possible actions.'
    )

    parser.add_argument(
        '--hidden_state_sz',
        type=int,
        default=512,
        help='size of hidden state of LSTM.'
    )

    parser.add_argument(
        '--attention-sz',
        type=int,
        default=512,
    )

    parser.add_argument(
        '--compute_spl',
        action='store_true',
        help='compute the spl.'
    )

    parser.add_argument(
        '--eval',
        action='store_true',
        help='run the test code'
    )

    parser.add_argument(
        '--include-test',
        action='store_true',
        help='run test during eval'
    )

    parser.add_argument(
        '--csiro',
        action='store_true',
        help='active when the code is used in csiro clusters'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='./datasets/Scene_Data',
        help='where dataset is stored.',
    )

    parser.add_argument(
        '--graph-file',
        type=str,
        default='graph.json',
        help='the name of the graph file'
    )

    parser.add_argument(
        '--grid-file',
        type=str,
        default='grid.json',
        help='the name of the grid file'
    )

    parser.add_argument(
        '--visible-map-file-name',
        type=str,
        default='visible_object_map_1.5.json',
        help='the name of the visible object map file'
    )

    parser.add_argument(
        '--detection-feature-file-name',
        type=str,
        default='det_feature_22_cates.hdf5',
        help='Which file store the detection feature?'
    )

    parser.add_argument(
        '--images-file-name',
        type=str,
        default='resnet18_featuremap.hdf5',
        help='Where the controller looks for images. Can be switched out to real images or Resnet features.',
    )

    parser.add_argument(
        '--optimal-action-file-name',
        type=str,
        default='optimal_action.json'
    )

    parser.add_argument(
        '--disable-strict_done',
        dest='strict_done',
        action='store_false'
    )

    parser.set_defaults(strict_done=True)

    parser.add_argument(
        '--results-json',
        type=str,
        default='metrics.json',
        help='Write the results.'
    )

    parser.add_argument(
        '--visualize-file-name',
        type=str,
        default='visual_temp.json'
    )

    parser.add_argument(
        '--agent_type',
        type=str,
        default='NavigationAgent',
        help='Which type of agent. Choices are NavigationAgent or RandomAgent.',
    )

    parser.add_argument(
        '--episode_type',
        type=str,
        default='BasicEpisode',
        help='Which type of agent. Choices are NavigationAgent or RandomAgent.',
    )

    parser.add_argument(
        '--fov',
        type=float,
        default=100.0,
        help='The field of view to use.'
    )

    parser.add_argument(
        '--il-rate',
        type=float,
        default=1,
        help='the rate of imitation learning loss'
    )

    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.25,
        help='The dropout ratio to use (default is no dropout).',
    )

    parser.add_argument(
        '--scene-types',
        nargs='+',
        default=['kitchen', 'living_room', 'bedroom', 'bathroom'],
    )

    parser.add_argument(
        '--gradient_limit',
        type=int,
        default=4,
        help='How many gradient steps allowed for MAML.',
    )

    parser.add_argument(
        '--test_or_val',
        default='val',
        help='test or val'
    )

    parser.add_argument(
        '--test-start-from',
        type=int,
        default=0,
        help='start from given episode'
    )

    parser.add_argument(
        '--multi-heads',
        type=int,
        default=1
    )

    parser.add_argument(
        '--multi-heads-attention',
        type=int,
        default=1,
    )

    parser.add_argument(
        '--update-meta-network',
        action='store_true',
        help='only update layers startswith meta'
    )

    parser.add_argument(
        '--meta-duplicate-action',
        action='store_true',
        help='update the meta network for duplicate actions'
    )

    parser.add_argument(
        '--meta-failed-action',
        action='store_true',
        help='update the meta network for failed actions'
    )

    parser.add_argument(
        '--meta-all-steps',
        action='store_true',
        help='update the meta network in all steps'
    )

    parser.add_argument(
        '--update-test-meta-learning',
        action='store_true',
        help='update the test model based on meta learning network predict'
    )

    parser.add_argument(
        '--update-test-long-term',
        action='store_true',
        help='update test model based on long term memory'
    )

    parser.add_argument(
        '--update-test-failure',
        action='store_true',
        help='update the parameter of the model based on failed actions during test'
    )

    parser.add_argument(
        '--memory-duplicate-learning',
        action='store_true',
        help='update the network include memory module to learn the duplicate steps',
    )

    parser.add_argument(
        '--memory-duplicate-rate',
        type=int,
        default=1,
        help='the weight of the memory duplicate loss'
    )

    parser.add_argument(
        '--imitation-learning',
        action='store_true',
        help='active imitation learning while the action failed'
    )

    parser.add_argument(
        '--il-duplicate-action',
        action='store_true',
        help='update network for duplicate actions based on optimal action'
    )

    parser.add_argument(
        '--il-failed-action',
        action='store_true',
        help='update network for failed actions based on optimal action'
    )

    parser.add_argument(
        '--il-each-action',
        action='store_true',
        help='lazy'
    )

    parser.add_argument(
        '--only-il-loss',
        action='store_true',
        help='only using imitation learning loss'
    )

    parser.add_argument(
        '--keep-ori-obs',
        action='store_true',
    )

    parser.add_argument(
        '--record-attention',
        action='store_true',
    )

    parser.add_argument(
        '--depth',
        action='store_true',
    )

    parser.add_argument(
        '--depth-maximum',
        type=int,
        default=0,
    )

    args = parser.parse_args()

    return args
