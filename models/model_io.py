class ModelOptions:
    def __init__(self, params=None):
        self.params = params


class ModelInput:
    """ Input to the model. """

    def __init__(
            self, state=None, hidden=None, target_class_embedding=None, action_probs=None, states_memory=None,
            action_memory=None, states_rep=None, obs_reps=None, depth=None, glove=None, scene=None, target_object=None
    ):
        self.state = state
        self.hidden = hidden
        self.target_class_embedding = target_class_embedding
        self.action_probs = action_probs
        self.states_memory = states_memory
        self.action_memory = action_memory
        self.states_rep = states_rep
        self.obs_reps = obs_reps
        self.depth = depth
        self.glove = glove
        self.scene = scene
        self.target_object = target_object


class ModelOutput:
    """ Output from the model. """

    def __init__(self, value=None, logit=None, hidden=None, state_representation=None, embedding=None,
                 state_memory=None, action_memory=None, meta_action=None, visual_info=None, obs_rep=None, match_score=None):
        self.value = value
        self.logit = logit
        self.hidden = hidden
        self.state_representation = state_representation
        self.embedding = embedding
        self.state_memory = state_memory
        self.action_memory = action_memory
        self.meta_action = meta_action
        self.visual_info = visual_info,
        self.obs_rep = obs_rep
        self.match_score = match_score
