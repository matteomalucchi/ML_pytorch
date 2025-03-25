# Courtesy of:
# https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
# Note, that the structure currently only allows distinction between two different parameters.
# If a 3rd is added, we will have to think about a good solution.

class EarlyStopper:
    def __init__(self, logger, patience=5, min_delta=1e-5, eval_param="loss"):
        assert eval_param in ["acc", "loss"]
        self.logger = logger
        self.patience = patience
        self.min_delta = min_delta

        self.eval_param = eval_param
        self.better_than = lambda a, b, param: a >= b + min_delta if param == "acc" else a <= b - min_delta

        self.counter = 0
        self.min_validation = float('inf') if eval_param == "loss" else -float('inf')

    def early_stop(self, validation,epoch):
        if epoch >= 6:
            self.logger.info(f"After: {validation}")
            self.logger.info(f"Before: {self.min_validation}")
            if self.better_than(validation, self.min_validation, self.eval_param):
                self.logger.info("Better")
                self.min_validation = validation
                self.counter = 0 
            else:
                self.logger.info("Worse")
                self.counter += 1
                if self.counter >= self.patience:
                    return True
        return False
