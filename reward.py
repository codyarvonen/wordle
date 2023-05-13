# Option 1: Only reward when puzzle is complete, give reward based on the number of guesses used
# Option 2: Reward upon completion as per option 1 but also give smaller intermediate rewards for green or yellow tiles


class ActionReward:
    def __init__(self, history: list[str], word: str, num_guesses: int, is_valid):
        self.history = history
        self.word = word
        self.num_guesses = num_guesses
        self.is_valid = is_valid

    def get_completion_factor(self):
        if self.history[-1] == self.word:
            return 100 / (len(self.history) ** 2)
        elif len(self.history) == self.num_guesses:
            return -10
        return 0
        
    def get_num_current_green(self):
        count = 0
        for i in range(len(self.word)):
            if self.history[-1][i] == self.word[i]:
                count += 1
        return count

    def get_num_total_green(self):
        count = 0
        for i in range(len(self.history)):
            for j in range(len(self.word)):
                if self.history[i][j] == self.word[j]:
                    count += 1
        return count

    def get_num_current_yellow(self):
        count = 0
        for i in range(len(self.word)):
            if self.history[-1][i] != self.word[i] and self.history[-1].count(self.history[-1][i]) <= self.word.count(self.history[-1][i]):
                count += 1
        return count

    def get_num_total_yellow(self):
        count = 0
        for i in range(len(self.history)):
            for j in range(len(self.word)):
                if self.history[j][i] != self.word[i] and self.history[j].count(self.history[j][i]) <= self.word.count(self.history[j][i]):
                    count += 1
        return count

    def get_total_reward(self, is_basic: bool):
        if not self.is_valid:
            return -10
        if is_basic:
            return self.get_completion_factor()
        return self.get_completion_factor() + self.get_num_current_green()


    # TODO: Add reward test functions!!!!
