import random


class APEXController:

    def __init__(
        self,
        examples,
        sample_size=50,
        evaluation_size=30,
        lookback=4,
        seed=42,
    ):
        self.examples = examples
        self.sample_size = sample_size
        self.lookback = lookback
        self.evaluation_size = evaluation_size
        self.rng = random.Random(seed)

        # Stores the performance history of each example.
        #
        # Example:
        # {
        #     "pair_001": [
        #         {"score": 1.0, "passed": True},
        #         {"score": 0.0, "passed": False},
        #     ]
        # }
        self.history = {}

    def _example_id(self, example):
        return example.pair_name

    def record_result(self, example, score, passed):
        """
        Record the result obtained by the current optimized
        comparator for one example.
        """

        example_id = self._example_id(example)

        if example_id not in self.history:
            self.history[example_id] = []

        self.history[example_id].append(
            {
                "score": score,
                "passed": passed,
            }
        )

    def classify(self, example):
        """
        Classify an example based on its recent performance history.

        Returns:
            unseen
            easy
            mixed
            hard
        """

        example_id = self._example_id(example)

        results = self.history.get(example_id, [])

        # No previous observation.
        if len(results) == 0:
            return "unseen"

        # Use only the most recent observations.
        recent = results[-self.lookback:]

        # Not enough history yet.
        if len(recent) < self.lookback:
            return "unseen"

        passed_values = [
            result["passed"]
            for result in recent
        ]

        # Passed consistently.
        if all(passed_values):
            return "easy"

        # Failed consistently.
        if not any(passed_values):
            return "hard"

        # Combination of passes and failures.
        return "mixed"

    def select_training_examples(self):
        """
        Select the next GEPA training set.

        Priority:
            1. Mixed examples
            2. Hard examples
            3. Unseen / remaining examples

        The selected set always contains at most
        self.sample_size examples.
        """

        mixed_examples = []
        hard_examples = []
        remaining_examples = []

        for example in self.examples:

            tier = self.classify(example)

            if tier == "mixed":
                mixed_examples.append(example)

            elif tier == "hard":
                hard_examples.append(example)

            else:
                remaining_examples.append(example)

        # Randomize examples within each tier.
        self.rng.shuffle(mixed_examples)
        self.rng.shuffle(hard_examples)
        self.rng.shuffle(remaining_examples)

        selected = []

        # -----------------------------------------------------
        # Priority 1: Mixed examples
        # -----------------------------------------------------

        for example in mixed_examples:

            if len(selected) >= self.sample_size:
                break

            selected.append(example)

        # -----------------------------------------------------
        # Priority 2: Hard examples
        # -----------------------------------------------------

        for example in hard_examples:

            if len(selected) >= self.sample_size:
                break

            selected.append(example)

        # -----------------------------------------------------
        # Priority 3: Remaining / unseen examples
        # -----------------------------------------------------

        for example in remaining_examples:

            if len(selected) >= self.sample_size:
                break

            selected.append(example)

        return selected

    def print_status(self):
        """
        Print the current distribution of examples
        across APEX difficulty tiers.
        """

        counts = {
            "easy": 0,
            "mixed": 0,
            "hard": 0,
            "unseen": 0,
        }

        for example in self.examples:

            tier = self.classify(example)

            counts[tier] += 1

        print("\n" + "=" * 70)
        print("APEX STATUS")
        print("=" * 70)

        print(f"Easy   : {counts['easy']}")
        print(f"Mixed  : {counts['mixed']}")
        print(f"Hard   : {counts['hard']}")
        print(f"Unseen : {counts['unseen']}")

        print("=" * 70)
        
    def select_evaluation_examples(self, evaluation_pool):
        """
        Select a small dynamic evaluation subset.

        Priority follows the APEX rank-sensitive idea:

        1. Mixed examples not yet evaluated in the current round
        2. Mixed examples that previously passed
        3. Easy examples not yet evaluated
        4. Mixed examples that previously failed
        5. Hard examples not yet evaluated
        6. Remaining examples

        Returns:
            A list containing at most self.evaluation_size examples.
        """

        required_mixed = []
        mixed_passed = []
        easy_unseen = []
        mixed_failed = []
        hard_unseen = []
        remaining = []

        for example in evaluation_pool:

            example_id = self._example_id(example)
            results = self.history.get(example_id, [])

            tier = self.classify(example)

            # -------------------------------------------------
            # No history
            # -------------------------------------------------

            if not results:

                if tier == "easy":
                    easy_unseen.append(example)

                elif tier == "hard":
                    hard_unseen.append(example)

                else:
                    remaining.append(example)

                continue

            # -------------------------------------------------
            # Last known result
            # -------------------------------------------------

            current_passed = results[-1]["passed"]

            # Mixed examples are the highest-value group.
            if tier == "mixed":

                if current_passed:
                    mixed_passed.append(example)
                else:
                    mixed_failed.append(example)

            elif tier == "easy":

                easy_unseen.append(example)

            elif tier == "hard":

                hard_unseen.append(example)

            else:

                remaining.append(example)

        # Randomize within each priority bucket.
        self.rng.shuffle(required_mixed)
        self.rng.shuffle(mixed_passed)
        self.rng.shuffle(easy_unseen)
        self.rng.shuffle(mixed_failed)
        self.rng.shuffle(hard_unseen)
        self.rng.shuffle(remaining)

        selected = []

        def add_examples(bucket):

            for example in bucket:

                if len(selected) >= self.evaluation_size:
                    break

                selected.append(example)

        # -----------------------------------------------------
        # APEX rank-sensitive priority
        # -----------------------------------------------------

        add_examples(required_mixed)

        add_examples(mixed_passed)

        add_examples(easy_unseen)

        add_examples(mixed_failed)

        add_examples(hard_unseen)

        add_examples(remaining)

        return selected    