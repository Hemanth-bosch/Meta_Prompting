from apex_dca import APEXController


class MockExample:

    def __init__(self, pair_name):
        self.pair_name = pair_name


# Create 20 fake DCA examples
examples = [
    MockExample(f"pair_{i}")
    for i in range(20)
]


# Create APEX controller
controller = APEXController(
    examples=examples,
    sample_size=5,
    lookback=4,
    seed=42,
)


print("\n" + "=" * 70)
print("APEX CONTROLLER TEST")
print("=" * 70)


# ---------------------------------------------------------
# ROUND 1
# ---------------------------------------------------------

round_1 = controller.select_training_examples()

print("\nROUND 1")
print("-" * 70)

print("Selected examples:")
print([
    example.pair_name
    for example in round_1
])

controller.print_status()


# ---------------------------------------------------------
# Create artificial history
# ---------------------------------------------------------

# Make some examples consistently fail.
for example in examples[:5]:

    for _ in range(4):

        controller.record_result(
            example=example,
            score=0.0,
            passed=False,
        )


# Make some examples mixed.
for example in examples[5:10]:

    controller.record_result(
        example=example,
        score=0.0,
        passed=False,
    )

    controller.record_result(
        example=example,
        score=1.0,
        passed=True,
    )

    controller.record_result(
        example=example,
        score=0.0,
        passed=False,
    )

    controller.record_result(
        example=example,
        score=1.0,
        passed=True,
    )


# Make some examples easy.
for example in examples[10:15]:

    for _ in range(4):

        controller.record_result(
            example=example,
            score=1.0,
            passed=True,
        )


# ---------------------------------------------------------
# ROUND 2
# ---------------------------------------------------------

round_2 = controller.select_training_examples()

print("\nROUND 2")
print("-" * 70)

print("Selected examples:")
print([
    example.pair_name
    for example in round_2
])

controller.print_status()


print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)