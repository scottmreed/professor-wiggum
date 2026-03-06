# Contribution Templates

Copy the template that matches your contribution track.

| Track | Template |
| --- | --- |
| Track 1: Few-shot examples | [track1_few_shot_pr.md](/Users/scottreed/PycharmProjects/Mechanistic/templates/contributions/track1_few_shot_pr.md) |
| Track 2: New subagents | [track2_subagent_pr.md](/Users/scottreed/PycharmProjects/Mechanistic/templates/contributions/track2_subagent_pr.md) |
| Track 3: New models | [track3_model_pr.md](/Users/scottreed/PycharmProjects/Mechanistic/templates/contributions/track3_model_pr.md) |
| Track 4: Harness changes | [track4_harness_pr.md](/Users/scottreed/PycharmProjects/Mechanistic/templates/contributions/track4_harness_pr.md) |
| Track 5: Single reaction submission | [track5_single_reaction_submission.md](/Users/scottreed/PycharmProjects/Mechanistic/templates/contributions/track5_single_reaction_submission.md) |

Track 5 files should be copied into `local_contributions/` and kept out of version control.

Trainee-track contributions should prefer exact-model overrides under
`skills/mechanistic/<call_name>/models/<model-slug>/` so changes do not leak
across model families or exact models. Reference a curriculum checkpoint id,
eval run id, or checkpoint manifest path in the PR when possible.
