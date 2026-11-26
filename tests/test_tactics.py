"""Unit tests for the pure tactical decisions in ``src/tactics.py``.

These run off-robot: they only need the data contract, not the simulation. Run
with ``python3 -m unittest discover -s tests -t .`` from the project root.
"""

from __future__ import annotations

import math
import unittest

from src.framework.types import (
    ADULT_FIELD_DIMENSIONS,
    BallState,
    Context,
    GameControlState,
    GameState,
    KICKING_TEAM_NONE,
    Pose2D,
    RobotState,
    SetPlay,
)
from src.param import (
    GUARD_LINE_OFFSET_M,
    GUARD_POST_MARGIN_M,
    KICKOFF_CLEAR_M,
    SET_PLAY_CLEAR_M,
    SHOT_AIM_POST_MARGIN_M,
)
from src import tactics


FIELD = ADULT_FIELD_DIMENSIONS
HALF_L = FIELD.length / 2.0
HALF_W = FIELD.width / 2.0


def make_context(
    ball=(0.0, 0.0),
    opponents=None,
    team_id=1,
    kicking_team=KICKING_TEAM_NONE,
    set_play=SetPlay.NONE,
    state=GameState.PLAYING,
):
    return Context(
        now=0.0,
        dt=1.0 / 30.0,
        team_id=team_id,
        field=FIELD,
        game=GameControlState(
            state=state, kicking_team=kicking_team, set_play=set_play,
        ),
        ball=None if ball is None else BallState(x=ball[0], y=ball[1]),
        teammates={},
        opponents={
            index: RobotState(player_id=index, pose=Pose2D(x=x, y=y))
            for index, (x, y) in enumerate(opponents or [], start=1)
        },
    )


class KickoffPushTargetTest(unittest.TestCase):
    def test_push_moves_ball_forward_and_off_the_goal_mouth(self):
        target = tactics.kickoff_push_target(make_context())
        self.assertGreater(target[0], 0.5)
        self.assertGreaterEqual(abs(target[1]), 1.0)
        # A touch this far from the spot satisfies the 0.5 m rule on its own.
        self.assertGreaterEqual(
            math.hypot(target[0], target[1]), 0.5,
        )
        self.assertLessEqual(abs(target[0]), HALF_L)
        self.assertLessEqual(abs(target[1]), HALF_W)

    def test_push_avoids_the_side_with_the_nearest_opponent(self):
        target = tactics.kickoff_push_target(make_context(opponents=[(2.2, 1.6)]))
        self.assertLess(target[1], 0.0)

    def test_push_mirrors_when_the_other_side_is_crowded(self):
        target = tactics.kickoff_push_target(make_context(opponents=[(2.2, -1.6)]))
        self.assertGreater(target[1], 0.0)

    def test_push_without_a_ball_is_the_kickoff_point(self):
        self.assertEqual(
            tactics.kickoff_push_target(make_context(ball=None)), (0.0, 0.0),
        )


class BallLeftRestartTest(unittest.TestCase):
    def test_thresholds_stay_above_the_rule_minimum(self):
        self.assertGreater(KICKOFF_CLEAR_M, 0.5)
        self.assertGreater(SET_PLAY_CLEAR_M, 0.5)

    def test_ball_at_the_spot_is_not_clear(self):
        context = make_context(ball=(0.0, 0.0))
        self.assertFalse(tactics.ball_left_restart(context, (0.0, 0.0), 0.7))

    def test_ball_exactly_at_the_threshold_counts_as_clear(self):
        context = make_context(ball=(0.7, 0.0))
        self.assertTrue(tactics.ball_left_restart(context, (0.0, 0.0), 0.7))

    def test_missing_ball_is_never_clear(self):
        context = make_context(ball=None)
        self.assertFalse(tactics.ball_left_restart(context, (0.0, 0.0), 0.7))


class SetPlayPushTargetTest(unittest.TestCase):
    def test_goal_kick_never_pushes_toward_our_own_goal(self):
        # Goal kick spot, just inside our goal line.
        context = make_context(ball=(-6.2, 1.0))
        target = tactics.set_play_push_target(context)
        self.assertGreater(target[0], -6.2)
        # The first touch must travel into the opponent's half, never toward
        # our own goal, or it is an own goal (G-04).
        direction = math.atan2(target[1] - 1.0, target[0] + 6.2)
        self.assertGreater(math.cos(direction), 0.5)

    def test_corner_push_stays_inside_the_lines(self):
        context = make_context(ball=(6.4, 4.4))
        target = tactics.set_play_push_target(context)
        self.assertLessEqual(target[0], HALF_L - 0.6 + 1e-9)
        self.assertLessEqual(abs(target[1]), HALF_W)

    def test_push_distance_clears_the_restart_spot(self):
        for spot in [(-6.2, 1.0), (0.0, 4.4), (3.0, -3.0), (6.4, 4.4)]:
            context = make_context(ball=spot)
            target = tactics.set_play_push_target(context)
            self.assertGreaterEqual(
                math.hypot(target[0] - spot[0], target[1] - spot[1]), 0.5,
            )


class ChooseShotTargetTest(unittest.TestCase):
    def test_empty_goal_aims_close_to_centre(self):
        target = tactics.choose_shot_target(make_context(ball=(0.0, 0.0)))
        self.assertEqual(target[0], HALF_L)
        self.assertAlmostEqual(target[1], 0.0, places=6)

    def test_aim_moves_away_from_a_keeper_on_the_positive_side(self):
        context = make_context(ball=(0.0, 0.0), opponents=[(6.9, 0.6)])
        target = tactics.choose_shot_target(context)
        self.assertLess(target[1], -0.5)

    def test_aim_moves_away_from_a_keeper_on_the_negative_side(self):
        context = make_context(ball=(0.0, 0.0), opponents=[(6.9, -0.6)])
        target = tactics.choose_shot_target(context)
        self.assertGreater(target[1], 0.5)

    def test_aim_stays_inside_the_goal_mouth(self):
        context = make_context(ball=(0.0, 0.0), opponents=[(6.9, 0.0)])
        target = tactics.choose_shot_target(context)
        self.assertLessEqual(
            abs(target[1]), FIELD.goal_width / 2.0 - SHOT_AIM_POST_MARGIN_M + 1e-9,
        )

    def test_missing_ball_still_returns_the_goal_centre(self):
        target = tactics.choose_shot_target(make_context(ball=None))
        self.assertEqual(target, (HALF_L, 0.0))


class GuardTargetTest(unittest.TestCase):
    def test_keeper_stands_on_our_goal_line(self):
        target = tactics.guard_target(make_context(ball=(0.0, 0.0)))
        self.assertAlmostEqual(target[0], -HALF_L + GUARD_LINE_OFFSET_M)
        self.assertAlmostEqual(target[1], 0.0)

    def test_keeper_tracks_the_ball_laterally(self):
        context = make_context(ball=(0.0, 1.0))
        target = tactics.guard_target(context)
        self.assertGreater(target[1], 0.0)

    def test_keeper_stays_inside_the_goal_mouth(self):
        limit = FIELD.goal_width / 2.0 - GUARD_POST_MARGIN_M
        for ball_y in (4.5, -4.5, 2.0):
            target = tactics.guard_target(make_context(ball=(-2.0, ball_y)))
            self.assertLessEqual(abs(target[1]), limit + 1e-9)

    def test_no_ball_returns_the_middle_of_the_goal(self):
        target = tactics.guard_target(make_context(ball=None))
        self.assertAlmostEqual(target[1], 0.0)


class SecondAttackerTargetTest(unittest.TestCase):
    def test_slot_is_ahead_of_the_ball_and_inside_the_field(self):
        target = tactics.second_attacker_target(make_context(ball=(0.0, 0.0)))
        self.assertGreater(target[0], 0.0)
        self.assertGreaterEqual(abs(target[1]), 1.0)
        self.assertLessEqual(abs(target[0]), HALF_L)
        self.assertLessEqual(abs(target[1]), HALF_W)

    def test_slot_takes_the_side_away_from_the_nearest_opponent(self):
        context = make_context(ball=(0.0, 0.0), opponents=[(1.5, 2.0)])
        self.assertLess(tactics.second_attacker_target(context)[1], 0.0)


class ReadySlotsTest(unittest.TestCase):
    def test_three_players_get_three_distinct_slots(self):
        slots = tactics.ready_slots(make_context(kicking_team=1), [1, 2, 3])
        self.assertEqual(set(slots), {1, 2, 3})
        self.assertEqual(len(set(slots.values())), 3)

    def test_penalised_player_does_not_shift_the_others(self):
        full = tactics.ready_slots(make_context(kicking_team=1), [1, 2, 3])
        short = tactics.ready_slots(make_context(kicking_team=1), [1, 3])
        self.assertEqual(short, {1: full[1], 3: full[3]})

    def test_circle_slot_stays_outside_the_centre_circle(self):
        for kicking_team in (1, 2):
            slots = tactics.ready_slots(make_context(kicking_team=kicking_team), [1, 2, 3])
            for x, y in slots.values():
                self.assertGreaterEqual(
                    math.hypot(x, y), FIELD.circle_radius + 1e-9,
                )

    def test_every_slot_is_inside_the_field(self):
        for kicking_team in (1, 2):
            slots = tactics.ready_slots(make_context(kicking_team=kicking_team), [1, 2, 3])
            for x, y in slots.values():
                self.assertLess(abs(x), HALF_L)
                self.assertLess(abs(y), HALF_W)


class AssignNormalRolesTest(unittest.TestCase):
    def test_three_players_field_two_attackers_and_a_guard(self):
        context = make_context(ball=(0.0, 0.0))
        roles = tactics.assign_normal_roles(
            context, [(1, (-1.0, 0.5)), (2, (2.0, 1.0)), (3, (-6.0, 0.0))],
        )
        self.assertEqual(sorted(roles.values()), ["attack", "attack2", "guard"])

    def test_nearest_player_attacks_and_the_other_attacks_second(self):
        context = make_context(ball=(0.0, 0.0))
        roles = tactics.assign_normal_roles(
            context, [(1, (0.5, 0.0)), (2, (3.0, 0.0)), (3, (-6.0, 0.0))],
        )
        self.assertEqual(roles[1], "attack")
        self.assertEqual(roles[2], "attack2")
        self.assertEqual(roles[3], "guard")

    def test_two_players_keep_a_guard(self):
        context = make_context(ball=(0.0, 0.0))
        roles = tactics.assign_normal_roles(
            context, [(1, (1.0, 0.0)), (2, (-5.0, 0.0))],
        )
        self.assertEqual(sorted(roles.values()), ["attack", "guard"])

    def test_single_player_attacks(self):
        context = make_context(ball=(0.0, 0.0))
        roles = tactics.assign_normal_roles(context, [(2, (1.0, 0.0))])
        self.assertEqual(roles, {2: "attack"})

    def test_locked_attacker_keeps_the_ball_until_clearly_beaten(self):
        context = make_context(ball=(0.0, 0.0))
        roles = tactics.assign_normal_roles(
            context,
            [(1, (0.30, 0.0)), (2, (0.10, 0.0)), (3, (-6.0, 0.0))],
            locked_attacker=1,
        )
        self.assertEqual(roles[1], "attack")

    def test_locked_attacker_is_released_when_far_behind(self):
        context = make_context(ball=(0.0, 0.0))
        roles = tactics.assign_normal_roles(
            context,
            [(1, (3.0, 0.0)), (2, (0.2, 0.0)), (3, (-6.0, 0.0))],
            locked_attacker=1,
        )
        self.assertEqual(roles[2], "attack")

    def test_guard_is_the_player_nearest_our_goal(self):
        context = make_context(ball=(0.0, 0.0))
        roles = tactics.assign_normal_roles(
            context, [(1, (1.0, 0.0)), (2, (2.0, 0.0)), (3, (-6.0, 0.0))],
        )
        self.assertEqual(roles[3], "guard")

    def test_no_candidates_returns_no_roles(self):
        self.assertEqual(tactics.assign_normal_roles(make_context(), []), {})

    def test_missing_ball_assigns_everyone_to_guard(self):
        roles = tactics.assign_normal_roles(
            make_context(ball=None), [(1, (1.0, 0.0)), (2, (2.0, 0.0))],
        )
        self.assertEqual(set(roles.values()), {"guard"})


if __name__ == "__main__":
    unittest.main()
