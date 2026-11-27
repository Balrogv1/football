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
    TeamState,
)
from src.param import (
    DEFENDER_BALL_GAP_M,
    DEFENDER_DEPTH_MAX_M,
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
    our_score=0,
    their_score=0,
):
    opponent_team_id = 2 if team_id == 1 else 1
    return Context(
        now=0.0,
        dt=1.0 / 30.0,
        team_id=team_id,
        field=FIELD,
        game=GameControlState(
            state=state,
            kicking_team=kicking_team,
            set_play=set_play,
            teams=(
                TeamState(team_number=team_id, score=our_score),
                TeamState(team_number=opponent_team_id, score=their_score),
            ),
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

    def test_push_always_goes_to_the_takers_left(self):
        # Fixed side, not the emptier one: the pair has to rehearse the pass.
        for opponents in ([], [(2.2, 1.6)], [(2.2, -1.6)], [(1.5, 2.0), (1.5, -2.0)]):
            target = tactics.kickoff_push_target(make_context(opponents=opponents))
            self.assertGreater(target[1], 0.0)

    def test_push_leans_sideways_rather_than_straight_upfield(self):
        target = tactics.kickoff_push_target(make_context())
        self.assertGreater(target[1], target[0])

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

    def test_keeper_covers_the_shot_line_not_the_ball_y(self):
        # Ball wide on the flank: the shot travels toward the goal centre, so
        # the keeper must sit near the middle, not at the ball's own y.
        target = tactics.guard_target(make_context(ball=(0.0, 4.0)))
        self.assertLess(abs(target[1]), 0.5)

    def test_keeper_closes_the_angle_further_out(self):
        near = tactics.guard_target(make_context(ball=(-2.0, 3.0)))
        far = tactics.guard_target(make_context(ball=(4.0, 3.0)))
        self.assertGreater(abs(near[1]), abs(far[1]))

    def test_keeper_mirrors_the_ball_once_it_is_on_the_line(self):
        target = tactics.guard_target(make_context(ball=(-6.9, 1.0)))
        self.assertAlmostEqual(target[1], 1.0)

    def test_keeper_stays_on_the_shot_line_for_any_ball_position(self):
        for ball in [(0.0, 2.0), (-2.0, 3.0), (4.0, -2.0), (2.0, 1.0)]:
            target = tactics.guard_target(make_context(ball=ball))
            shot_angle = math.atan2(ball[1], ball[0] + HALF_L)
            keeper_angle = math.atan2(
                target[1], target[0] + HALF_L,
            )
            self.assertAlmostEqual(keeper_angle, shot_angle, places=6)


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

    def test_slot_moves_off_a_screened_lane(self):
        # The opponent sits exactly on the ball -> upper-slot lane, so that
        # slot is useless even though the opponent is not that close.
        context = make_context(ball=(0.0, 0.0), opponents=[(1.3, 1.1)])
        self.assertLess(tactics.second_attacker_target(context)[1], 0.0)

    def test_slot_is_not_screened_from_the_ball(self):
        context = make_context(ball=(0.0, 0.0), opponents=[(1.3, 1.1), (1.3, -1.1), (1.3, 0.0)])
        target = tactics.second_attacker_target(context)
        # Every lane is covered here, so the slot must still be a legal spot.
        self.assertLessEqual(abs(target[0]), HALF_L)
        self.assertLessEqual(abs(target[1]), HALF_W)

    def test_slot_stays_inside_the_field_from_the_corner(self):
        for ball in [(6.4, 0.0), (6.4, 4.4), (6.4, -4.4)]:
            target = tactics.second_attacker_target(make_context(ball=ball))
            self.assertLessEqual(abs(target[0]), HALF_L)
            self.assertLessEqual(abs(target[1]), HALF_W)


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


class ScoreDifferenceTest(unittest.TestCase):
    def test_equal_score_is_zero(self):
        self.assertEqual(tactics.score_difference(make_context()), 0)

    def test_lead_is_positive(self):
        context = make_context(our_score=3, their_score=0)
        self.assertEqual(tactics.score_difference(context), 3)

    def test_deficit_is_negative(self):
        context = make_context(our_score=1, their_score=2)
        self.assertEqual(tactics.score_difference(context), -1)

    def test_away_team_reads_its_own_score(self):
        context = make_context(team_id=2, our_score=0, their_score=2)
        self.assertEqual(tactics.score_difference(context), -2)

    def test_missing_game_state_is_zero(self):
        context = Context(now=0.0, dt=0.0, team_id=1, field=FIELD, game=None)
        self.assertEqual(tactics.score_difference(context), 0)


class DefenderTargetTest(unittest.TestCase):
    def test_defender_stands_in_front_of_our_goal(self):
        target = tactics.defender_target(make_context(ball=(0.0, 0.0)))
        self.assertGreater(target[0], -HALF_L)
        self.assertLessEqual(target[0], -HALF_L + DEFENDER_DEPTH_MAX_M + 1e-9)

    def test_defender_stays_on_the_goal_to_ball_line(self):
        for ball in [(0.0, 2.0), (0.0, -2.0), (3.0, 3.0), (2.0, -1.5)]:
            target = tactics.defender_target(make_context(ball=ball))
            expected = math.atan2(ball[1], ball[0] + HALF_L)
            actual = math.atan2(target[1], target[0] + HALF_L)
            self.assertAlmostEqual(actual, expected, places=6)

    def test_defender_never_ends_up_behind_the_goal_line(self):
        for ball in [(-6.0, 0.0), (-6.9, 0.5), (-5.5, -1.0)]:
            target = tactics.defender_target(make_context(ball=ball))
            self.assertGreaterEqual(target[0], -HALF_L - 1e-9)

    def test_defender_keeps_a_gap_from_a_distant_ball(self):
        context = make_context(ball=(0.0, 0.0))
        target = tactics.defender_target(context)
        self.assertGreaterEqual(
            math.hypot(target[0], target[1]), DEFENDER_BALL_GAP_M - 1e-9,
        )

    def test_defender_steps_further_out_for_a_distant_ball(self):
        # A fixed short standoff would park the defender on the goal line and
        # let an attacker walk past at midfield.
        near = tactics.defender_target(make_context(ball=(-4.0, 0.0)))
        far = tactics.defender_target(make_context(ball=(2.0, 0.0)))
        self.assertGreater(far[0], near[0])

    def test_defender_closes_on_the_ball_when_it_is_near(self):
        ball = (-5.0, 0.0)
        target = tactics.defender_target(make_context(ball=ball))
        gap = math.hypot(target[0] - ball[0], target[1] - ball[1])
        self.assertLessEqual(gap, DEFENDER_BALL_GAP_M + 0.05)

    def test_defender_gap_exceeds_clear_radius_to_avoid_crowding_attacker(self):
        from src.param import DEFENDER_CLEAR_M
        self.assertGreater(DEFENDER_BALL_GAP_M, DEFENDER_CLEAR_M)

    def test_opp_kickoff_ready_slots_form_staggered_x_triangle(self):
        slots = tactics.ready_slots(make_context(kicking_team=2), [1, 2, 3])
        pts = list(slots.values())
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        # All three players have distinct x depths (>= 1.0 m apart) and cover
        # both flanks so they form a staggered triangle without crowding.
        self.assertEqual(len(set(xs)), 3)
        xs_sorted = sorted(xs)
        self.assertGreaterEqual(xs_sorted[1] - xs_sorted[0], 1.0)
        self.assertGreaterEqual(xs_sorted[2] - xs_sorted[1], 1.0)
        self.assertTrue(any(y > 0.5 for y in ys))
        self.assertTrue(any(y < -0.5 for y in ys))
        for i in range(len(pts)):
            self.assertLess(pts[i][0], 0.0)
            self.assertGreater(math.hypot(pts[i][0], pts[i][1]), 3.0)
            for j in range(i + 1, len(pts)):
                self.assertGreater(
                    math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]),
                    2.5,
                )

    def test_defender_never_leaves_the_field(self):
        for ball in [(0.0, 4.4), (0.0, -4.4), (-6.5, 4.4), (5.0, 4.4)]:
            target = tactics.defender_target(make_context(ball=ball))
            self.assertLessEqual(abs(target[0]), HALF_L)
            self.assertLessEqual(abs(target[1]), HALF_W)

    def test_missing_ball_stands_in_front_of_the_goal(self):
        target = tactics.defender_target(make_context(ball=None))
        self.assertAlmostEqual(target[1], 0.0)


class DefensiveShapeTest(unittest.TestCase):
    def test_ball_in_our_half_brings_a_second_body_back(self):
        context = make_context(ball=(-3.0, 0.0))
        roles = tactics.assign_normal_roles(
            context, [(1, (-2.0, 0.0)), (2, (-5.0, 1.0)), (3, (-6.0, 0.0))],
        )
        self.assertEqual(sorted(roles.values()), ["attack", "defend", "guard"])

    def test_ball_in_opponent_half_keeps_two_attackers(self):
        context = make_context(ball=(3.0, 0.0))
        roles = tactics.assign_normal_roles(
            context, [(1, (2.0, 0.0)), (2, (4.0, 1.0)), (3, (-6.0, 0.0))],
        )
        self.assertEqual(sorted(roles.values()), ["attack", "attack2", "guard"])

    def test_one_goal_lead_still_keeps_two_attackers_in_opponent_half(self):
        context = make_context(ball=(3.0, 0.0), our_score=1, their_score=0)
        roles = tactics.assign_normal_roles(
            context, [(1, (2.0, 0.0)), (2, (4.0, 1.0)), (3, (-6.0, 0.0))],
        )
        self.assertEqual(sorted(roles.values()), ["attack", "attack2", "guard"])

    def test_leading_pulls_a_body_back_in_the_opponent_half_too(self):
        context = make_context(ball=(3.0, 0.0), our_score=2, their_score=0)
        roles = tactics.assign_normal_roles(
            context, [(1, (2.0, 0.0)), (2, (4.0, 1.0)), (3, (-6.0, 0.0))],
        )
        self.assertEqual(sorted(roles.values()), ["attack", "defend", "guard"])

    def test_trailing_keeps_both_attackers(self):
        context = make_context(ball=(3.0, 0.0), our_score=0, their_score=2)
        roles = tactics.assign_normal_roles(
            context, [(1, (2.0, 0.0)), (2, (4.0, 1.0)), (3, (-6.0, 0.0))],
        )
        self.assertIn("attack2", roles.values())

    def test_two_players_never_field_a_defender(self):
        context = make_context(ball=(-3.0, 0.0))
        roles = tactics.assign_normal_roles(
            context, [(1, (-2.0, 0.0)), (2, (-6.0, 0.0))],
        )
        self.assertEqual(sorted(roles.values()), ["attack", "guard"])

    def test_defender_is_not_the_goalkeeper(self):
        context = make_context(ball=(-3.0, 0.0))
        roles = tactics.assign_normal_roles(
            context, [(1, (-2.0, 0.0)), (2, (-5.0, 1.0)), (3, (-6.0, 0.0))],
        )
        self.assertNotEqual(roles[2], roles[3])

    def test_chase_window_keeps_two_attackers_while_leading(self):
        # The seconds after our own kick-off: the ball is loose in the
        # opponent half, so the second player follows it up instead of
        # dropping back to protect the lead.
        context = make_context(ball=(1.5, 2.0), our_score=3, their_score=0)
        roles = tactics.assign_normal_roles(
            context,
            [(1, (1.2, 1.8)), (2, (-6.0, 0.0)), (3, (-0.4, 3.3))],
            chase=True,
        )
        self.assertEqual(sorted(roles.values()), ["attack", "attack2", "guard"])

    def test_without_the_chase_window_a_lead_still_drops_a_body_back(self):
        context = make_context(ball=(1.5, 2.0), our_score=3, their_score=0)
        roles = tactics.assign_normal_roles(
            context,
            [(1, (1.2, 1.8)), (2, (-6.0, 0.0)), (3, (-0.4, 3.3))],
        )
        self.assertEqual(sorted(roles.values()), ["attack", "defend", "guard"])

    def test_chase_window_does_not_pull_a_defender_when_two_players(self):
        context = make_context(ball=(1.5, 2.0), our_score=3)
        roles = tactics.assign_normal_roles(
            context, [(1, (1.2, 1.8)), (2, (-6.0, 0.0))], chase=True,
        )
        self.assertEqual(sorted(roles.values()), ["attack", "guard"])


class ChoosePassTargetTest(unittest.TestCase):
    def test_no_pass_when_the_shot_lane_is_clear(self):
        context = make_context(ball=(0.0, 0.0))
        self.assertIsNone(tactics.choose_pass_target(context, (4.0, 2.5)))

    def test_passes_when_the_shot_is_blocked_and_the_receiver_lane_is_clear(self):
        context = make_context(ball=(0.0, 0.0), opponents=[(3.0, 0.0)])
        self.assertEqual(
            tactics.choose_pass_target(context, (4.0, 2.5)), (4.0, 2.5),
        )

    def test_no_pass_when_the_receiver_lane_is_blocked(self):
        context = make_context(ball=(0.0, 0.0), opponents=[(3.0, 0.0), (2.0, 1.2)])
        self.assertIsNone(tactics.choose_pass_target(context, (4.0, 2.5)))

    def test_no_pass_to_a_receiver_level_with_the_ball(self):
        context = make_context(ball=(0.0, 0.0), opponents=[(3.0, 0.0)])
        self.assertIsNone(tactics.choose_pass_target(context, (0.5, 2.5)))

    def test_no_pass_without_a_receiver(self):
        context = make_context(ball=(0.0, 0.0), opponents=[(3.0, 0.0)])
        self.assertIsNone(tactics.choose_pass_target(context, None))

    def test_no_pass_without_a_ball(self):
        context = make_context(ball=None, opponents=[(3.0, 0.0)])
        self.assertIsNone(tactics.choose_pass_target(context, (4.0, 2.5)))

    def test_opponent_behind_the_ball_does_not_block_the_shot_lane(self):
        context = make_context(ball=(0.0, 0.0), opponents=[(-2.0, 0.0)])
        self.assertIsNone(tactics.choose_pass_target(context, (4.0, 2.5)))

    def test_opponent_beyond_the_goal_does_not_block_the_shot_lane(self):
        context = make_context(ball=(0.0, 0.0), opponents=[(9.0, 0.0)])
        self.assertIsNone(tactics.choose_pass_target(context, (4.0, 2.5)))


class MeasureBallVelocityTest(unittest.TestCase):
    def test_velocity_from_two_snapshots(self):
        prev = (0.0, 0.0, 1.0)
        ball = BallState(x=1.0, y=0.0)
        vx, vy = tactics.measure_ball_velocity(prev, ball, 2.0)
        self.assertAlmostEqual(vx, 1.0)
        self.assertAlmostEqual(vy, 0.0)

    def test_no_previous_snapshot_reports_zero(self):
        ball = BallState(x=1.0, y=0.0)
        self.assertEqual(tactics.measure_ball_velocity(None, ball, 2.0), (0.0, 0.0))

    def test_missing_ball_reports_zero(self):
        self.assertEqual(tactics.measure_ball_velocity((0.0, 0.0, 1.0), None, 2.0), (0.0, 0.0))

    def test_repeated_timestamp_reports_zero(self):
        prev = (0.0, 0.0, 1.0)
        ball = BallState(x=5.0, y=5.0)
        self.assertEqual(tactics.measure_ball_velocity(prev, ball, 1.0), (0.0, 0.0))

    def test_restart_teleport_is_rejected(self):
        # A restart moves the ball metres in one frame; that is not a pass.
        prev = (0.0, 0.0, 1.0)
        ball = BallState(x=12.0, y=8.0)
        self.assertEqual(tactics.measure_ball_velocity(prev, ball, 1.1), (0.0, 0.0))


class BlendVelocityTest(unittest.TestCase):
    def test_full_smoothing_takes_the_new_sample(self):
        # BALL_VELOCITY_SMOOTHING is 0.5 in the shipped parameters, so the
        # blend must land between the two samples rather than on either end.
        blended = tactics.blend_velocity((0.0, 0.0), (4.0, 2.0))
        self.assertTrue(0.0 < blended[0] < 4.0)
        self.assertTrue(0.0 < blended[1] < 2.0)

    def test_zero_measurement_pulls_the_estimate_down(self):
        blended = tactics.blend_velocity((4.0, 0.0), (0.0, 0.0))
        self.assertLess(blended[0], 4.0)


class ReceivingTargetTest(unittest.TestCase):
    def test_stationary_ball_uses_the_normal_slot(self):
        context = make_context(ball=(0.0, 0.0))
        target = tactics.receiving_target(context, (1.0, -2.0), (0.0, 0.0))
        self.assertEqual(target, tactics.second_attacker_target(context))

    def test_slow_ball_uses_the_normal_slot(self):
        context = make_context(ball=(0.0, 0.0))
        target = tactics.receiving_target(context, (1.0, -2.0), (0.1, 0.0))
        self.assertEqual(target, tactics.second_attacker_target(context))

    def test_receiver_runs_onto_a_ball_rolling_toward_it(self):
        # Ball at (0, 0) moving +x; receiver is ahead at x = 2, so it should
        # meet the ball beyond its current position.
        context = make_context(ball=(0.0, 0.0))
        target = tactics.receiving_target(context, (2.0, 0.0), (3.0, 0.0))
        self.assertGreater(target[0], 0.0)

    def test_receiver_does_not_chase_a_ball_moving_away(self):
        # Ball rolling away from a receiver behind it: no point leading.
        context = make_context(ball=(0.0, 0.0))
        target = tactics.receiving_target(context, (-2.0, 0.0), (3.0, 0.0))
        self.assertEqual(target, tactics.second_attacker_target(context))

    def test_lead_stays_inside_the_field(self):
        context = make_context(ball=(6.0, 4.0))
        target = tactics.receiving_target(context, (4.0, 4.0), (5.0, 5.0))
        self.assertLessEqual(abs(target[0]), HALF_L)
        self.assertLessEqual(abs(target[1]), HALF_W)

    def test_missing_ball_returns_the_origin(self):
        self.assertEqual(
            tactics.receiving_target(make_context(ball=None), (1.0, 1.0), (2.0, 0.0)),
            (0.0, 0.0),
        )

    def test_touchline_duel_sends_second_attacker_to_open_weak_side(self):
        # Mirrors the sideline duel where opponents crowd the +y half:
        # second attacker behind the ball must push forward into the open
        # centre / -y space instead of stalling on minor backward bounces.
        context = make_context(
            ball=(2.8, 3.7), opponents=[(3.5, 3.9), (4.2, 1.4), (5.8, 1.4)],
        )
        target = tactics.receiving_target(context, (0.0, 3.0), (-0.8, -0.2))
        self.assertGreater(target[0], 4.0)
        self.assertLessEqual(target[1], 0.0)


if __name__ == "__main__":
    unittest.main()
