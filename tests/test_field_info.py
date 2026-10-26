import json
import unittest
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock

from src.framework.config import SoccerConfig
from src.framework.field_codec import field_dimensions_from_json
from src.framework.runtime import SoccerRuntime
from src.framework.types import ADULT_FIELD_DIMENSIONS, WorldSnapshot

try:
    from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
    from std_msgs.msg import String
    from src.framework.ros_source import RosContextSource
except ImportError:
    RosContextSource = None


def payload(**changes):
    dimensions = asdict(ADULT_FIELD_DIMENSIONS)
    dimensions.update(changes)
    return json.dumps({"schema_version": 1, "dimensions": dimensions})


def make_runtime(source):
    agent = SimpleNamespace(
        config=SoccerConfig(),
        player_class=lambda **kwargs: SimpleNamespace(context=None),
        play=Mock(),
    )
    return SoccerRuntime(agent, context_source=source)


class FieldCodecTests(unittest.TestCase):
    def test_all_published_dimensions_are_used(self):
        dimensions = field_dimensions_from_json(payload(
            length=18, width=12, goal_height=2.2, goal_depth=0.9,
        ))
        self.assertEqual(dimensions, replace(
            ADULT_FIELD_DIMENSIONS,
            length=18.0, width=12.0, goal_height=2.2, goal_depth=0.9,
        ))

    def test_rejects_invalid_envelopes(self):
        for data in (
            "{", "[]", "null", "{}",
            '{"schema_version":2,"dimensions":{}}',
            '{"schema_version":true,"dimensions":{}}',
            '{"schema_version":1,"dimensions":[]}',
            '{"schema_version":1,"dimensions":{}}',
        ):
            with self.subTest(data=data), self.assertRaises(ValueError):
                field_dimensions_from_json(data)

    def test_rejects_missing_and_invalid_dimensions(self):
        for key in asdict(ADULT_FIELD_DIMENSIONS):
            data = json.loads(payload())
            del data["dimensions"][key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                field_dimensions_from_json(json.dumps(data))
            for value in (None, True, "14", [], {}, 0, -1, float("nan"),
                          float("inf"), 10 ** 400):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    field_dimensions_from_json(payload(**{key: value}))


class RuntimeFieldTests(unittest.TestCase):
    def test_missing_topic_uses_default_without_blocking_strategy(self):
        source = SimpleNamespace(get_snapshot=lambda: WorldSnapshot())
        runtime = make_runtime(source)
        self.assertIs(runtime._build_context(100.0, 0.03).field, ADULT_FIELD_DIMENSIONS)
        runtime._draw_world = Mock()
        runtime._tick(100.0)
        runtime._tick(10000.0)
        self.assertEqual(runtime._agent.play.call_count, 2)
        for call in runtime._agent.play.call_args_list:
            self.assertIs(call.args[0].field, ADULT_FIELD_DIMENSIONS)

    def test_uses_latest_field_without_freshness_timeout(self):
        source = SimpleNamespace(snapshot=WorldSnapshot(field=ADULT_FIELD_DIMENSIONS))
        source.get_snapshot = lambda: source.snapshot
        runtime = make_runtime(source)
        initial = runtime._build_context(100.0, 0.03)
        updated = replace(ADULT_FIELD_DIMENSIONS, length=18.0, goal_depth=0.9)
        source.snapshot = WorldSnapshot(field=updated)
        self.assertIs(runtime._build_context(10000.0, 0.03).field, updated)
        self.assertIs(initial.field, ADULT_FIELD_DIMENSIONS)
        runtime._draw_world = Mock()
        runtime._tick(10001.0)
        runtime._agent.play.assert_called_once()
        self.assertIs(runtime._agent.play.call_args.args[0].field, updated)

    def test_offline_default(self):
        runtime = make_runtime(None)
        self.assertIs(runtime._build_context(100.0, 0.03).field, ADULT_FIELD_DIMENSIONS)

@unittest.skipIf(RosContextSource is None, "ROS Python packages unavailable")
class RosFieldTests(unittest.TestCase):
    def test_subscription_qos_and_json_callback(self):
        source = RosContextSource(SoccerConfig())
        source._node = Mock()
        source._create_subscriptions()
        subscription = next(
            call.args for call in source._node.create_subscription.call_args_list
            if call.args[1] == "/soccer/field_info"
        )
        msg_type, _, callback, qos = subscription
        self.assertIs(msg_type, String)
        self.assertEqual(qos.history, QoSHistoryPolicy.KEEP_LAST)
        self.assertEqual(qos.depth, 1)
        self.assertEqual(qos.durability, QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.assertEqual(qos.reliability, QoSReliabilityPolicy.RELIABLE)
        self.assertIsNone(source.get_snapshot().field)

        runtime = make_runtime(source)
        initial = runtime._build_context(100.0, 0.03)
        self.assertIs(initial.field, ADULT_FIELD_DIMENSIONS)
        with self.assertLogs("src.framework.ros_source", level="WARNING"):
            callback(String(data="{}"))
        self.assertIs(runtime._build_context(101.0, 0.03).field, ADULT_FIELD_DIMENSIONS)

        with self.assertLogs("src.framework.ros_source", level="INFO") as logs:
            callback(String(data=payload(length=18.0, goal_depth=0.9)))
        self.assertIn("field info received from /soccer/field_info", logs.output[0])
        self.assertIn("length=18.0", logs.output[0])
        first = source.get_snapshot()
        self.assertEqual(first.field.length, 18.0)
        self.assertEqual(first.field.goal_depth, 0.9)
        self.assertIs(runtime._build_context(102.0, 0.03).field, first.field)
        self.assertIs(initial.field, ADULT_FIELD_DIMENSIONS)
        with self.assertLogs("src.framework.ros_source", level="WARNING"):
            callback(String(data=payload(length=-1)))
        self.assertIs(source.get_snapshot().field, first.field)

        callback(String(data=payload(length=24.0)))
        self.assertEqual(source.get_snapshot().field.length, 24.0)
        self.assertEqual(first.field.length, 18.0)
        self.assertEqual(runtime._build_context(100.0, 0.03).field.length, 24.0)


if __name__ == "__main__":
    unittest.main()
