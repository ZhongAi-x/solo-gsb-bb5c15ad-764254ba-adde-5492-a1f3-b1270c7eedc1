"""互斥试管隔离(精确搜索)回归测试: python -m unittest discover -s tests

核心断言:
- 互斥试管对所在槽位在按角度环形排序后绝不相邻(禁用槽仍占据相邻关系);
- 精确搜索结果与独立的全排列参考实现一致(最小评分 + 管号升序槽位序列裁决 +
  并列条数);
- 含互斥对时限 7 管 / 10 启用槽 / 12 对, 越界拒绝;
- 锁定管互斥冲突或无合法摆放时 ValueError(API 层映射为 400, 不保存方案);
- 无互斥对时仍走原有启发式求解;
- 互斥对随结果保留(保存/回放所需), 增量重排的移动罚权同样生效。
"""
import itertools
import math
import random
import unittest

from app.balancer import MutexPair, Solver, Slot, Tube, ring_adjacent_slots, run_plan


def payload(tubes, slots=12, disabled=(), pairs=(), **over):
    if isinstance(slots, int):
        slot_list = [
            {"index": i, "angle": 360.0 * i / slots, "enabled": i not in disabled}
            for i in range(slots)
        ]
        n = slots
    else:
        slot_list = slots
        n = len(slot_list)
    p = {
        "slot_count": n,
        "slots": slot_list,
        "radius_mm": 100.0,
        "rpm": 4000.0,
        "tubes": tubes,
        "mutex_pairs": [{"tube_a": a, "tube_b": b} for a, b in pairs],
    }
    p.update(over)
    return p


def _brute(slot_objs, tubes, pairs, move_weight=0.0, origin=None):
    """独立全排列参考: 返回 (legal 条目列表), 每条 (score, 管号升序槽位序列)。"""
    adj = ring_adjacent_slots(slot_objs)
    enabled = [s.index for s in slot_objs if s.enabled]
    ang = {s.index: s.angle for s in slot_objs}
    locked = {t["id"]: t["slot"] for t in tubes if t.get("locked")}
    free = [t["id"] for t in tubes if not t.get("locked")]
    mass = {t["id"]: float(t["mass"]) for t in tubes}
    pool = [s for s in enabled if s not in set(locked.values())]
    out = []
    for perm in itertools.permutations(pool, len(free)):
        loc = dict(locked)
        loc.update({free[k]: perm[k] for k in range(len(free))})
        if any(frozenset((loc[a], loc[b])) in adj for a, b in pairs):
            continue
        u1 = sum(
            mass[t] * complex(math.cos(math.radians(ang[loc[t]])),
                              math.sin(math.radians(ang[loc[t]])))
            for t in mass
        )
        u2 = sum(
            mass[t] * complex(math.cos(math.radians(2 * ang[loc[t]])),
                              math.sin(math.radians(2 * ang[loc[t]])))
            for t in mass
        )
        moved = 0
        if move_weight and origin:
            moved = sum(
                1 for t in free
                if origin.get(t) is not None and loc[t] != origin[t]
            )
        sc = abs(u1) ** 2 + 1e-6 * abs(u2) ** 2 + move_weight * moved
        out.append((sc, tuple(loc[t] for t in sorted(mass))))
    return out


class TestRingAdjacency(unittest.TestCase):
    def test_uniform_ring(self):
        slots = [Slot(i, i * 30.0, True) for i in range(12)]
        adj = ring_adjacent_slots(slots)
        self.assertIn(frozenset((0, 1)), adj)
        self.assertIn(frozenset((0, 11)), adj)      # 环绕
        self.assertNotIn(frozenset((0, 2)), adj)

    def test_disabled_slot_still_occupies_adjacency(self):
        slots = [Slot(0, 0.0, True), Slot(1, 30.0, False),
                 Slot(2, 60.0, True), Slot(3, 90.0, True)]
        adj = ring_adjacent_slots(slots)
        # 启用槽 0 与 2 之间隔着禁用槽 1, 不相邻
        self.assertNotIn(frozenset((0, 2)), adj)
        self.assertIn(frozenset((0, 1)), adj)
        self.assertIn(frozenset((1, 2)), adj)

    def test_irregular_angles_sorted_by_angle(self):
        slots = [Slot(0, 0.0, True), Slot(1, 95.0, True),
                 Slot(2, 180.0, True), Slot(3, 272.0, True)]
        adj = ring_adjacent_slots(slots)
        self.assertEqual(adj, {
            frozenset((0, 1)), frozenset((1, 2)),
            frozenset((2, 3)), frozenset((3, 0)),
        })


class TestMutexExact(unittest.TestCase):
    def _check_against_brute(self, p, pairs, move_weight=0.0, origin=None):
        r = run_plan(p)
        slot_objs = [
            Slot(int(s["index"]), float(s["angle"]), bool(s.get("enabled", True)))
            for s in p["slots"]
        ]
        legal = _brute(slot_objs, p["tubes"], pairs, move_weight, origin)
        self.assertTrue(legal)
        best = min(s for s, _ in legal)
        tol = 1e-9 * max(1.0, best)
        tied = [k for s, k in legal if s <= best + tol]
        loc = {x["tube_id"]: x["slot"] for x in r["placements"]}
        key = tuple(loc[t] for t in sorted(loc))
        self.assertEqual(key, min(tied))
        self.assertEqual(r["exact_search"]["legal_arrangements"], len(legal))
        self.assertEqual(r["exact_search"]["optimal_arrangement_count"], len(tied))
        # 物理约束: 每对互斥管不相邻
        adj = ring_adjacent_slots(slot_objs)
        for a, b in pairs:
            self.assertNotIn(frozenset((loc[a], loc[b])), adj)
        return r

    def test_basic_isolation_perfect_balance(self):
        tubes = [{"id": i, "mass": 50} for i in range(1, 5)]
        p = payload(tubes, slots=10, pairs=[(1, 3)])
        r = self._check_against_brute(p, [(1, 3)])
        self.assertAlmostEqual(r["metrics"]["u1_mag_g_mm"], 0.0, places=6)
        self.assertTrue(r["config"]["exact_mutex"])
        self.assertEqual(r["config"]["variant"], "mutex_exact")
        # 回放步骤与互斥对保留
        self.assertTrue(r["steps"])
        self.assertEqual(r["mutex_pairs"], [{"tube_a": 1, "tube_b": 3}])

    def test_random_cases_match_brute_force(self):
        rng = random.Random(20260924)
        for _ in range(25):
            n = rng.choice([5, 6, 8, 10])
            angles = sorted(rng.sample(range(0, 360, 10), n))
            disabled = set(rng.sample(range(n), rng.randint(0, 2)))
            slot_objs = [Slot(i, float(angles[i]), i not in disabled) for i in range(n)]
            n_enabled = n - len(disabled)
            nt = rng.randint(2, min(5, n_enabled - 1))
            ids = list(range(1, nt + 1))
            tubes = [{"id": i, "mass": rng.choice([10, 20, 30, 50, 51, 49])}
                     for i in ids]
            if rng.random() < 0.4:
                lid = rng.choice(ids)
                en = [s.index for s in slot_objs if s.enabled]
                # 锁定一个不与其他锁定管冲突的槽(本测试至多一把锁)
                tubes[lid - 1].update(
                    locked=True, slot=rng.choice(en)
                )
            pair_set = set()
            for _ in range(rng.randint(1, 2)):
                a, b = rng.sample(ids, 2)
                pair_set.add((min(a, b), max(a, b)))
            pairs = sorted(pair_set)
            p = payload(
                tubes,
                slots=[{"index": s.index, "angle": s.angle, "enabled": s.enabled}
                       for s in slot_objs],
                pairs=pairs,
            )
            legal = _brute(slot_objs, tubes, pairs)
            if not legal:
                with self.assertRaises(ValueError):
                    run_plan(p)
            else:
                self._check_against_brute(p, pairs)

    def test_disabled_slot_blocks_separation(self):
        # 禁用槽把 0 与 2 在环上隔开; 但 0 的物理邻居是禁用槽 1 与槽 3
        slot_list = [
            {"index": 0, "angle": 0.0, "enabled": True},
            {"index": 1, "angle": 30.0, "enabled": False},
            {"index": 2, "angle": 60.0, "enabled": True},
            {"index": 3, "angle": 90.0, "enabled": True},
        ]
        tubes = [{"id": 1, "mass": 50}, {"id": 2, "mass": 50}]
        p = payload(tubes, slots=slot_list, pairs=[(1, 2)])
        r = run_plan(p)
        loc = sorted(x["slot"] for x in r["placements"])
        # 合法位置只有 {0, 2} 与 {2, 3}(0/3 相邻); 评分下 0/2 为夹角 60°
        self.assertIn(loc, ([0, 2], [2, 3]))

    def test_locked_mutex_conflict_raises(self):
        tubes = [
            {"id": 1, "mass": 50, "locked": True, "slot": 0},
            {"id": 2, "mass": 50, "locked": True, "slot": 1},
        ]
        p = payload(tubes, slots=10, pairs=[(1, 2)])
        with self.assertRaises(ValueError) as ctx:
            run_plan(p)
        self.assertIn("不可行", str(ctx.exception))

    def test_locked_free_mutex_enforced(self):
        tubes = [
            {"id": 1, "mass": 50, "locked": True, "slot": 0},
            {"id": 2, "mass": 50},
        ]
        p = payload(tubes, slots=10, pairs=[(1, 2)])
        r = run_plan(p)
        s2 = next(x["slot"] for x in r["placements"] if x["tube_id"] == 2)
        s1 = next(x["slot"] for x in r["placements"] if x["tube_id"] == 1)
        # 槽 0 的物理邻居是 1 与 9
        self.assertEqual(s1, 0)
        self.assertNotIn(s2, (1, 9))

    def test_no_legal_arrangement_raises(self):
        # 3 支试管占满 3 个环槽, 任何两支都相邻 → 无合法摆放
        tubes = [{"id": i, "mass": 50} for i in range(1, 4)]
        p = payload(tubes, slots=3, pairs=[(1, 2)])
        with self.assertRaises(ValueError) as ctx:
            run_plan(p)
        self.assertIn("不可行", str(ctx.exception))

    def test_limit_tubes_slots_pairs(self):
        base_tubes = [{"id": i, "mass": 50} for i in range(1, 9)]  # 8 支
        with self.assertRaises(ValueError):
            run_plan(payload(base_tubes, slots=10, pairs=[(1, 2)]))
        # 11 个启用槽
        tubes7 = [{"id": i, "mass": 50} for i in range(1, 8)]
        slots = [{"index": i, "angle": 360.0 * i / 11, "enabled": True}
                 for i in range(11)]
        with self.assertRaises(ValueError):
            run_plan(payload(tubes7, slots=slots, pairs=[(1, 2)]))
        # 13 对互斥
        many_pairs = [(1, i) for i in range(2, 9)]  # 7 对
        many_pairs += [(2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (3, 4)]  # 共 13
        with self.assertRaises(ValueError):
            run_plan(payload(base_tubes[:7], slots=10, pairs=many_pairs))

    def test_unknown_and_duplicate_pairs_rejected(self):
        tubes = [{"id": i, "mass": 50} for i in range(1, 4)]
        with self.assertRaises(ValueError):
            run_plan(payload(tubes, slots=10, pairs=[(1, 9)]))
        with self.assertRaises(ValueError):
            run_plan(payload(tubes, slots=10, pairs=[(2, 2)]))
        with self.assertRaises(ValueError):
            run_plan(payload(tubes, slots=10, pairs=[(1, 2), (2, 1)]))

    def test_no_pairs_uses_existing_solver(self):
        tubes = [{"id": i, "mass": 50} for i in range(1, 5)]
        r = run_plan(payload(tubes, slots=12))
        self.assertFalse(r["config"]["exact_mutex"])
        self.assertNotEqual(r["config"]["variant"], "mutex_exact")
        self.assertEqual(r["mutex_pairs"], [])
        self.assertIsNone(r["exact_search"])

    def test_incremental_move_weight_respected(self):
        # 高移动罚权: 上一方案的合法摆放应被完整保留
        tubes = [{"id": 1, "mass": 50}, {"id": 2, "mass": 50}]
        previous = [{"tube_id": 1, "slot": 2}, {"tube_id": 2, "slot": 7}]
        p = payload(tubes, slots=10, pairs=[(1, 2)],
                    move_weight=1e9, previous=previous)
        r = run_plan(p)
        loc = {x["tube_id"]: x["slot"] for x in r["placements"]}
        self.assertEqual(loc, {1: 2, 2: 7})
        self.assertEqual(r["moved_tube_ids"], [])

    def test_tie_break_lexicographic_slot_key(self):
        # 8 槽均布、4 支等质量: (0,2,4,6) 是键最小的完美布局之一
        tubes = [{"id": i, "mass": 50} for i in range(1, 5)]
        p = payload(tubes, slots=8, pairs=[(1, 3)])
        r = run_plan(p)
        loc = {x["tube_id"]: x["slot"] for x in r["placements"]}
        self.assertEqual(loc, {1: 0, 2: 2, 3: 4, 4: 6})
        self.assertGreater(r["exact_search"]["optimal_arrangement_count"], 1)
        self.assertTrue(r["exact_search"]["tie_broken"])

    def test_deterministic(self):
        tubes = [{"id": i, "mass": 40 + i} for i in range(1, 6)]
        p = payload(tubes, slots=10, pairs=[(1, 2), (3, 4)])
        r1 = run_plan(payload(tubes, slots=10, pairs=[(1, 2), (3, 4)]))
        r2 = run_plan(p)
        k1 = sorted((x["tube_id"], x["slot"]) for x in r1["placements"])
        k2 = sorted((x["tube_id"], x["slot"]) for x in r2["placements"])
        self.assertEqual(k1, k2)

    def test_all_locked_with_mutex(self):
        tubes = [
            {"id": 1, "mass": 50, "locked": True, "slot": 0},
            {"id": 2, "mass": 50, "locked": True, "slot": 5},
        ]
        p = payload(tubes, slots=10, pairs=[(1, 2)])
        r = run_plan(p)
        self.assertEqual(r["exact_search"]["legal_arrangements"], 1)
        self.assertEqual(r["exact_search"]["total_arrangements"], 1)

    def test_solver_dataclass_api(self):
        slots = [Slot(i, i * 36.0, True) for i in range(10)]
        tubes = [Tube(1, 50.0), Tube(2, 50.0)]
        solver = Solver(slots=slots, tubes=tubes,
                        mutex_pairs=[MutexPair(1, 2)])
        result = solver.solve()
        loc = {p["tube_id"]: p["slot"] for p in result["placements"]}
        adj = ring_adjacent_slots(slots)
        self.assertNotIn(frozenset((loc[1], loc[2])), adj)


if __name__ == "__main__":
    unittest.main()
