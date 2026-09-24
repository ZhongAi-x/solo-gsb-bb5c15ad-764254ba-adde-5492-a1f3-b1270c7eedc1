"""离心机转子配平引擎。

模型假设:
- 单面(single-plane)转子, 所有试管位于同一半径 R 的圆周上;
- 槽位角度由用户给定(度), 0° 在正上方, 顺时针为正(前端按同一约定绘制);
- 每个可用槽位最多放置一支试管, 禁用槽位不能放置;
- 质量单位克(g), 半径单位毫米(mm)。

物理量:
- 一阶不平衡量 U1 = Σ m_k · R · e^{iθ_k}        (质径积, g·mm)  → 离心合力方向
- 二阶不平衡量 U2 = Σ m_k · R · e^{i·2θ_k}      (g·mm)          → 2 倍频残余
- 离心合力 F = ω² · |U1|/1000  (N, 质量换算为 kg)
- 合力矩(关于轴心) τ = F · R/1000  (N·m)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional

TWO_PI = 2.0 * math.pi

# 互斥试管隔离(精确搜索)的规模上限
MUTEX_MAX_PAIRS = 12
MUTEX_MAX_TUBES = 7
MUTEX_MAX_ENABLED_SLOTS = 10
# 并列评分的数值容差(与 app/evacuation.py 一致)
_SCORE_TOL = 1e-9


# ---------------------------------------------------------------------------
# 几何工具
# ---------------------------------------------------------------------------
def _d(angle_deg: float) -> float:
    """角度(度) → 弧度。"""
    return math.radians(angle_deg)


def signed_angle_diff(a: float, b: float) -> float:
    """a-b 归一化到 [-180, 180]。"""
    return (a - b + 180.0) % 360.0 - 180.0


def angle_between(a: float, b: float) -> float:
    """两个角度之间的最小夹角 [0, 180]。"""
    return abs(signed_angle_diff(a, b))


def opposite_deviation(a: float, b: float) -> float:
    """槽位 a 与 b 偏离严格对径(180°)的程度 [0,180]。"""
    return abs(abs(signed_angle_diff(a, b)) - 180.0)


def ring_adjacent_slots(slots: list[Slot]) -> set[frozenset[int]]:
    """按角度环形排序后的物理相邻槽位对(含首尾环绕)。

    禁用槽仍参与环形排序并占据物理相邻关系: 两个启用槽之间若隔着禁用槽,
    它们在环上不相邻。同角度槽位以槽号为次序键, 同角度槽彼此相邻。
    """
    order = sorted(slots, key=lambda s: (s.angle % 360.0, s.index))
    adj: set[frozenset[int]] = set()
    m = len(order)
    for i in range(m):
        j = (i + 1) % m
        if i != j:  # 单槽时不与自身相邻
            adj.add(frozenset((order[i].index, order[j].index)))
    return adj


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------
@dataclass
class Slot:
    index: int
    angle: float          # 度
    enabled: bool = True


@dataclass
class Tube:
    id: int
    mass: float
    locked: bool = False
    slot: Optional[int] = None   # 锁定试管的固定槽位; 非锁定为 None


@dataclass(frozen=True)
class MutexPair:
    """互斥试管对: 两支试管不得落在按角度环形排序后相邻的物理槽位。"""
    a: int
    b: int


@dataclass
class Solver:
    slots: list[Slot]
    tubes: list[Tube]
    mutex_pairs: list[MutexPair] = field(default_factory=list)
    radius_mm: float = 100.0
    rpm: float = 4000.0
    w1: float = 1.0             # 一阶不平衡量目标权重(主导)
    w2: float = 0.000001        # 二阶不平衡量目标权重(仅在一阶几乎持平时作裁决)
    move_weight: float = 0.0    # 增量重排: 每支被移动试管的罚分(g² 当量); 默认不限制移动
    restarts: int = 10
    chain_steps: int = 2500
    seed: int = 20260923

    # —— 运行期状态 ——
    _enc: dict[int, int] = field(default_factory=dict)   # slot 索引 → tube id
    _loc: dict[int, int] = field(default_factory=dict)   # tube id → slot 索引
    _origin: dict[int, Optional[int]] = field(default_factory=dict)
    _u1: complex = 0j
    _u2: complex = 0j
    _rng: Optional[random.Random] = None
    _steps: list[dict[str, Any]] = field(default_factory=list)
    _track: bool = False
    _score_cache: float = 0.0
    _track_best: float = math.inf
    _anneal_recorded: int = 0
    _anneal_record_cap: int = 24
    # 互斥精确搜索的统计(仅含互斥对时填充)
    _exact: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._chain_steps = self.chain_steps

    # ------------------------------------------------------------------
    # 校验与预处理
    # ------------------------------------------------------------------
    def validate(self) -> None:
        if not self.slots:
            raise ValueError("至少需要定义一个槽位。")
        n = len(self.slots)
        for s in self.slots:
            if not (0 <= s.index < n):
                raise ValueError(f"槽位索引 {s.index} 超出范围。")
        seen = set()
        for s in self.slots:
            if not s.enabled:
                continue
            key = round(s.angle % 360.0, 6)
            if key in seen:
                raise ValueError(f"槽位 {s.index} 与其他启用槽位角度重复({s.angle}°)。")
            seen.add(key)
        enabled = {s.index for s in self.slots if s.enabled}
        if not enabled:
            raise ValueError("至少需要启用一个槽位。")
        ids = set()
        for t in self.tubes:
            if t.id in ids:
                raise ValueError(f"试管编号 {t.id} 重复。")
            ids.add(t.id)
            if t.mass <= 0:
                raise ValueError(f"试管 {t.id} 质量必须为正数。")
        for t in self.tubes:
            if t.locked:
                if t.slot is None:
                    raise ValueError(f"锁定试管 {t.id} 必须指定槽位。")
                if t.slot not in range(n):
                    raise ValueError(f"锁定试管 {t.id} 的槽位 {t.slot} 不存在。")
                if t.slot not in enabled:
                    raise ValueError(f"锁定试管 {t.id} 不能放在禁用槽位 {t.slot}。")
        locked_slots = [t.slot for t in self.tubes if t.locked]
        if len(set(locked_slots)) != len(locked_slots):
            raise ValueError("同一槽位不能放置多支锁定试管。")
        if len([t for t in self.tubes]) > len(enabled):
            raise ValueError(
                f"试管数量({len(self.tubes)})超过可用槽位数量({len(enabled)})。"
            )
        if self.radius_mm <= 0 or self.rpm < 0:
            raise ValueError("半径必须为正, 转速不能为负。")
        # —— 互斥试管对 ——
        self._validate_mutex_pairs(enabled)

    def _validate_mutex_pairs(self, enabled: set[int]) -> None:
        if not self.mutex_pairs:
            return
        if len(self.mutex_pairs) > MUTEX_MAX_PAIRS:
            raise ValueError(
                f"互斥试管对至多 {MUTEX_MAX_PAIRS} 对(当前 {len(self.mutex_pairs)} 对)。"
            )
        if len(self.tubes) > MUTEX_MAX_TUBES:
            raise ValueError(
                f"含互斥对的请求至多 {MUTEX_MAX_TUBES} 支试管"
                f"(当前 {len(self.tubes)} 支), 精确搜索规模受限。"
            )
        if len(enabled) > MUTEX_MAX_ENABLED_SLOTS:
            raise ValueError(
                f"含互斥对的请求至多 {MUTEX_MAX_ENABLED_SLOTS} 个启用槽"
                f"(当前 {len(enabled)} 个), 精确搜索规模受限。"
            )
        ids = {t.id for t in self.tubes}
        seen: set[frozenset[int]] = set()
        for mp in self.mutex_pairs:
            if mp.a == mp.b:
                raise ValueError(f"互斥试管对的两支试管编号不能相同({mp.a})。")
            if mp.a not in ids or mp.b not in ids:
                raise ValueError(
                    f"互斥试管对 ({mp.a}, {mp.b}) 引用了不存在的试管编号。"
                )
            key = frozenset((mp.a, mp.b))
            if key in seen:
                raise ValueError(f"互斥试管对 ({min(mp.a, mp.b)}, {max(mp.a, mp.b)}) 重复。")
            seen.add(key)
        # 锁定管互斥冲突: 两支都锁定且固定槽在物理环上相邻 → 结构上不可行
        locked_slot = {t.id: t.slot for t in self.tubes if t.locked}
        adj = ring_adjacent_slots(self.slots)
        for mp in self.mutex_pairs:
            if mp.a in locked_slot and mp.b in locked_slot:
                if frozenset((locked_slot[mp.a], locked_slot[mp.b])) in adj:
                    raise ValueError(
                        f"不可行: 互斥试管 {mp.a} 与 {mp.b} 均已锁定, "
                        f"而锁定槽位 {locked_slot[mp.a]} 与 {locked_slot[mp.b]} "
                        "在按角度环形排序后相邻(禁用槽仍占据相邻关系)。"
                    )

    # ------------------------------------------------------------------
    # 复向量
    # ------------------------------------------------------------------
    def _c1(self, slot_idx: int, m: float) -> complex:
        a = _d(self.slots[slot_idx].angle)
        return complex(m * math.cos(a), m * math.sin(a))

    def _c2(self, slot_idx: int, m: float) -> complex:
        a = _d(self.slots[slot_idx].angle)
        return complex(m * math.cos(2 * a), m * math.sin(2 * a))

    def _penalty(self, loc: dict[int, int]) -> float:
        moved = sum(
            1 for tid, slot in loc.items()
            if not self._by_id[tid].locked and self._origin.get(tid) is not None
            and self._origin[tid] != slot
        )
        return self.move_weight * moved

    # ------------------------------------------------------------------
    # 状态初始化
    # ------------------------------------------------------------------
    def _reset(self) -> None:
        self._enc = {}
        self._loc = {}
        self._u1 = 0j
        self._u2 = 0j
        self._kept = []
        self._steps = []
        self._by_id = {t.id: t for t in self.tubes}

    def _place(self, tube_id: int, slot_idx: int) -> None:
        t = self._by_id[tube_id]
        self._enc[slot_idx] = tube_id
        self._loc[tube_id] = slot_idx
        self._u1 += self._c1(slot_idx, t.mass)
        self._u2 += self._c2(slot_idx, t.mass)

    def _remove(self, tube_id: int) -> None:
        slot_idx = self._loc.pop(tube_id)
        t = self._by_id[tube_id]
        del self._enc[slot_idx]
        self._u1 -= self._c1(slot_idx, t.mass)
        self._u2 -= self._c2(slot_idx, t.mass)

    def _score(self, u1: complex, u2: complex, loc: dict[int, int]) -> float:
        return (
            self.w1 * abs(u1) ** 2
            + self.w2 * abs(u2) ** 2
            + self._penalty(loc)
        )

    # ------------------------------------------------------------------
    # 锁定试管放置
    # ------------------------------------------------------------------
    def _place_locked(self, record: bool = True) -> None:
        for t in sorted((x for x in self.tubes if x.locked), key=lambda x: x.id):
            assert t.slot is not None
            self._place(t.id, t.slot)
        if record and self._track:
            n_lock = len([t for t in self.tubes if t.locked])
            if n_lock:
                self._snapshot("init", None, f"载入 {n_lock} 支锁定试管作为初始约束。")
            else:
                self._snapshot("init", None, "开始贪心构造初始对称布局。")

    # ------------------------------------------------------------------
    # 初始布局构造
    # variants:
    #   0                              —— 质量贪心配对, 槽位确定性打分
    #   1 .. GREEDY_VARIANTS           —— 随机化配对 + 槽位打分抖动
    #   > GREEDY_VARIANTS              —— 完全随机摆放(增强搜索多样性)
    #   "incremental"                  —— 优先沿用旧布局
    # ------------------------------------------------------------------
    GREEDY_VARIANTS = 4

    def _free_pairs(self, variant: int) -> list[list[int]]:
        free = [t for t in self.tubes if not t.locked]
        if variant != 0:
            # 随机变体: 洗牌改变配对构成与放置顺序
            self._rng.shuffle(free)
        else:
            # 质量贪心: 重试管先放, 相邻质量优先配对
            free = sorted(free, key=lambda t: (-t.mass, t.id))
        pairs: list[list[int]] = []
        for i in range(0, len(free) - 1, 2):
            pairs.append([free[i].id, free[i + 1].id])
        if len(free) % 2 == 1:
            pairs.append([free[-1].id])
        return pairs

    def _construct_random(self) -> None:
        free_slots = [s.index for s in self.slots if s.enabled]
        free_tubes = [t.id for t in self.tubes if not t.locked]
        self._rng.shuffle(free_slots)
        self._rng.shuffle(free_tubes)
        for tid, slot in zip(free_tubes, free_slots):
            self._place(tid, slot)
        if self._track:
            self._snapshot(
                "init", free_tubes,
                f"随机起点: 将 {len(free_tubes)} 支试管随机分布到可用槽位, 交由模拟退火搜索。",
            )

    def _place_group_greedy(self, group: list[int], variant: int) -> tuple[int, Optional[int]]:
        """把一个配对组(1~2 支试管)放到使目标增量最小的槽位。"""
        avail = [s.index for s in self.slots if s.enabled and s.index not in self._enc]
        jitter = 1e-9 if variant != 0 else 0.0

        if len(group) == 2:
            id0, id1 = group[0], group[1]
            t0, t1 = self._by_id[id0], self._by_id[id1]
            best: Optional[tuple[float, float, int, Optional[int]]] = None
            for x in range(len(avail)):
                for y in range(x + 1, len(avail)):
                    a, b = avail[x], avail[y]
                    dev = opposite_deviation(self.slots[a].angle, self.slots[b].angle)
                    # 两种朝向: t0→a/t1→b 与 t0→b/t1→a
                    for ia, ib, ta, tb in ((a, b, t0, t1), (b, a, t0, t1)):
                        nu1 = self._u1 + self._c1(ia, ta.mass) + self._c1(ib, tb.mass)
                        nu2 = self._u2 + self._c2(ia, ta.mass) + self._c2(ib, tb.mass)
                        trial_loc = dict(self._loc)
                        trial_loc[ta.id] = ia
                        trial_loc[tb.id] = ib
                        sc = self._score(nu1, nu2, trial_loc)
                        if jitter:
                            sc += self._rng.random() * jitter
                        # 同分时偏好接近对径的槽位对
                        cand = (sc, dev, a, b)
                        if best is None or cand < best:
                            best = cand
            assert best is not None
            _, _, a, b = best
            # 在 (a, b) 上比较两种朝向, 确定谁进 a
            def orient_score(sa, sb, ta, tb):
                nu1 = self._u1 + self._c1(sa, ta.mass) + self._c1(sb, tb.mass)
                nu2 = self._u2 + self._c2(sa, ta.mass) + self._c2(sb, tb.mass)
                trial_loc = dict(self._loc)
                trial_loc[ta.id] = sa
                trial_loc[tb.id] = sb
                return self._score(nu1, nu2, trial_loc)

            if orient_score(a, b, t0, t1) <= orient_score(a, b, t1, t0):
                self._place(t0.id, a)
                self._place(t1.id, b)
            else:
                self._place(t1.id, a)
                self._place(t0.id, b)
            return a, b

        # 单支
        t = self._by_id[group[0]]
        best = None
        for a in avail:
            nu1 = self._u1 + self._c1(a, t.mass)
            nu2 = self._u2 + self._c2(a, t.mass)
            trial_loc = dict(self._loc)
            trial_loc[t.id] = a
            sc = self._score(nu1, nu2, trial_loc)
            if jitter:
                sc += self._rng.random() * jitter
            if best is None or sc < best[0]:
                best = (sc, a)
        a = best[1]
        self._place(t.id, a)
        return a, None

    def _seed_incremental(self, previous: list[dict[str, Any]]) -> list[int]:
        """增量重排: 沿用上一方案中仍然有效的摆放, 返回仍需放置的试管 id。"""
        prev_by_tube = {}
        for item in previous:
            tid = int(item["tube_id"])
            prev_by_tube[tid] = int(item["slot"])
        cur_ids = {t.id for t in self.tubes}
        kept: list[int] = []
        blocked: list[int] = []
        occupied = set(self._loc.values())  # 锁定试管占用的槽
        # 按旧槽顺序依次保留, 冲突者进入 blocked
        ordered = sorted(
            (tid for tid in prev_by_tube if tid in cur_ids),
            key=lambda tid: (prev_by_tube[tid], tid),
        )
        for tid in ordered:
            t = self._by_id[tid]
            if t.locked:
                continue
            old_slot = prev_by_tube[tid]
            slot_ok = (
                0 <= old_slot < len(self.slots)
                and self.slots[old_slot].enabled
                and old_slot not in occupied
            )
            if slot_ok:
                self._place(tid, old_slot)
                occupied.add(old_slot)
                kept.append(tid)
            else:
                blocked.append(tid)
        new_tubes = [
            t.id for t in self.tubes if not t.locked and t.id not in prev_by_tube
        ]
        # blocked 中的试管旧槽失效/冲突, 稍后必须重新放置
        self._kept = kept
        if self._track:
            if kept:
                self._snapshot(
                    "init", kept,
                    f"增量重排: {len(kept)} 支试管沿用上一方案槽位"
                    f"({', '.join(map(str, kept))}); "
                    f"{len(blocked)} 支旧槽失效/冲突需重放, "
                    f"{len(new_tubes)} 支为新增试管。",
                )
            else:
                self._snapshot(
                    "init", None,
                    f"增量重排: 没有可沿用的旧摆放, "
                    f"{len(new_tubes) + len(blocked)} 支试管待放置。",
                )
        # 剩余待放试管: 被阻塞的 + 新增的(已保留的不再返回)
        return blocked + new_tubes

    def _construct(self, variant, previous=None):
        self._reset()
        self._rng = random.Random(self.seed + (0 if variant == "incremental" else variant))
        incremental = variant == "incremental"
        self._place_locked(record=not incremental)
        if isinstance(variant, int) and variant > self.GREEDY_VARIANTS:
            # 完全随机起点
            self._construct_random()
            return self._score(self._u1, self._u2, self._loc)
        if incremental:
            groups = [[tid] for tid in self._seed_incremental(previous)]
        else:
            groups = self._free_pairs(variant)
        for g in groups:
            avail = [s for s in self.slots if s.enabled and s.index not in self._enc]
            if not avail:
                break
            if len(g) == 2 and len(avail) < 2:
                g = g[:1]
            sa, sb = self._place_group_greedy(g, variant if variant != "incremental" else 0)
            if self._track:
                self._record_build(g, sa, sb)
        return self._score(self._u1, self._u2, self._loc)

    def _record_build(self, group, sa, sb) -> None:
        if len(group) == 2 and sb is not None:
            t1, t2 = self._by_id[group[0]], self._by_id[group[1]]
            dev = opposite_deviation(self.slots[sa].angle, self.slots[sb].angle)
            if dev <= 0.5:
                geo = f"槽位 {sa} 与 {sb} 互为对径({self.slots[sa].angle:g}°/{self.slots[sb].angle:g}°)"
            else:
                geo = (f"槽位 {sa} 与 {sb} 夹角 {angle_between(self.slots[sa].angle, self.slots[sb].angle):.1f}°"
                       f"(偏离对径 {dev:.1f}°)")
            dm = abs(t1.mass - t2.mass)
            if dm < 1e-9:
                mass_txt = "质量完全相等"
            else:
                mass_txt = f"质量差 {dm:g} g, 为剩余试管中最接近的一对"
            reason = f"试管 {t1.id}({t1.mass:g}g) 与 {t2.id}({t2.mass:g}g) 配对: {mass_txt}; {geo}, 一阶力近似对消。"
            kind = "pair"
            ref = sorted(group)
        else:
            t = self._by_id[group[0]]
            reason = (f"试管 {t.id}({t.mass:g}g) 放入槽位 {sa}({self.slots[sa].angle:g}°), "
                      f"作为当前残余向量的补偿方向。")
            kind = "single"
            ref = [t.id]
        self._snapshot(kind, ref, reason)

    # ------------------------------------------------------------------
    # 局部搜索: 交换(两支试管) / 迁移(移到空槽)
    # ------------------------------------------------------------------
    def _move_delta(self, tid: int, frm: int, to: int) -> float:
        """试管 tid 从 frm 移到 to 后的目标变化(不含另一支)。"""
        t = self._by_id[tid]
        nu1 = self._u1 - self._c1(frm, t.mass) + self._c1(to, t.mass)
        nu2 = self._u2 - self._c2(frm, t.mass) + self._c2(to, t.mass)
        trial = dict(self._loc)
        trial[tid] = to
        return self._score(nu1, nu2, trial) - self._score_cache, nu1, nu2

    def _swap_delta(self, tid_a: int, a: int, tid_b: int, b: int) -> float:
        ta, tb = self._by_id[tid_a], self._by_id[tid_b]
        nu1 = (
            self._u1
            - self._c1(a, ta.mass) - self._c1(b, tb.mass)
            + self._c1(b, ta.mass) + self._c1(a, tb.mass)
        )
        nu2 = (
            self._u2
            - self._c2(a, ta.mass) - self._c2(b, tb.mass)
            + self._c2(b, ta.mass) + self._c2(a, tb.mass)
        )
        trial = dict(self._loc)
        trial[tid_a] = b
        trial[tid_b] = a
        return self._score(nu1, nu2, trial) - self._score_cache, nu1, nu2

    # ------------------------------------------------------------------
    # 局部搜索: 模拟退火(逃出局部最优) + 爬山(收敛)
    # ------------------------------------------------------------------
    def _state(self):
        return dict(self._loc), complex(self._u1), complex(self._u2)

    def _restore(self, state) -> None:
        loc, u1, u2 = state
        self._loc = dict(loc)
        self._enc = {v: k for k, v in self._loc.items()}
        self._u1, self._u2 = u1, u2

    def _apply_move(self, move) -> tuple[float, str]:
        """执行一个邻域动作, 返回 (目标变化, 描述)。"""
        if move[0] == "move":
            _, tid, a, b, d, nu1, nu2 = move
            self._remove(tid)
            self._place(tid, b)
            self._u1, self._u2 = nu1, nu2
            return d, (f"微调: 试管 {tid} 由槽位 {a} 迁到空槽 {b}")
        _, ti, a, tj, b, d, nu1, nu2 = move
        self._remove(ti)
        self._remove(tj)
        self._place(ti, b)
        self._place(tj, a)
        self._u1, self._u2 = nu1, nu2
        return d, (f"微调: 交换试管 {ti}(槽 {a}) 与试管 {tj}(槽 {b})")

    def _all_moves(self) -> list[tuple]:
        free_ids = [t.id for t in self.tubes if not t.locked]
        enabled_slots = [s.index for s in self.slots if s.enabled]
        empty = [s for s in enabled_slots if s not in self._enc]
        moves: list[tuple] = []
        self._score_cache = self._score(self._u1, self._u2, self._loc)
        for tid in free_ids:
            a = self._loc[tid]
            for b in empty:
                if a == b:
                    continue
                d, nu1, nu2 = self._move_delta(tid, a, b)
                moves.append(("move", tid, a, b, d, nu1, nu2))
        for i in range(len(free_ids)):
            for j in range(i + 1, len(free_ids)):
                ti, tj = free_ids[i], free_ids[j]
                a, b = self._loc[ti], self._loc[tj]
                d, nu1, nu2 = self._swap_delta(ti, a, tj, b)
                moves.append(("swap", ti, a, tj, b, d, nu1, nu2))
        return moves

    def _random_move(self):
        free_ids = [t.id for t in self.tubes if not t.locked]
        enabled_slots = [s.index for s in self.slots if s.enabled]
        empty = [s for s in enabled_slots if s not in self._enc]
        candidates = (1 if empty else 0) + 1  # move / swap
        kind = "move" if (empty and self._rng.randrange(candidates) == 0) else "swap"
        self._score_cache = self._score(self._u1, self._u2, self._loc)
        if kind == "move":
            tid = self._rng.choice(free_ids)
            a = self._loc[tid]
            b = self._rng.choice(empty)
            while b == a and len(empty) > 1:
                b = self._rng.choice(empty)
            if a == b:
                return None
            d, nu1, nu2 = self._move_delta(tid, a, b)
            return ("move", tid, a, b, d, nu1, nu2)
        if len(free_ids) < 2:
            return None
        ti, tj = self._rng.sample(free_ids, 2)
        a, b = self._loc[ti], self._loc[tj]
        d, nu1, nu2 = self._swap_delta(ti, a, tj, b)
        return ("swap", ti, a, tj, b, d, nu1, nu2)

    def _anneal_chain(self, steps: int, t0_frac: float = 0.05) -> None:
        """一次退火短链: 从当前布局出发, 指数降温。"""
        if len([t for t in self.tubes if not t.locked]) < 2:
            return
        cur_score = self._score(self._u1, self._u2, self._loc)
        t0 = max(cur_score * t0_frac, 1e-6)
        t1 = 1e-7
        cooling = (t1 / t0) ** (1.0 / max(steps, 1))
        temp = t0
        for _ in range(steps):
            move = self._random_move()
            if move is None:
                break
            d = move[4]
            if d < 0 or self._rng.random() < math.exp(-d / max(temp, 1e-18)):
                self._apply_move(move)
                cur_score += d
                # 仅记录追踪运行中"刷新全局最优"的改进, 保持回放精简
                if self._track and d < -1e-9 and cur_score < self._track_best - 1e-9 \
                        and self._anneal_recorded < self._anneal_record_cap:
                    self._track_best = cur_score
                    self._anneal_recorded += 1
                    self._record_anneal_step(move, d)
            temp *= cooling

    def _record_anneal_step(self, move, d: float) -> None:
        if move[0] == "move":
            _, tid, a, b, _, _, _ = move
            desc = f"退火搜索: 试管 {tid} 由槽位 {a} 迁到空槽 {b}"
            refs = [tid]
        else:
            _, ti, a, tj, b, _, _, _ = move
            desc = f"退火搜索: 交换试管 {ti}(槽 {a}) 与试管 {tj}(槽 {b})"
            refs = [ti, tj]
        self._snapshot(
            "refine", refs,
            f"{desc}, 目标函数下降 {abs(d):.4g}, 残余不平衡量减小。",
        )

    def _hill_climb(self, record_refines: bool = False) -> int:
        improvements = 0
        max_passes = 400
        for _ in range(max_passes):
            moves = self._all_moves()
            best = min(moves, key=lambda x: x[4], default=None)
            if best is None or best[4] >= -1e-9:
                break
            d, desc = self._apply_move(best)
            improvements += 1
            if record_refines:
                self._snapshot(
                    "refine",
                    list(best[1:5:3]) if best[0] == "swap" else [best[1]],
                    f"{desc}, 目标函数下降 {abs(d):.4g}, 残余不平衡量进一步减小。",
                )
        return improvements

    def _local_search(self, record: bool = False) -> None:
        """单个起点的局部搜索: 一次退火短链 + 爬山。"""
        free = [t for t in self.tubes if not t.locked]
        if len(free) < 2:
            self._hill_climb(record_refines=record)
            return
        self._anneal_chain(self._chain_steps)
        self._hill_climb(record_refines=record)

    # ------------------------------------------------------------------
    # 步骤快照
    # ------------------------------------------------------------------
    def _snapshot(self, kind: str, refs: Optional[list[int]], reason: str) -> None:
        u1g_mm = self.radius_mm * self._u1
        u2g_mm = self.radius_mm * self._u2
        self._steps.append({
            "seq": len(self._steps),
            "kind": kind,
            "refs": refs or [],
            "reason": reason,
            "placements": self._placement_snapshot(),
            "vectors": {
                "u1_mag": abs(u1g_mm),
                "u1_angle": (math.degrees(math.atan2(u1g_mm.imag, u1g_mm.real)) % 360.0),
                "u2_mag": abs(u2g_mm),
                "u2_angle": (math.degrees(math.atan2(u2g_mm.imag, u2g_mm.real)) % 360.0),
            },
        })

    def _placement_snapshot(self) -> list[dict[str, Any]]:
        out = []
        for t in self.tubes:
            slot = self._loc.get(t.id)
            out.append({
                "tube_id": t.id,
                "mass": t.mass,
                "locked": t.locked,
                "slot": slot,
            })
        return out

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def solve(self, previous: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
        self.validate()
        self._by_id = {t.id: t for t in self.tubes}
        # origin: 增量重排时各试管的原槽位
        self._origin = {}
        if previous:
            prev_map = {int(x["tube_id"]): int(x["slot"]) for x in previous}
            for t in self.tubes:
                self._origin[t.id] = prev_map.get(t.id) if not t.locked else None

        if self.mutex_pairs:
            return self._solve_mutex_exact(previous)

        free_cnt = len([t for t in self.tubes if not t.locked])
        greedy_variants = list(range(self.GREEDY_VARIANTS + 1))
        random_variants = list(range(self.GREEDY_VARIANTS + 1, self.restarts + 1))
        if free_cnt == 0:
            variants = [0]
        elif previous is not None:
            # 增量重排: 增量种子 + 少量贪心/随机起点对照
            variants = ["incremental"] + greedy_variants[1:] + random_variants
        else:
            variants = greedy_variants + random_variants

        best_score = math.inf
        best_state = None
        best_variant = None
        for v in variants:
            self._track = False
            self._construct(v, previous=previous if v == "incremental" else None)
            self._local_search()
            sc = self._score(self._u1, self._u2, self._loc)
            if sc < best_score - 1e-12:
                best_score, best_variant = sc, v
                best_state = self._state()

        # 复跑最佳变体(确定性 RNG)并记录可解释步骤
        self._track = True
        self._anneal_recorded = 0
        self._track_best = math.inf
        self._construct(best_variant, previous=previous if best_variant == "incremental" else None)
        self._track_best = self._score(self._u1, self._u2, self._loc)
        self._local_search(record=True)

        # 安全校验: 复跑结果应与最佳一致(确定性), 否则直接恢复最佳布局
        re_score = self._score(self._u1, self._u2, self._loc)
        if re_score > best_score + 1e-6:  # pragma: no cover - 理论上不发生
            self._restore(best_state)
            self._steps = []
            self._snapshot("init", None, "输出搜索得到的最优布局。")

        incremental = previous is not None
        return self._build_result(best_variant if best_variant is not None else 0,
                                  incremental, previous)

    # ------------------------------------------------------------------
    # 互斥试管隔离: 全排列精确搜索(管 ≤ 7、启用槽 ≤ 10)
    # ------------------------------------------------------------------
    def _solve_mutex_exact(self,
                           previous: Optional[list[dict[str, Any]]]) -> dict[str, Any]:
        """对全部合法摆放做精确枚举, 取现有评分最小值。

        合法摆放: 每支非锁定试管占据互不相同的启用槽, 且每对互斥试管所在槽位
        在按角度环形排序的物理邻接图上不相邻(禁用槽仍占据相邻关系)。
        评分沿用 w1·|U1|² + w2·|U2|² + move_weight·移动管数;
        容差内平分时按管号升序的槽位序列(字典序)裁决。
        """
        incremental = previous is not None
        self._reset()
        self._exact = {}

        adj = ring_adjacent_slots(self.slots)
        enabled_idx = sorted(s.index for s in self.slots if s.enabled)
        pos_of = {s: k for k, s in enumerate(enabled_idx)}

        free_tubes = sorted(
            (t for t in self.tubes if not t.locked), key=lambda t: t.id
        )
        free_ids = [t.id for t in free_tubes]
        locked_ids = sorted(t.id for t in self.tubes if t.locked)

        # 锁定试管先落位(其合法/不冲突已在 validate 中保证)
        self._place_locked(record=False)
        locked_occ = set(self._loc.values())

        n_free = len(free_ids)
        n_slots = len(enabled_idx)
        n_free_slots = n_slots - len(locked_occ)
        # 全部槽位分配(不考虑互斥): P(空闲启用槽, 非锁定管数)
        total = (
            math.comb(n_free_slots, n_free) * math.factorial(n_free)
            if n_free else 1
        )
        adj_pos: list[list[int]] = [[] for _ in range(n_slots)]
        for pair in adj:
            x, y = tuple(pair)
            if x in pos_of and y in pos_of:
                px, py = pos_of[x], pos_of[y]
                adj_pos[px].append(py)
                adj_pos[py].append(px)
        idx_of_tube = {tid: k for k, tid in enumerate(free_ids)}
        mutex_of: dict[int, list[int]] = {}
        for mp in self.mutex_pairs:
            mutex_of.setdefault(mp.a, []).append(mp.b)
            mutex_of.setdefault(mp.b, []).append(mp.a)

        # 锁定槽占用掩码 + 与锁定互斥搭档相邻的静态禁用位置掩码
        locked_pos_mask = 0
        locked_static_adj = [0] * n_free
        locked_slot_by_id = {t.id: t.slot for t in self.tubes if t.locked}
        for s in locked_occ:
            locked_pos_mask |= 1 << pos_of[s]
        for k, tid in enumerate(free_ids):
            for other in mutex_of.get(tid, []):
                if other in locked_slot_by_id:
                    lp = pos_of[locked_slot_by_id[other]]
                    locked_static_adj[k] |= 1 << lp
                    for q in adj_pos[lp]:
                        locked_static_adj[k] |= 1 << q

        # 预计算各管在各启用槽的一/二阶向量
        v1 = [[0j] * n_slots for _ in range(n_free)]
        v2 = [[0j] * n_slots for _ in range(n_free)]
        for k, tid in enumerate(free_ids):
            m = self._by_id[tid].mass
            for p, sidx in enumerate(enabled_idx):
                v1[k][p] = self._c1(sidx, m)
                v2[k][p] = self._c2(sidx, m)
        base_u1 = complex(self._u1)
        base_u2 = complex(self._u2)
        mw = self.move_weight
        # 增量重排: 各管放到各启用槽的移动罚增量
        move_add = [[0] * n_slots for _ in range(n_free)]
        if mw:
            for k, tid in enumerate(free_ids):
                origin = self._origin.get(tid)
                if origin is None:
                    continue
                for p, sidx in enumerate(enabled_idx):
                    if sidx != origin:
                        move_add[k][p] = mw
        w1, w2 = self.w1, self.w2

        # 允许位掩码(启用且未被锁定占用); 每个位置的物理相邻位掩码
        allowed_mask = ((1 << n_slots) - 1) ^ locked_pos_mask
        adj_mask = [0] * n_slots
        for px in range(n_slots):
            for py in adj_pos[px]:
                adj_mask[px] |= 1 << py
        # free→free 互斥搭档索引列表
        mutex_peer = [[] for _ in range(n_free)]
        for k, tid in enumerate(free_ids):
            for other in mutex_of.get(tid, []):
                oj = idx_of_tube.get(other)
                if oj is not None:
                    mutex_peer[k].append(oj)

        leaves = 0
        best_score = math.inf
        best_key: Optional[tuple[int, ...]] = None
        tie_count = 0
        assign = [-1] * n_free
        v1k, v2k, mak = v1, v2, move_add  # 局部别名, 减少闭包查找

        def dfs(k: int, occ: int, u1: complex, u2: complex,
                move_pen: float, final_best: float) -> None:
            """枚举第 k.. 支试管的放法。final_best 为 inf 时第一遍求最小分;
            否则只统计容差带内的并列并裁决槽位序列(管号升序字典序)。"""
            nonlocal leaves, best_score, best_key, tie_count
            if k == n_free:
                sc = (
                    w1 * (u1.real * u1.real + u1.imag * u1.imag)
                    + w2 * (u2.real * u2.real + u2.imag * u2.imag)
                    + move_pen
                )
                if final_best == math.inf:
                    leaves += 1
                    if sc < best_score:
                        best_score = sc
                elif sc <= final_best:
                    tie_count += 1
                    key = tuple(enabled_idx[p] for p in assign)
                    if best_key is None or key < best_key:
                        best_key = key
                return
            # 禁用位: 锁定互斥搭档相邻位 + 已放置(free)互斥搭档相邻位
            forb = locked_static_adj[k]
            for oj in mutex_peer[k]:
                if oj < k:
                    forb |= adj_mask[assign[oj]]
            cands = allowed_mask & ~occ & ~forb
            while cands:
                bit = cands & -cands
                cands ^= bit
                p = bit.bit_length() - 1
                assign[k] = p
                dfs(k + 1, occ | bit, u1 + v1k[k][p], u2 + v2k[k][p],
                    move_pen + mak[k][p], final_best)
                assign[k] = -1

        if n_free:
            dfs(0, 0, base_u1, base_u2, 0.0, math.inf)
            if leaves:
                tie_band = best_score + _SCORE_TOL * max(1.0, best_score)
                dfs(0, 0, base_u1, base_u2, 0.0, tie_band)
        else:
            leaves = 1
            best_score = self._score(self._u1, self._u2, self._loc)
            best_key = ()
            tie_count = 1

        self._exact = {
            "legal_arrangements": leaves,
            "optimal_arrangement_count": tie_count,
            "total_arrangements": total,
            "mutex_pair_count": len(self.mutex_pairs),
            "enabled_slot_count": n_slots,
            "tie_broken": tie_count > 1,
        }

        if leaves == 0:
            pair_txt = "、".join(
                f"({mp.a},{mp.b})" for mp in self.mutex_pairs
            )
            raise ValueError(
                f"不可行: 在互斥试管对 {pair_txt} 的物理相邻隔离约束下"
                f"(禁用槽仍占据相邻关系), {n_free} 支非锁定试管无法在 "
                f"{n_slots} 个启用槽中合法摆放; 共枚举 {total} 种摆放, "
                "无一满足全部互斥约束。"
            )

        # 用裁决得到的最优槽位序列重建状态并记录回放
        assert best_key is not None
        for tid in free_ids:
            self._loc.pop(tid, None)
        self._enc = {v: k for k, v in self._loc.items()}
        self._u1, self._u2 = base_u1, base_u2
        self._track = True
        self._steps = []
        if n_free:
            placed = ", ".join(
                f"试管 {tid}→槽 {slot}" for tid, slot in zip(free_ids, best_key)
            )
            self._snapshot(
                "exact", None,
                f"互斥隔离精确搜索: 锁定试管 {len(locked_ids)} 支先行落位"
                + (f"({', '.join(map(str, locked_ids))}), " if locked_ids else ", ")
                + f"对 {len(self.mutex_pairs)} 对互斥试管强制物理非相邻"
                "(相邻关系按角度环形排序, 禁用槽仍占位); "
                + f"在 {total} 种槽位分配中, {leaves} 种合法, "
                f"已按现有评分精确取最小值。",
            )
            for tid, slot in zip(free_ids, best_key):
                self._place(tid, slot)
            self._snapshot(
                "exact", free_ids,
                f"精确最优摆放: {placed}; 评分容差内平分时已按管号升序的"
                "槽位序列(字典序)裁决。",
            )
        else:
            self._snapshot(
                "exact", None,
                f"全部 {len(locked_ids)} 支试管均为锁定试管, 无需搜索; "
                f"已验证 {len(self.mutex_pairs)} 对互斥试管均不物理相邻。",
            )

        return self._build_result("mutex_exact", incremental, previous)

    # ------------------------------------------------------------------
    # 结果组装
    # ------------------------------------------------------------------
    def _build_result(self, variant, incremental: bool,
                      previous: Optional[list[dict[str, Any]]]) -> dict[str, Any]:
        R = self.radius_mm
        u1v, u2v = R * self._u1, R * self._u2
        m_total = sum(t.mass for t in self.tubes)
        metrics = self._metrics(u1v, u2v, m_total)

        placements = []
        moved_ids: list[int] = []
        prev_map = {}
        if previous:
            prev_map = {int(x["tube_id"]): int(x["slot"]) for x in previous}
        for t in sorted(self.tubes, key=lambda x: (self._loc[x.id], x.id)):
            slot = self._loc[t.id]
            moved = False
            if incremental and not t.locked and t.id in prev_map:
                moved = prev_map[t.id] != slot
                if moved:
                    moved_ids.append(t.id)
            placements.append({
                "tube_id": t.id,
                "mass": t.mass,
                "locked": t.locked,
                "slot": slot,
                "angle": self.slots[slot].angle if slot is not None else None,
                "previous_slot": prev_map.get(t.id) if incremental else None,
                "moved": moved,
                "u1": self._tube_vector(t, slot, 1),
                "u2": self._tube_vector(t, slot, 2),
            })

        pairs, singletons = self._extract_pairs()
        slot_state = []
        for s in self.slots:
            tid = self._enc.get(s.index)
            slot_state.append({
                "slot": s.index,
                "angle": s.angle,
                "enabled": s.enabled,
                "tube_id": tid,
                "locked": (self._by_id[tid].locked if tid is not None else False),
            })

        mutex_out = [
            {"tube_a": mp.a, "tube_b": mp.b}
            for mp in sorted(self.mutex_pairs, key=lambda x: (x.a, x.b))
        ]

        return {
            "config": {
                "slot_count": len(self.slots),
                "radius_mm": R,
                "rpm": self.rpm,
                "w1": self.w1,
                "w2": self.w2,
                "move_weight": self.move_weight,
                "variant": variant,
                "incremental": incremental,
                "exact_mutex": bool(self.mutex_pairs),
            },
            "metrics": metrics,
            "placements": placements,
            "slot_state": slot_state,
            "pairs": pairs,
            "singletons": singletons,
            "mutex_pairs": mutex_out,
            "exact_search": (dict(self._exact) if self._exact else None),
            "summary": self._summary_text(metrics, pairs, singletons, incremental, moved_ids),
            "moved_tube_ids": moved_ids,
            "steps": self._steps,
            "total_tubes": len(self.tubes),
        }

    def _tube_vector(self, t: Tube, slot: Optional[int], order: int) -> Optional[dict[str, Any]]:
        if slot is None:
            return None
        ang = _d(self.slots[slot].angle) * order
        v = t.mass * self.radius_mm * complex(math.cos(ang), math.sin(ang))
        return {"x": v.real, "y": v.imag, "mag": abs(v)}

    def _metrics(self, u1v: complex, u2v: complex, m_total: float) -> dict[str, Any]:
        omega = TWO_PI * self.rpm / 60.0
        u1_mag, u2_mag = abs(u1v), abs(u2v)
        # u1v 单位 g·mm → kg·m 需 ×1e-6
        u1_si = u1v * 1e-6
        force = omega ** 2 * abs(u1_si)          # N
        torque = force * self.radius_mm / 1000.0  # N·m
        ecc = (u1_mag / m_total) if m_total else 0.0  # mm 等效偏心距
        return {
            "u1_mag_g_mm": u1_mag,
            "u1_angle_deg": math.degrees(math.atan2(u1v.imag, u1v.real)) % 360.0,
            "u1_x_g_mm": u1v.real,
            "u1_y_g_mm": u1v.imag,
            "u2_mag_g_mm": u2_mag,
            "u2_angle_deg": math.degrees(math.atan2(u2v.imag, u2v.real)) % 360.0,
            "u2_x_g_mm": u2v.real,
            "u2_y_g_mm": u2v.imag,
            "centrifugal_force_n": force,
            "resultant_moment_n_m": torque,
            "equivalent_eccentricity_um": ecc * 1000.0,
            "total_mass_g": m_total,
            "grade": self._grade(ecc),
            "rpm": self.rpm,
            "angular_velocity_rad_s": omega,
        }

    @staticmethod
    def _grade(ecc_mm: float) -> str:
        e_um = ecc_mm * 1000.0
        if e_um <= 1.0:
            return "A"
        if e_um <= 5.0:
            return "B"
        if e_um <= 20.0:
            return "C"
        return "D"

    # ------------------------------------------------------------------
    # 对径配对分析
    # ------------------------------------------------------------------
    def _extract_pairs(self):
        n = len(self.slots)
        occupied = {s for s in self._enc if self.slots[s].enabled}
        used: set[int] = set()
        pairs: list[dict[str, Any]] = []
        singletons: list[dict[str, Any]] = []

        # 为每个占用槽找“最佳对径搭档”(偏差最小), 贪心匹配
        while True:
            best = None  # (dev, a, b)
            for a in occupied - used:
                for b in occupied - used:
                    if b <= a:
                        continue
                    dev = opposite_deviation(self.slots[a].angle, self.slots[b].angle)
                    if best is None or dev < best[0]:
                        best = (dev, a, b)
            if best is None:
                break
            dev, a, b = best
            if dev > 30.0:
                break
            used.add(a)
            used.add(b)
            pairs.append(self._pair_info(a, b, dev))

        for a in sorted(occupied - used):
            singletons.append(self._singleton_info(a))
        pairs.sort(key=lambda p: (p["slot_a"], p["slot_b"]))
        singletons.sort(key=lambda x: x["slot"])
        return pairs, singletons

    def _pair_info(self, a: int, b: int, dev: float) -> dict[str, Any]:
        ta = self._by_id[self._enc[a]]
        tb = self._by_id[self._enc[b]]
        v1 = self._c1(a, ta.mass) + self._c1(b, tb.mass)
        v2 = self._c2(a, ta.mass) + self._c2(b, tb.mass)
        ideal_single = abs(self._c1(a, ta.mass)) + abs(self._c1(b, tb.mass))
        cancel_u1 = (1.0 - abs(v1) / ideal_single) * 100.0 if ideal_single else 100.0
        locked_pair = ta.locked and tb.locked
        if dev <= 0.5:
            geo_txt = f"槽位 {a} 与 {b} 严格对径({self.slots[a].angle:g}°/{self.slots[b].angle:g}°)"
        else:
            geo_txt = (f"槽位 {a} 与 {b} 夹角 {angle_between(self.slots[a].angle, self.slots[b].angle):.1f}°"
                       f", 偏离对径 {dev:.1f}°")
        dm = abs(ta.mass - tb.mass)
        if dm < 1e-9:
            mass_txt = "两支质量相等, 一阶离心力完全反向"
        else:
            mass_txt = f"两支质量相差 {dm:g} g, 残余由较重一侧主导"
        if dev <= 0.5 and dm < 1e-9:
            verdict = "理想对置: 一阶力严格对消"
            if abs(v2) < 1e-9:
                verdict += ", 二阶分量也为零"
        elif cancel_u1 >= 90:
            verdict = f"高质量配对: 一阶力对消约 {cancel_u1:.1f}%"
        else:
            verdict = f"几何受限配对: 一阶力对消约 {cancel_u1:.1f}%, 由整体布局补偿"
        lock_txt = "; 两支均为锁定试管, 布局在其约束下完成" if locked_pair else (
            "; 含锁定试管" if (ta.locked or tb.locked) else "")
        reason = (f"试管 {ta.id}({ta.mass:g}g)↔试管 {tb.id}({tb.mass:g}g): {geo_txt}; "
                  f"{mass_txt}。{verdict}{lock_txt}。")
        return {
            "slot_a": a, "slot_b": b,
            "tube_a": ta.id, "tube_b": tb.id,
            "mass_a": ta.mass, "mass_b": tb.mass,
            "locked_a": ta.locked, "locked_b": tb.locked,
            "opposite_deviation_deg": dev,
            "mass_diff_g": dm,
            "u1_residual_g_mm": self.radius_mm * abs(v1),
            "u2_residual_g_mm": self.radius_mm * abs(v2),
            "u1_cancel_pct": max(0.0, cancel_u1),
            "reason": reason,
        }

    def _singleton_info(self, a: int) -> dict[str, Any]:
        t = self._by_id[self._enc[a]]
        # 启用槽中最接近对径的槽
        best = None
        for s in self.slots:
            if not s.enabled or s.index == a:
                continue
            dev = opposite_deviation(self.slots[a].angle, s.angle)
            if best is None or dev < best[0]:
                best = (dev, s.index)
        occupied_opp = best is not None and best[1] in self._enc
        if best is None:
            opp_txt = "转子上没有可用于对置的槽位"
        elif occupied_opp:
            opp_txt = f"对径方向槽位 {best[1]} 已被占用"
        else:
            opp_txt = f"对径方向槽位 {best[1]} 空置(偏差 {best[0]:.1f}°), 由整体合力补偿"
        lock_txt = "该试管为锁定试管, 不能移动; " if t.locked else ""
        reason = (f"试管 {t.id}({t.mass:g}g) 位于槽位 {a}({self.slots[a].angle:g}°)无对径搭档: "
                  f"{lock_txt}{opp_txt}。")
        return {
            "slot": a,
            "tube_id": t.id,
            "mass": t.mass,
            "locked": t.locked,
            "nearest_opposite_slot": best[1] if best else None,
            "nearest_opposite_deviation_deg": best[0] if best else None,
            "reason": reason,
        }

    def _summary_text(self, metrics, pairs, singletons, incremental, moved_ids) -> str:
        lines = [
            f"共放置 {len(self.tubes)} 支试管, 总质量 {metrics['total_mass_g']:g} g。"
            f"一阶不平衡量 {metrics['u1_mag_g_mm']:.2f} g·mm"
            f"(方向 {metrics['u1_angle_deg']:.1f}°), "
            f"二阶不平衡量 {metrics['u2_mag_g_mm']:.2f} g·mm, "
            f"等效偏心 {metrics['equivalent_eccentricity_um']:.2f} μm, 评级 {metrics['grade']}。"
            f"在 {self.rpm:g} rpm 下离心合力约 {metrics['centrifugal_force_n']:.2f} N, "
            f"合力矩约 {metrics['resultant_moment_n_m']:.4f} N·m。",
            f"识别出 {len(pairs)} 个对径配对、{len(singletons)} 支无搭档试管。",
        ]
        if incremental:
            lines.append(
                f"增量重排: 共移动 {len(moved_ids)} 支非锁定试管"
                + (f"(编号 {', '.join(map(str, moved_ids))})。" if moved_ids else ", 原布局完全保留。")
            )
        if self.mutex_pairs:
            ex = self._exact or {}
            pair_txt = "、".join(f"({mp.a},{mp.b})" for mp in self.mutex_pairs)
            lines.append(
                f"互斥隔离(精确搜索): 互斥试管对 {pair_txt} 均不落在物理相邻槽位"
                f"(禁用槽仍占据相邻关系); 共评估 {ex.get('legal_arrangements', 0)} 种合法摆放"
                f"(全部槽位分配 {ex.get('total_arrangements', 0)} 种), "
                "取现有评分最小值, 平分时按管号升序的槽位序列裁决。"
            )
        return " ".join(lines)


def run_plan(payload: dict[str, Any]) -> dict[str, Any]:
    """API 层入口: 解析 JSON 风格的请求并求解。"""
    slots = [
        Slot(
            index=int(s["index"]),
            angle=float(s.get("angle", 360.0 * int(s["index"]) / int(payload["slot_count"]))),
            enabled=bool(s.get("enabled", True)),
        )
        for s in payload["slots"]
    ]
    tubes = [
        Tube(
            id=int(t["id"]),
            mass=float(t["mass"]),
            locked=bool(t.get("locked", False)),
            slot=(int(t["slot"]) if t.get("slot") is not None else None),
        )
        for t in payload["tubes"]
    ]
    mutex_pairs = [
        MutexPair(a=int(mp["tube_a"]), b=int(mp["tube_b"]))
        for mp in payload.get("mutex_pairs", []) or []
    ]
    solver = Solver(
        slots=slots,
        tubes=tubes,
        mutex_pairs=mutex_pairs,
        radius_mm=float(payload.get("radius_mm", 100.0)),
        rpm=float(payload.get("rpm", 4000.0)),
        w1=float(payload.get("w1", 1.0)),
        w2=float(payload.get("w2", 0.000001)),
        move_weight=float(payload.get("move_weight", 0.0)),
        restarts=int(payload.get("restarts", 10)),
        chain_steps=int(payload.get("chain_steps", 2500)),
        seed=int(payload.get("seed", 20260923)),
    )
    previous = payload.get("previous")
    return solver.solve(previous=previous)
