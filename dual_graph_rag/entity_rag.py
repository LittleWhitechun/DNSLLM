# entity_rag.py
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx


# -----------------------------
# Globals (initialized by init)
# -----------------------------
_EG: Optional[nx.MultiDiGraph] = None
_RETRIEVER: Optional["EntityPathRetriever"] = None


def node_id(ntype: str, value: str) -> str:
    return f"{ntype}:{value}"


def base_rtype(rtype: str) -> str:
    return rtype[:-4] if rtype.endswith("_rev") else rtype


def default_relation_importance() -> Dict[str, float]:
    return {
        "country_has_policy": 0.95,
        "policy_targets_country": 0.95,
        "resolver_in_org": 0.85,
        "answer_ip_in_org": 0.85,
        "resolver_in_asn": 0.55,
        "answer_ip_in_asn": 0.55,
        "domain_category": 0.55,
        "domain_in_country": 0.35,
        "resolver_in_country": 0.30,
        "answer_ip_in_country": 0.30,
    }


def compute_pagerank_centrality(G: nx.MultiDiGraph, alpha: float = 0.85, max_iter: int = 100) -> Dict[str, float]:
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, _, _ in G.edges(keys=True, data=True):
        if H.has_edge(u, v):
            H[u][v]["weight"] += 1.0
        else:
            H.add_edge(u, v, weight=1.0)
    return nx.pagerank(H, alpha=alpha, max_iter=max_iter, weight="weight")


@dataclass
class Path:
    nodes: List[str]
    edges: List[str]
    sum_score: float

    def re_score(self) -> float:
        L = len(self.edges)
        return 0.0 if L == 0 else self.sum_score / math.sqrt(L)


class EntityPathRetriever:
    def __init__(
        self,
        G: nx.MultiDiGraph,
        centrality: Optional[Dict[str, float]] = None,
        relation_importance: Optional[Dict[str, float]] = None,
        decay_alpha: float = 0.6,
        max_expand_edges: int = 5,
        avoid_cycles: bool = True,
        per_neighbor_keep: int = 2,
    ):
        self.G = G
        self.I = relation_importance or default_relation_importance()
        self.C = centrality or compute_pagerank_centrality(G)
        self.decay_alpha = decay_alpha
        self.max_expand_edges = max_expand_edges
        self.avoid_cycles = avoid_cycles
        self.per_neighbor_keep = per_neighbor_keep

    def _I(self, rtype: str) -> float:
        return float(self.I.get(base_rtype(rtype), 0.3))

    def _C(self, v: str) -> float:
        return float(self.C.get(v, 1e-6))

    def _beta(self, hop_i: int) -> float:
        return math.exp(-self.decay_alpha * (hop_i - 1))

    def _top_expansions(self, u: str, visited: Optional[set], target: Optional[str]) -> List[Tuple[str, str, float]]:
        """
        局部扩展 topK，并修复“target 被 topK 剪掉”的问题：若 u -> target 存在则强制保留。
        """
        if u not in self.G:
            return []

        candidates: List[Tuple[str, str, float]] = []
        forced: Optional[Tuple[str, str, float]] = None

        for _, v, k, data in self.G.out_edges(u, keys=True, data=True):
            if self.avoid_cycles and visited is not None and v in visited:
                continue
            rtype = str(data.get("rtype", "unknown"))
            score = self._I(rtype) * self._C(v)
            candidates.append((v, rtype, score))
            if target is not None and v == target and forced is None:
                forced = (v, rtype, score)

        if not candidates:
            return []

        by_v: Dict[str, List[Tuple[str, str, float]]] = {}
        for v, rtype, score in candidates:
            by_v.setdefault(v, []).append((v, rtype, score))

        compressed: List[Tuple[str, str, float]] = []
        for v, arr in by_v.items():
            arr.sort(key=lambda x: x[2], reverse=True)
            compressed.extend(arr[: self.per_neighbor_keep])

        compressed.sort(key=lambda x: x[2], reverse=True)
        out = compressed[: self.max_expand_edges]

        if forced is not None:
            key_forced = (forced[0], forced[1])
            if all((v, r) != key_forced for v, r, _ in out):
                if len(out) < self.max_expand_edges:
                    out.append(forced)
                else:
                    out[-1] = forced
        return out

    def beam_search_paths(self, start: str, target: str, max_depth: int = 5, beam_width: int = 20, top_k: int = 5) -> List[Path]:
        if start not in self.G or target not in self.G:
            return []

        frontier: List[Path] = [Path(nodes=[start], edges=[], sum_score=0.0)]
        completed: List[Path] = []

        for _ in range(max_depth):
            nxt: List[Path] = []
            for p in frontier:
                u = p.nodes[-1]
                visited = set(p.nodes) if self.avoid_cycles else None
                for v, rtype, _ in self._top_expansions(u, visited, target):
                    hop_i = len(p.edges) + 1
                    contrib = self._beta(hop_i) * self._I(rtype) * self._C(v)
                    np = Path(nodes=p.nodes + [v], edges=p.edges + [rtype], sum_score=p.sum_score + contrib)
                    (completed if v == target else nxt).append(np)

            if not nxt:
                break
            nxt.sort(key=lambda x: x.re_score(), reverse=True)
            frontier = nxt[:beam_width]

        completed.sort(key=lambda x: x.re_score(), reverse=True)
        return completed[:top_k]


# -----------------------------
# Public API
# -----------------------------
def init(
    graph_path: str = "data/entity_graph.gpickle",
    decay_alpha: float = 0.6,
    max_expand_edges: int = 5,
    beam_width: int = 20,
    avoid_cycles: bool = True,
) -> None:
    """
    Load entity graph and init retriever. Call once.
    """
    global _EG, _RETRIEVER
    with open(graph_path, "rb") as f:
        _EG = pickle.load(f)

    # 这里 retriever 内部计算 pagerank 可能耗时一次，但只做一次
    _RETRIEVER = EntityPathRetriever(
        _EG,
        decay_alpha=decay_alpha,
        max_expand_edges=max_expand_edges,
        avoid_cycles=avoid_cycles,
    )


def get_paths(
    domain: str,
    resolver_ip: str,
    answer_ip: str,
    max_depth: int = 5,
    beam_width: int = 20,
    top_k_each_pair: int = 3,
) -> List[Dict[str, Any]]:
    """
    输入解析后的字段（domain/resolver_ip/answer_ip），输出 entity paths。
    """
    if _RETRIEVER is None:
        raise RuntimeError("entity_rag not initialized. Call entity_rag.init() first.")

    D = node_id("D", domain.strip().lower())
    R = node_id("R", resolver_ip.strip())
    A = node_id("A", answer_ip.strip())

    pairs = [("R->D", R, D), ("D->A", D, A), ("R->A", R, A)]
    out: List[Dict[str, Any]] = []

    for tag, s, t in pairs:
        for p in _RETRIEVER.beam_search_paths(s, t, max_depth=max_depth, beam_width=beam_width, top_k=top_k_each_pair):
            out.append({"pair": tag, "score": p.re_score(), "nodes": p.nodes, "edges": p.edges})

    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def get_paths_from_record(
    record: Dict[str, Any],
    max_depth: int = 5,
    beam_width: int = 20,
    top_k_each_pair: int = 3,
) -> List[Dict[str, Any]]:
    """
    直接输入 dataset_sample.jsonl 的一条 dict，内部提取 domain/resolver_ip/answer_ip(第一个A记录)
    若你想对多个 answer_ip 都跑，在外层循环处理更灵活。
    """
    domain = (record.get("name") or "").strip().lower()
    data = record.get("data") or {}
    resolver = (data.get("resolver") or "").strip()
    resolver_ip = resolver.split(":")[0].strip() if resolver else ""

    # 默认取第一条 A 记录（你外层若要对每个 answer 都跑，建议用 get_paths(...)）
    answer_ip = ""
    for ans in (data.get("answers") or []):
        if str(ans.get("type", "")).upper() == "A":
            answer_ip = str(ans.get("answer", "")).strip()
            break

    if not domain or not resolver_ip or not answer_ip:
        return []
    return get_paths(domain, resolver_ip, answer_ip, max_depth=max_depth, beam_width=beam_width, top_k_each_pair=top_k_each_pair)
