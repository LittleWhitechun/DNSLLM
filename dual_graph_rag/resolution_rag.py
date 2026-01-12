# resolution_rag.py
from __future__ import annotations

import math
import pickle
import re
from dataclasses import dataclass
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx


# -----------------------------
# Globals (initialized by init)
# -----------------------------
_RG: Optional[nx.DiGraph] = None
_RETRIEVER: Optional["ResolutionPathRetriever"] = None
_ASN_LOOKUP: Optional["ASNLookup"] = None


# -----------------------------
# Utils
# -----------------------------
_IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

def is_ipv4(ip: str) -> bool:
    if not ip or not _IP_RE.match(ip):
        return False
    try:
        return all(0 <= int(p) <= 255 for p in ip.split("."))
    except Exception:
        return False

def nodeD(domain: str) -> str:
    return f"D:{domain.strip().lower()}"

def nodeS(asn: int | str) -> str:
    return f"S:{int(asn)}"

def parse_day_yyyymmdd(day: str) -> date:
    return datetime.strptime(day, "%Y%m%d").date()

def extract_day_yyyymmdd(ts: str) -> Optional[str]:
    # "2025-09-04T08:44:02+08:00" -> "20250904"
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts).strftime("%Y%m%d")
    except Exception:
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", ts)
        if not m:
            return None
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"

def clamp_nonneg(x: int) -> int:
    return x if x >= 0 else 0


# -----------------------------
# ASN lookup (geoip2 or maxminddb)
# -----------------------------
class ASNLookup:
    def __init__(self, asn_mmdb_path: str):
        self._mode = None
        self._reader = None
        try:
            import geoip2.database  # type: ignore
            self._mode = "geoip2"
            self._reader = geoip2.database.Reader(asn_mmdb_path)
        except Exception:
            self._mode = "maxminddb"
            import maxminddb  # type: ignore
            self._reader = maxminddb.open_database(asn_mmdb_path)

        self._cache: Dict[str, Optional[int]] = {}

    def close(self) -> None:
        try:
            if self._reader is not None:
                self._reader.close()
        except Exception:
            pass

    def lookup_asn(self, ip: str) -> Optional[int]:
        if ip in self._cache:
            return self._cache[ip]
        asn = None
        try:
            if self._mode == "geoip2":
                resp = self._reader.asn(ip)  # type: ignore
                asn = getattr(resp, "autonomous_system_number", None)
            else:
                data = self._reader.get(ip)  # type: ignore
                if isinstance(data, dict):
                    asn = data.get("autonomous_system_number")
        except Exception:
            asn = None

        if asn is not None:
            try:
                asn = int(asn)
            except Exception:
                asn = None
        self._cache[ip] = asn
        return asn


# -----------------------------
# Edge temporal weight
# -----------------------------
def edge_temporal_weight(G: nx.DiGraph, u: str, v: str, query_day: str, lam: float = 0.25) -> float:
    # directed D->S, but traverse undirected: try both (u,v) and (v,u)
    data = G[u][v] if G.has_edge(u, v) else (G[v][u] if G.has_edge(v, u) else None)
    if data is None:
        return 0.0

    day_counts = data.get("day_counts")
    if not isinstance(day_counts, dict) or not day_counts:
        days = data.get("days")
        if isinstance(days, (set, list, tuple)) and days:
            day_counts = {str(d): 1 for d in days}
        else:
            return 0.0

    qd = parse_day_yyyymmdd(query_day)
    num = 0.0
    den = 0.0
    for d, c in day_counts.items():
        try:
            dd = parse_day_yyyymmdd(str(d))
            c = int(c)
        except Exception:
            continue
        if c <= 0:
            continue
        delta = clamp_nonneg((qd - dd).days)
        w = math.exp(-lam * delta)
        num += c * w
        den += c
    return (num / den) if den > 0 else 0.0


# -----------------------------
# Beam search
# -----------------------------
@dataclass
class RPath:
    nodes: List[str]
    edge_weights: List[float]

    def score(self) -> float:
        L = len(self.edge_weights)
        return 0.0 if L == 0 else (sum(self.edge_weights) / L) / math.sqrt(L)


class ResolutionPathRetriever:
    def __init__(self, G: nx.DiGraph, lam: float = 0.25, max_expand_neighbors: int = 5, avoid_cycles: bool = True):
        self.G = G
        self.lam = lam
        self.max_expand_neighbors = max_expand_neighbors
        self.avoid_cycles = avoid_cycles

    def _neighbors_undirected(self, u: str) -> List[str]:
        if u not in self.G:
            return []
        nbrs = set(self.G.successors(u)) | set(self.G.predecessors(u))
        return list(nbrs)

    def _top_neighbors(self, u: str, query_day: str, visited: Optional[set]) -> List[Tuple[str, float]]:
        cand = []
        for v in self._neighbors_undirected(u):
            if self.avoid_cycles and visited is not None and v in visited:
                continue
            cand.append((v, edge_temporal_weight(self.G, u, v, query_day, lam=self.lam)))
        cand.sort(key=lambda x: x[1], reverse=True)
        return cand[: self.max_expand_neighbors]

    def beam_search(self, start: str, target: str, query_day: str, max_hops: int = 4, beam_width: int = 30, top_k: int = 5) -> List[RPath]:
        if start not in self.G or target not in self.G:
            return []

        frontier = [RPath(nodes=[start], edge_weights=[])]
        completed: List[RPath] = []

        for _ in range(max_hops):
            nxt: List[RPath] = []
            for p in frontier:
                u = p.nodes[-1]
                visited = set(p.nodes) if self.avoid_cycles else None
                for v, w in self._top_neighbors(u, query_day, visited):
                    np = RPath(nodes=p.nodes + [v], edge_weights=p.edge_weights + [w])
                    (completed if v == target else nxt).append(np)

            if not nxt:
                break
            nxt.sort(key=lambda x: x.score(), reverse=True)
            frontier = nxt[:beam_width]

        completed.sort(key=lambda x: x.score(), reverse=True)
        return completed[:top_k]


# -----------------------------
# Public API
# -----------------------------
def init(
    graph_path: str = "data/resolution_graph.gpickle",
    asn_mmdb: str = "data/GeoLite2-ASN_20250702/GeoLite2-ASN.mmdb",
    lam: float = 0.25,
    max_expand_neighbors: int = 5,
    avoid_cycles: bool = True,
) -> None:
    """
    Load graph + init retriever + init asn lookup.
    Call once in your pipeline.
    """
    global _RG, _RETRIEVER, _ASN_LOOKUP
    with open(graph_path, "rb") as f:
        _RG = pickle.load(f)
    _RETRIEVER = ResolutionPathRetriever(_RG, lam=lam, max_expand_neighbors=max_expand_neighbors, avoid_cycles=avoid_cycles)
    _ASN_LOOKUP = ASNLookup(asn_mmdb)


def close() -> None:
    global _ASN_LOOKUP
    if _ASN_LOOKUP is not None:
        _ASN_LOOKUP.close()
        _ASN_LOOKUP = None


def get_paths(
    domain: str,
    answer_ips: List[str],
    day: str,
    max_hops: int = 4,
    beam_width: int = 30,
    top_k_each_asn: int = 3,
) -> List[Dict[str, Any]]:
    """
    输入解析后的字段（domain/answer_ips/day），输出 resolution paths。
    """
    if _RETRIEVER is None or _RG is None or _ASN_LOOKUP is None:
        raise RuntimeError("resolution_rag not initialized. Call resolution_rag.init() first.")

    domain = domain.strip().lower()
    if not domain or not day:
        return []

    # answer_ip -> asn
    asns: List[int] = []
    for ip in answer_ips:
        ip = ip.strip()
        if not is_ipv4(ip):
            continue
        asn = _ASN_LOOKUP.lookup_asn(ip)
        if asn is not None:
            asns.append(asn)

    # 去重保持顺序
    seen = set()
    asns = [x for x in asns if not (x in seen or seen.add(x))]
    if not asns:
        return []

    start = nodeD(domain)
    all_paths: List[Dict[str, Any]] = []
    for asn in asns:
        target = nodeS(asn)
        for p in _RETRIEVER.beam_search(start, target, day, max_hops=max_hops, beam_width=beam_width, top_k=top_k_each_asn):
            all_paths.append({
                "pair": f"{start} <-> {target}",
                "score": p.score(),
                "nodes": p.nodes,
                "edge_weights": p.edge_weights,
            })
    all_paths.sort(key=lambda x: x["score"], reverse=True)
    return all_paths


def get_paths_from_record(
    record: Dict[str, Any],
    max_hops: int = 4,
    beam_width: int = 30,
    top_k_each_asn: int = 3,
) -> List[Dict[str, Any]]:
    """
    直接输入 dataset_sample.jsonl 解析出来的一条 dict，内部提取 domain/answer_ips/day
    """
    domain = (record.get("name") or "").strip().lower()
    day = extract_day_yyyymmdd((record.get("timestamp") or "").strip())
    answers = ((record.get("data") or {}).get("answers") or [])

    answer_ips = []
    for ans in answers:
        if str(ans.get("type", "")).upper() != "A":
            continue
        ip = str(ans.get("answer", "")).strip()
        if is_ipv4(ip):
            answer_ips.append(ip)

    # 去重保持顺序
    seen = set()
    answer_ips = [x for x in answer_ips if not (x in seen or seen.add(x))]

    return get_paths(domain, answer_ips, day, max_hops=max_hops, beam_width=beam_width, top_k_each_asn=top_k_each_asn)
