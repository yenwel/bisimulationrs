//! Network flows on small graphs with real capacities: maximum flow (for
//! the weight-function test of (ε,δ)-bisimulation) and optimal transport
//! between two discrete distributions (for Kantorovich distances).

use std::collections::VecDeque;

/// Flows below this are treated as zero.
const EPS: f64 = 1e-12;

/// A flow network built edge by edge.
#[derive(Debug, Clone)]
pub struct FlowNetwork {
    /// Per node: indices into `edges`.
    adj: Vec<Vec<usize>>,
    /// `(to, residual capacity, cost)`; edge `i ^ 1` is the reverse of `i`.
    edges: Vec<(usize, f64, f64)>,
}

impl FlowNetwork {
    pub fn new(nodes: usize) -> Self {
        Self {
            adj: vec![Vec::new(); nodes],
            edges: Vec::new(),
        }
    }

    /// Adds `from → to` with `capacity` and per-unit `cost`.
    pub fn add_edge(&mut self, from: usize, to: usize, capacity: f64, cost: f64) {
        self.adj[from].push(self.edges.len());
        self.edges.push((to, capacity, cost));
        self.adj[to].push(self.edges.len());
        self.edges.push((from, 0.0, -cost));
    }

    /// Maximum flow from `source` to `sink` (Dinic: BFS levels, blocking
    /// flows by depth-first search with an explicit stack).
    pub fn max_flow(&mut self, source: usize, sink: usize) -> f64 {
        let n = self.adj.len();
        let mut total = 0.0;
        loop {
            let mut level = vec![usize::MAX; n];
            level[source] = 0;
            let mut queue = VecDeque::from([source]);
            while let Some(u) = queue.pop_front() {
                for &e in &self.adj[u] {
                    let (v, cap, _) = self.edges[e];
                    if cap > EPS && level[v] == usize::MAX {
                        level[v] = level[u] + 1;
                        queue.push_back(v);
                    }
                }
            }
            if level[sink] == usize::MAX {
                return total;
            }
            let mut next = vec![0usize; n];
            loop {
                let pushed = self.augment(source, sink, &level, &mut next);
                if pushed <= EPS {
                    break;
                }
                total += pushed;
            }
        }
    }

    /// One augmenting path in the level graph; returns the flow pushed.
    fn augment(&mut self, source: usize, sink: usize, level: &[usize], next: &mut [usize]) -> f64 {
        let mut path: Vec<usize> = Vec::new();
        let mut u = source;
        loop {
            if u == sink {
                let pushed = path
                    .iter()
                    .map(|&e| self.edges[e].1)
                    .fold(f64::INFINITY, f64::min);
                for &e in &path {
                    self.edges[e].1 -= pushed;
                    self.edges[e ^ 1].1 += pushed;
                }
                return pushed;
            }
            let mut advanced = false;
            while next[u] < self.adj[u].len() {
                let e = self.adj[u][next[u]];
                let (v, cap, _) = self.edges[e];
                if cap > EPS && level[v] == level[u] + 1 {
                    path.push(e);
                    u = v;
                    advanced = true;
                    break;
                }
                next[u] += 1;
            }
            if !advanced {
                // Dead end: retreat and skip the edge that led here.
                let Some(e) = path.pop() else {
                    return 0.0;
                };
                u = self.edges[e ^ 1].0;
                next[u] += 1;
            }
        }
    }
}

/// An optimal coupling of `a` and `b` (non-negative, equal totals) under
/// `cost[i][j]`: `(total cost, [(i, j, mass)])`. Successive shortest paths
/// with Bellman–Ford, so negative reduced costs need no potentials; meant
/// for supports of a few dozen points.
///
/// # Panics
/// If the totals of `a` and `b` differ by more than `1e-9`.
pub fn transport(a: &[f64], b: &[f64], cost: &[Vec<f64>]) -> (f64, Vec<(usize, usize, f64)>) {
    let (m, n) = (a.len(), b.len());
    let (ta, tb): (f64, f64) = (a.iter().sum(), b.iter().sum());
    assert!(
        (ta - tb).abs() <= 1e-9 * ta.max(1.0),
        "unequal masses {ta} and {tb}"
    );
    // Nodes: source, a-side 0..m, b-side m..m+n, sink.
    let (source, sink) = (m + n, m + n + 1);
    let mut net = FlowNetwork::new(m + n + 2);
    for (i, &ai) in a.iter().enumerate() {
        net.add_edge(source, i, ai, 0.0);
    }
    for (j, &bj) in b.iter().enumerate() {
        net.add_edge(m + j, sink, bj, 0.0);
    }
    let first_pair = net.edges.len();
    for (i, row) in cost.iter().enumerate().take(m) {
        for (j, &c) in row.iter().enumerate().take(n) {
            net.add_edge(i, m + j, f64::INFINITY, c);
        }
    }
    let mut remaining = ta.min(tb);
    let nodes = m + n + 2;
    while remaining > EPS * ta.max(1.0) {
        // Bellman–Ford shortest path in the residual graph.
        let mut dist = vec![f64::INFINITY; nodes];
        let mut via = vec![usize::MAX; nodes];
        dist[source] = 0.0;
        for _ in 0..nodes {
            let mut changed = false;
            for u in 0..nodes {
                if dist[u].is_infinite() {
                    continue;
                }
                for &e in &net.adj[u] {
                    let (v, cap, c) = net.edges[e];
                    if cap > EPS && dist[u] + c < dist[v] - 1e-15 {
                        dist[v] = dist[u] + c;
                        via[v] = e;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        if dist[sink].is_infinite() {
            break;
        }
        let mut push = remaining;
        let mut v = sink;
        while v != source {
            let e = via[v];
            push = push.min(net.edges[e].1);
            v = net.edges[e ^ 1].0;
        }
        let mut v = sink;
        while v != source {
            let e = via[v];
            net.edges[e].1 -= push;
            net.edges[e ^ 1].1 += push;
            v = net.edges[e ^ 1].0;
        }
        remaining -= push;
    }
    let mut plan = Vec::new();
    let mut total = 0.0;
    for (i, row) in cost.iter().enumerate().take(m) {
        for (j, &c) in row.iter().enumerate().take(n) {
            let e = first_pair + 2 * (i * n + j);
            let mass = net.edges[e ^ 1].1;
            if mass > EPS {
                plan.push((i, j, mass));
                total += mass * c;
            }
        }
    }
    (total, plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_flow_of_a_small_network() {
        // Classic: 0→1 (3), 0→2 (2), 1→2 (1), 1→3 (2), 2→3 (3): max flow 5.
        let mut g = FlowNetwork::new(4);
        for (u, v, c) in [
            (0, 1, 3.0),
            (0, 2, 2.0),
            (1, 2, 1.0),
            (1, 3, 2.0),
            (2, 3, 3.0),
        ] {
            g.add_edge(u, v, c, 0.0);
        }
        assert!((g.max_flow(0, 3) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn transport_matches_brute_force_on_two_points() {
        // Moving mass on a line: the optimal cost is the 1-Wasserstein
        // distance |0.3 - 0.0| * 1 = 0.3 for a = (0.5, 0.5), b = (0.8, 0.2)
        // with |i - j| costs.
        let cost = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let (c, plan) = transport(&[0.5, 0.5], &[0.8, 0.2], &cost);
        assert!((c - 0.3).abs() < 1e-12, "{c}");
        let moved: f64 = plan.iter().map(|p| p.2).sum();
        assert!((moved - 1.0).abs() < 1e-12);
    }

    #[test]
    fn transport_prefers_cheap_cycles() {
        // Three points on a line, a = δ₀ + δ₂ halves, b = uniform-ish.
        let pos: [f64; 3] = [0.0, 1.0, 2.0];
        let cost: Vec<Vec<f64>> = pos
            .iter()
            .map(|x| pos.iter().map(|y| (x - y).abs()).collect())
            .collect();
        let a = [0.5, 0.0, 0.5];
        let b = [0.25, 0.5, 0.25];
        let (c, _) = transport(&a, &b, &cost);
        // W1 on the line = ∫|F_a − F_b| = 0.25 + 0.25.
        assert!((c - 0.5).abs() < 1e-12, "{c}");
    }
}
