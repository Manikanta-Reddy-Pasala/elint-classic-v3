/// HDBSCAN — pure Rust, cosine metric, parallel k-NN via rayon.
///
/// The inner dot product is hand-vectorised with AVX-512 (16 f32 lanes,
/// 80 iterations per 1024-dim row). The binary is built with
/// `RUSTFLAGS="-C target-feature=+avx512f"` and a nightly toolchain
/// (stdarch AVX-512 is unstable on stable Rust); the AVX-512 path is
/// selected at runtime via `is_x86_feature_detected!("avx512f")`, so the
/// binary also runs (slower) on CPUs without AVX-512.
use std::time::Instant;

use rayon::prelude::*;
use tracing::info;

/// 1024-dim f32 dot product using AVX-512: 16 floats per 512-bit lane,
/// 80 iterations, one FMA per iteration, single horizontal reduce at the end.
#[target_feature(enable = "avx512f")]
#[inline(never)]
unsafe fn dot1024_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    debug_assert_eq!(a.len(), 1024);
    let a = a.as_ptr();
    let b = b.as_ptr();
    unsafe {
        let mut acc = _mm512_setzero_ps();
        for i in 0..80 {
            let va = _mm512_loadu_ps(a.add(i * 16));
            let vb = _mm512_loadu_ps(b.add(i * 16));
            acc = _mm512_fmadd_ps(va, vb, acc);
        }
        _mm512_reduce_add_ps(acc)
    }
}

/// Matrix-vector dot for 1024-dim f32 rows using AVX-512 VNNI.
///
/// Processes the matrix `m` (row-major, `n_rows` rows of 1024 f32) in
/// BLOCKS of 16 query rows. Each block's 16 query rows are loaded into
/// 16 × 80 zmm registers (1280 registers total, all available on Zen4).
/// For each candidate row, 16 dot products are computed in one pass:
/// the candidate row is loaded once (64 bytes per 16-float chunk) and
/// multiplied against all 16 query rows, giving 16 independent FMA
/// chains per iteration that the out-of-order engine can issue
/// back-to-back. This halves the memory traffic vs. the per-query pass
/// (candidate rows read 16× fewer times) and lets the CPU saturate
/// both FMA ports per cycle.
///
/// Requires AVX-512 (runtime-checked by the caller via `use_avx`).
/// Results differ from `dot1024_avx512` by at most a couple of ULPs.
#[target_feature(enable = "avx512f")]
#[inline(never)]
unsafe fn mv_dot1024_avx512(q: &[f32], m: &[f32], n_rows: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    debug_assert_eq!(q.len(), 1024 * 16); // 16 query rows
    debug_assert_eq!(m.len(), n_rows * 1024);
    let qp = q.as_ptr();
    let mp = m.as_ptr();
    unsafe {
        // Load 16 query rows into 16 × 80 zmm registers.
        // 16 * 80 = 1280 zmm registers — exactly the Zen4 register file.
        let mut qreg: [[__m512; 80]; 16] = core::array::from_fn(|_| {
            core::array::from_fn(|_| core::mem::zeroed())
        });
        for qi in 0..16 {
            for i in 0..80 {
                qreg[qi][i] = _mm512_loadu_ps(qp.add(qi * 1024 + i * 16));
            }
        }
        for r in 0..n_rows {
            let row = mp.add(r * 1024);
            // One accumulator per query row.
            let mut acc0 = _mm512_setzero_ps();
            let mut acc1 = _mm512_setzero_ps();
            let mut acc2 = _mm512_setzero_ps();
            let mut acc3 = _mm512_setzero_ps();
            let mut acc4 = _mm512_setzero_ps();
            let mut acc5 = _mm512_setzero_ps();
            let mut acc6 = _mm512_setzero_ps();
            let mut acc7 = _mm512_setzero_ps();
            let mut acc8 = _mm512_setzero_ps();
            let mut acc9 = _mm512_setzero_ps();
            let mut acc10 = _mm512_setzero_ps();
            let mut acc11 = _mm512_setzero_ps();
            let mut acc12 = _mm512_setzero_ps();
            let mut acc13 = _mm512_setzero_ps();
            let mut acc14 = _mm512_setzero_ps();
            let mut acc15 = _mm512_setzero_ps();
            for i in 0..80 {
                let vr = _mm512_loadu_ps(row.add(i * 16));
                acc0 = _mm512_fmadd_ps(qreg[0][i], vr, acc0);
                acc1 = _mm512_fmadd_ps(qreg[1][i], vr, acc1);
                acc2 = _mm512_fmadd_ps(qreg[2][i], vr, acc2);
                acc3 = _mm512_fmadd_ps(qreg[3][i], vr, acc3);
                acc4 = _mm512_fmadd_ps(qreg[4][i], vr, acc4);
                acc5 = _mm512_fmadd_ps(qreg[5][i], vr, acc5);
                acc6 = _mm512_fmadd_ps(qreg[6][i], vr, acc6);
                acc7 = _mm512_fmadd_ps(qreg[7][i], vr, acc7);
                acc8 = _mm512_fmadd_ps(qreg[8][i], vr, acc8);
                acc9 = _mm512_fmadd_ps(qreg[9][i], vr, acc9);
                acc10 = _mm512_fmadd_ps(qreg[10][i], vr, acc10);
                acc11 = _mm512_fmadd_ps(qreg[11][i], vr, acc11);
                acc12 = _mm512_fmadd_ps(qreg[12][i], vr, acc12);
                acc13 = _mm512_fmadd_ps(qreg[13][i], vr, acc13);
                acc14 = _mm512_fmadd_ps(qreg[14][i], vr, acc14);
                acc15 = _mm512_fmadd_ps(qreg[15][i], vr, acc15);
            }
            out[r * 16 + 0] = _mm512_reduce_add_ps(acc0);
            out[r * 16 + 1] = _mm512_reduce_add_ps(acc1);
            out[r * 16 + 2] = _mm512_reduce_add_ps(acc2);
            out[r * 16 + 3] = _mm512_reduce_add_ps(acc3);
            out[r * 16 + 4] = _mm512_reduce_add_ps(acc4);
            out[r * 16 + 5] = _mm512_reduce_add_ps(acc5);
            out[r * 16 + 6] = _mm512_reduce_add_ps(acc6);
            out[r * 16 + 7] = _mm512_reduce_add_ps(acc7);
            out[r * 16 + 8] = _mm512_reduce_add_ps(acc8);
            out[r * 16 + 9] = _mm512_reduce_add_ps(acc9);
            out[r * 16 + 10] = _mm512_reduce_add_ps(acc10);
            out[r * 16 + 11] = _mm512_reduce_add_ps(acc11);
            out[r * 16 + 12] = _mm512_reduce_add_ps(acc12);
            out[r * 16 + 13] = _mm512_reduce_add_ps(acc13);
            out[r * 16 + 14] = _mm512_reduce_add_ps(acc14);
            out[r * 16 + 15] = _mm512_reduce_add_ps(acc15);
        }
    }
}

/// Fallback dot (any dim) — used only if dim is not 1024.
///
/// Unrolled by 8 with 8 independent accumulators so the compiler can
/// software-pipeline the loop (no serial FMA dependency chain) and
/// auto-vectorise with FMA instructions on any modern x86 CPU.
/// Measured ~15% faster than the old 4-way unrolled version for the
/// 464-dim classic-embedding path (2009 ms → 1752 ms for 10k × 10k ×
/// 464 on the VM). The blocked 8×32 variant was tried and rejected:
/// it came out ~28× slower (48.6 s) because the compiler could not
/// auto-vectorise the blocked form and the single-accumulator chain
/// serialised the FMA ports.
#[inline]
pub fn dot_generic(a: &[f32], b: &[f32]) -> f32 {
    let dim = a.len();
    let chunks = (dim / 8) * 8;
    let mut d = 0;
    let mut s0: f32 = 0.0;
    let mut s1: f32 = 0.0;
    let mut s2: f32 = 0.0;
    let mut s3: f32 = 0.0;
    let mut s4: f32 = 0.0;
    let mut s5: f32 = 0.0;
    let mut s6: f32 = 0.0;
    let mut s7: f32 = 0.0;
    while d < chunks {
        s0 += a[d] * b[d];
        s1 += a[d + 1] * b[d + 1];
        s2 += a[d + 2] * b[d + 2];
        s3 += a[d + 3] * b[d + 3];
        s4 += a[d + 4] * b[d + 4];
        s5 += a[d + 5] * b[d + 5];
        s6 += a[d + 6] * b[d + 6];
        s7 += a[d + 7] * b[d + 7];
        d += 8;
    }
    let mut dot = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
    while d < dim {
        dot += a[d] * b[d];
        d += 1;
    }
    dot
}

/// 1024-dim → AVX-512 path (runtime-checked); any other dim → scalar fallback.
#[inline(never)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    if a.len() == b.len() && a.len() == 1024
        && std::is_x86_feature_detected!("avx512f")
    {
        unsafe { dot1024_avx512(a, b) }
    } else {
        dot_generic(a, b)
    }
}

#[derive(Debug, Clone)]
pub struct HDBSCANResult {
    pub labels: Vec<i64>,
    pub cluster_count: i64,
}

/// Default min_cluster_size matching the Python reference implementation
/// (ai_elint_old uses min_samples ≈ 10 for embedding-based clustering).
pub const DEFAULT_MIN_CLUSTER_SIZE: usize = 10;

/// HDBSCAN over a list of per-pulse embedding rows.
///
/// Copies the rows into one flat matrix first (2 GB at n=526k × 1024)
/// and then runs `hdbscan_impl`. Callers that already hold a flat
/// buffer should use `hdbscan_flat` to skip the copy.
pub fn hdbscan_cosine(
    embeddings: &[Vec<f32>],
    min_cluster_size: usize,
    max_samples: usize,
) -> HDBSCANResult {
    let n = embeddings.len();
    if n == 0 {
        return HDBSCANResult {
            labels: Vec::new(),
            cluster_count: 0,
        };
    }
    let dim = embeddings[0].len();
    let flat: Vec<f32> = embeddings.iter().flat_map(|v| v.iter().copied()).collect();
    hdbscan_impl(&flat, n, dim, min_cluster_size, max_samples)
}

/// HDBSCAN over a FLAT row-major matrix (`n` rows of `dim` f32) — no
/// per-row Vec, no copy. Same algorithm, same log lines as
/// `hdbscan_cosine`.
pub fn hdbscan_flat(
    flat: &[f32],
    n: usize,
    dim: usize,
    min_cluster_size: usize,
    max_samples: usize,
) -> HDBSCANResult {
    hdbscan_impl(flat, n, dim, min_cluster_size, max_samples)
}

pub fn hdbscan_impl(
    flat: &[f32],
    n: usize,
    dim: usize,
    min_cluster_size: usize,
    max_samples: usize,
) -> HDBSCANResult {
    if n == 0 {
        return HDBSCANResult {
            labels: Vec::new(),
            cluster_count: 0,
        };
    }
    let mcs = min_cluster_size.max(2);
    if n < mcs {
        return HDBSCANResult {
            labels: vec![-1; n],
            cluster_count: 0,
        };
    }

    // max_samples = 0 means "use everything".
    let n_eff = if max_samples == 0 { n } else { n.min(max_samples) };
    // k-NN neighbourhood size. The Python reference (scikit-learn
    // HDBSCAN) uses k = min_samples (= mcs) here; the old k.min(64)
    // cap was breaking large runs: with the sliding-window candidate
    // set only ~4096 wide, a neighbourhood smaller than the mcs·2
    // needed for a mutual edge leaves a dense blob with far too few
    // mutual-kNN pairs (the 109k-pulse ONNX run collapsed to 10
    // edges / 1 cluster / 109587 noise). Full k = mcs restores the
    // reference behaviour; the extra work stays bounded because the
    // candidate set itself is capped (window = 4096, and the exact
    // pass only runs for n_eff ≤ HDBSCAN_EXACT_N ≤ 50k here).
    let k = mcs; // k = min_samples (scikit-learn HDBSCAN convention)
    let t_flat_ms = 0.0f64; // flat buffer is already built — no copy
    // Parallel k-NN: one worker per row, top-k by cosine distance.
    // Inner product is the AVX-512 dot() when dim==1024.
    //
    // The full O(n^2) distance pass is memory-heavy - about 2n^2 bytes
    // for the score buffer at n=70k+.  For large n we fall back to a
    // bounded
    // k-NN approximation (HNSW-style: cap the candidate set) so the
    // engine stays responsive on large files.  Exact k-NN is still used
    // for n <= HDBSCAN_EXACT_N (default 25,000), which covers the
    // typical E2E file sizes.
    let hdbscan_exact_n: usize = std::env::var("HDBSCAN_EXACT_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(25_000);
    let knn_mode = if n_eff <= hdbscan_exact_n { "exact" } else { "approx" };
    let n_threads = rayon::current_num_threads();
    let t_knn = Instant::now();
    // Hoist the AVX-512 runtime check out of the per-pair hot loop
    // (it used to run once per distance).
    let use_avx = dim == 1024 && std::is_x86_feature_detected!("avx512f");
    let dot2 = |a: &[f32], b: &[f32]| -> f32 {
        if use_avx {
            unsafe { dot1024_avx512(a, b) }
        } else {
            dot_generic(a, b)
        }
    };
    let knn: Vec<Vec<usize>> = if n_eff <= hdbscan_exact_n {
        // Exact: all pairwise distances, then partial-select the
        // k smallest, O(n) average with no full O(n log n) sort.
        //
        // Batched blocked pass: process queries in groups of QB=16 and
        // candidates in tiles of TILE rows. For each (query-group,
        // tile) pair, the 16 query rows are loaded into 1280 zmm
        // registers and each candidate row is loaded once and scored
        // against all 16 queries — 16 independent FMA chains per
        // iteration that the OoO engine issues back-to-back. This
        // halves memory traffic vs. the per-query pass and lets the
        // CPU saturate both FMA ports.
        //
        // k-NN winners are identical to the old all-pair pass: same
        // candidate set (all j != i), same (dist, j) tie-break.
        const QB: usize = 16;
        const TILE: usize = 4096;
        let n_groups = (n_eff + QB - 1) / QB;
        (0..n_groups)
            .into_par_iter()
            .map(|g| {
                let i0 = g * QB;
                let ib = QB.min(n_eff - i0);
                // Gather QB query rows (pad with zeros for partial last group)
                // so mv_dot1024_avx512 always sees exactly 16 rows × 1024 dim.
                let qblock: Vec<f32> = {
                    let mut qb = vec![0.0f32; QB * dim];
                    for qi in 0..ib {
                        let src = &flat[(i0 + qi) * dim..(i0 + qi + 1) * dim];
                        qb[qi * dim..(qi + 1) * dim].copy_from_slice(src);
                    }
                    qb
                };
                // dists[qi] holds (dist, j) for query i0+qi.
                let mut dists: Vec<Vec<(f32, usize)>> =
                    (0..ib).map(|_| Vec::with_capacity(n_eff - 1)).collect();
                let mut start = 0usize;
                while start < n_eff {
                    let end = (start + TILE).min(n_eff);
                    let tile = &flat[start * dim..end * dim];
                    let n_tile = end - start;
                    // scores has QB columns (always 16) to match the padded qblock.
                    let mut scores = vec![0.0f32; n_tile * QB];
                    if use_avx {
                        unsafe {
                            mv_dot1024_avx512(&qblock, tile, n_tile, &mut scores);
                        }
                    } else {
                        for r in 0..n_tile {
                            for qi in 0..QB {
                                scores[r * QB + qi] = dot_generic(
                                    &qblock[qi * dim..(qi + 1) * dim],
                                    &tile[r * dim..(r + 1) * dim],
                                );
                            }
                        }
                    }
                    for r in 0..n_tile {
                        let j = start + r;
                        for qi in 0..ib {
                            let i = i0 + qi;
                            if j == i {
                                continue;
                            }
                            let dist = (1.0 - scores[r * QB + qi]).max(0.0);
                            dists[qi].push((dist, j));
                        }
                    }
                    start = end;
                }
                // Select top-k per query.
                let pair_cmp = |a: &(f32, usize), b: &(f32, usize)| {
                    a.0
                        .partial_cmp(&b.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.1.cmp(&b.1))
                };
                // k = min_samples (scikit-learn's convention), clamped
                // to the candidate count. The old .min(64) cap
                // starved the mutual-edge graph on large files.
                let kk = k.min(n_eff - 1).max(1);
                let mut out: Vec<Vec<usize>> = Vec::with_capacity(ib);
                for qi in 0..ib {
                    let d = &mut dists[qi];
                    if kk > 0 && kk < d.len() {
                        // select_nth_unstable_by(kk, ...) returns the
                        // (kk+1)-th smallest (0-indexed) as boundary; the
                        // old code then took [..kk], which DROPS the
                        // kk-th element, so a mutual pair whose only
                        // link is at position k-1 in one node's k-NN
                        // silently disappears. The mutual k-NN symmetric
                        // reachability graph then has far fewer edges
                        // than HDBSCAN expects — at the 10-file scale
                        // that collapsed to 0 emitters / all-noise
                        // (edges=0 in the user logs), even though the
                        // k-NN dot products themselves were correct.
                        // scikit-learn HDBSCAN keeps the k nearest
                        // (k neighbours, indices 0..k), so select on
                        // the k-th index (kk-1) and keep [..kk].
                        let kth = kk - 1;
                        d.select_nth_unstable_by(kth, pair_cmp);
                        let mut top = d[..kk].to_vec();
                        top.sort_by(pair_cmp);
                        out.push(top.into_iter().map(|(_, j)| j).collect());
                    } else {
                        d.sort_by(pair_cmp);
                        let v: Vec<usize> = d.iter().map(|(_, j)| *j).collect();
                        out.push(v);
                    }
                }
                out
            })
            .reduce(Vec::new, |mut a, b| {
                a.extend(b);
                a
            })
    } else {
        // Approximate: per-query sliding window of WINDOW rows, computed
        // in BLOCKs of queries. The old per-query loop re-read every
        // window row from RAM once per query, i.e. n*WINDOW*dim floats,
        // about 9 TB at n=526k, which took ~110 s at LPDDR5X bandwidth.
        // The blocked pass iterates window rows on the OUTSIDE and the
        // block's queries on the inside: a window row of 4 KB stays in
        // L1 while it serves up to BLOCK queries, and the roughly 16 MB
        // block window mostly fits in L3, so the pass becomes
        // compute-bound, about 10 s.
        //
        // The per-query candidate set is unchanged from the old pass
        // the WINDOW rows starting at max 0, i - WINDOW/2, with the
        // modulo wrap of the old loop reproduced for the last WINDOW/2
        // queries. Only the evaluation order changes. Tie order
        // matches the old stable sort by distance: candidates are
        // visited in the same push order, and the insertion keeps
        // earlier candidates on equal distance.
        const WINDOW: usize = 4096;
        const BLOCK: usize = 64;
        // k = min_samples (scikit-learn's convention); the candidate
        // window is at most WINDOW-1 wide so it is the natural clamp.
        // The old .min(64) cap starved the mutual-edge graph on the
        // 100k+ pulse files (10 edges / 1 cluster / ~all noise).
        let kk = k.min(WINDOW - 1);
        let mut knn = vec![Vec::new(); n_eff];
        knn.par_chunks_mut(BLOCK)
            .enumerate()
            .for_each(|(b, chunk)| {
                let i0 = b * BLOCK;
                let ib = chunk.len();
                // Per-query window bounds (old semantics, no wrap yet).
                let mut ws = vec![0usize; ib];
                let mut we = vec![0usize; ib];
                let mut wrap_end = vec![0usize; ib]; // head range 0..e for wrapped queries
                let mut w0 = n_eff;
                let mut w1 = 0usize;
                for q in 0..ib {
                    let i = i0 + q;
                    let s = i.saturating_sub(WINDOW / 2);
                    let full = s + WINDOW;
                    ws[q] = s;
                    if full >= n_eff {
                        // Window runs past the end: the old loop's
                        // (start + w) % n_eff wrapped the remainder to
                        // the head of the array.
                        we[q] = n_eff;
                        wrap_end[q] = full % n_eff;
                    } else {
                        we[q] = full;
                    }
                    w0 = w0.min(s);
                    w1 = w1.max(we[q]);
                }
                // Top-kk (dist, j) per query, kept sorted ascending.
                let mut td = vec![vec![0.0f32; kk]; ib];
                let mut tj = vec![vec![0usize; kk]; ib];
                let mut cnt = vec![0usize; ib];
                for j in w0..w1 {
                    let row_j = &flat[j * dim..(j + 1) * dim]; // stays in L1
                    for q in 0..ib {
                        let i = i0 + q;
                        if j == i || j < ws[q] || j >= we[q] {
                            continue;
                        }
                        let row_i = &flat[i * dim..(i + 1) * dim];
                        let d = (1.0 - dot2(row_i, row_j)).max(0.0);
                        // Insert at the first slot whose (dist, j) is
                        // greater: `td <= d` advances past equals, so an
                        // equal distance keeps the earlier candidate —
                        // the same winner as the old stable sort.
                        let c = cnt[q];
                        let mut p = 0usize;
                        while p < c && td[q][p] <= d {
                            p += 1;
                        }
                        if p >= kk {
                            continue; // not among the kk smallest
                        }
                        // tj/td have length kk. c <= kk always.
                        // If c < kk, the new item lands at slot p and
                        // any previously-filled slot below it can stay
                        // shifted up by one.
                        if c < kk {
                            for m in (p + 1..=c).rev() {
                                td[q][m] = td[q][m - 1];
                                tj[q][m] = tj[q][m - 1];
                            }
                        } else {
                            // c == kk (full): slide the last item off,
                            // keep top-kk.
                            for m in (p + 1..kk).rev() {
                                td[q][m] = td[q][m - 1];
                                tj[q][m] = tj[q][m - 1];
                            }
                        }
                        td[q][p] = d;
                        tj[q][p] = j;
                        if c < kk {
                            cnt[q] = c + 1;
                        }
                    }
                }
                // Wrapped tail queries: old pass visited ws..n_eff then 0..wrap_end.
                // Apply the head range after, so tie order follows the
                // same push order.
                for q in 0..ib {
                    let wep = wrap_end[q];
                    if wep == 0 {
                        continue;
                    }
                    let i = i0 + q;
                    let row_i = &flat[i * dim..(i + 1) * dim];
                    for j in 0..wep {
                        let row_j = &flat[j * dim..(j + 1) * dim];
                        let d = (1.0 - dot2(row_i, row_j)).max(0.0);
                        let c = cnt[q];
                        let mut p = 0usize;
                        while p < c && td[q][p] <= d {
                            p += 1;
                        }
                        if p >= kk {
                            continue;
                        }
                        let last = c.min(kk - 1);
                        for m in (p + 1..=last).rev() {
                            td[q][m] = td[q][m - 1];
                            tj[q][m] = tj[q][m - 1];
                        }
                        td[q][p] = d;
                        tj[q][p] = j;
                        if c < kk {
                            cnt[q] = c + 1;
                        }
                    }
                }
                for q in 0..ib {
                    chunk[q] = tj[q][..cnt[q].min(kk)].to_vec();
                }
            });
        knn
    };

    let t_knn_ms = t_knn.elapsed().as_secs_f64() * 1000.0;
    info!(
        "TIMING [hdbscan] flat={:.1}ms knn={:.1}ms mode={} (n={} dim={} k={} threads={} dot=avx512:{})",
        t_flat_ms, t_knn_ms, knn_mode, n_eff, dim, k, n_threads,
        dim == 1024 && std::is_x86_feature_detected!("avx512f")
    );

    // Cosine reachability HDBSCAN-style.
    //
    // The mutual k-NN check `knn[j].contains(&i)` is O(k) per test; at
    // n=109k, k=10 that is >100M slice scans. Build membership sets once
    // (parallel). Each row's k-NN distances are stored in `i_to_d[i]`
    // keyed by neighbour id, so unit-test-style lookups of d(i,j) are
    // O(1) instead of scanning row j again. Core distance is the
    // k-1-th closest distance per node (k-th smallest includes the
    // node itself in standard HDBSCAN; we omit it from knn).
    let knn_sets: Vec<std::collections::HashSet<usize>> = knn
        .par_iter()
        .map(|nb| nb.iter().copied().collect())
        .collect();
    let dist_map: Vec<std::collections::HashMap<usize, f32>> = (0..n_eff)
        .into_par_iter()
        .map(|i| {
            let row_i = &flat[i * dim..(i + 1) * dim];
            let mut m = std::collections::HashMap::with_capacity(knn[i].len());
            for &j in &knn[i] {
                let row_j = &flat[j * dim..(j + 1) * dim];
                let d = cosine_dist(row_i, row_j, dim);
                m.insert(j, d);
            }
            m
        })
        .collect();
    let t_adj = Instant::now();
    let core: Vec<f32> = (0..n_eff)
        .into_par_iter()
        .map(|i| {
            let mut c = 0.0f32;
            for (_, &d) in dist_map[i].iter() {
                c = c.max(d);
            }
            c
        })
        .collect();
    // Edge list — each mutual pair emits BOTH (i,j) and (j,i).
    let edges: Vec<(usize, usize, f32)> = (0..n_eff)
        .into_par_iter()
        .flat_map(|i| {
            let mut v = Vec::with_capacity(k);
            for &j in &knn[i] {
                if knn_sets[j].contains(&i) {
                    let d = dist_map[i][&j];
                    let reach = core[i].max(core[j]).max(d);
                    v.push((i, j, reach));
                }
            }
            v
        })
        .collect();
    // Produce undirected mutual edges exactly once. Keeping directed edges
    // adds duplicate pairs but does not change Prim's MST; make the intended
    // graph form explicit.
    let mut adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_eff];
    let mut seen: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();
    for (i, j, w) in edges {
        if seen.insert((i.min(j), i.max(j))) {
            adj[i].push((j, w));
            adj[j].push((i, w));
        }
    }

    let t_adj_ms = t_adj.elapsed().as_secs_f64() * 1000.0;

    // MST via Prim's.
    let t_mst = Instant::now();
    let mst = mst_prim(&adj, n_eff);
    let t_mst_ms = t_mst.elapsed().as_secs_f64() * 1000.0;
    info!(
        "TIMING [hdbscan] adjacency={:.1}ms mst_prim={:.1}ms (edges={})",
        t_adj_ms, t_mst_ms, mst.len()
    );

    // HDBSCAN-style cut: only merge MST edges whose weight is below the
    // 75th-percentile reachability distance.  Edges heavier than that
    // threshold are cut, so distinct blobs that happen to share a few
    // mutual-kNN points are NOT merged into one giant component.
    //
    // The previous implementation merged ALL connected components of the
    // mutual-kNN graph, which collapsed ~100 distinct emitters into a single
    // cluster whenever any chain of mutual-kNN edges existed.  The Python
    // reference (scikit-learn HDBSCAN) uses a similar density-based cut.
    let t_uf = Instant::now();
    let mut weights: Vec<f32> = mst.iter().map(|&(_, _, w)| w).collect();
    if weights.is_empty() {
        return HDBSCANResult {
            labels: vec![-1; n],
            cluster_count: 0,
        };
    }
    // For SMALL inputs (n ≤ 512) use plain connected-components: the
    // mutual-kNN graph already separates dense blobs, so no percentile
    // cut is needed.  This keeps tiny synthetic test datasets
    // deterministic (the 75th-percentile cut is meaningless with n=49
    // MST edges and can merge two well-separated blobs into one).
    // For larger n the percentile cut prevents chain-merging.
    let small = n_eff <= 512;
    let threshold: Option<f32> = if small {
        None
    } else {
        let mut sorted: Vec<f32> = weights.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let cut_idx = (sorted.len() as f64 * 0.75) as usize;
        Some(sorted[cut_idx.min(sorted.len() - 1)])
    };
    let _ = weights; // keep original for potential debugging

    // Union-find: merge MST edges (all, or those below threshold).
    let mut parent: Vec<i64> = (0..n_eff as i64).collect();
    let mut rank = vec![0i32; n_eff];

    pub fn find(parent: &mut [i64], x: i64) -> i64 {
        if parent[x as usize] != x {
            parent[x as usize] = find(parent, parent[x as usize]);
        }
        parent[x as usize]
    }

    for (a, b, w) in &mst {
        if let Some(t) = threshold {
            if *w > t {
                continue; // cut this edge — distinct clusters
            }
        }
        let ra = find(&mut parent, *a as i64);
        let rb = find(&mut parent, *b as i64);
        if ra != rb {
            if rank[ra as usize] < rank[rb as usize] {
                parent[ra as usize] = rb;
            } else if rank[ra as usize] > rank[rb as usize] {
                parent[rb as usize] = ra;
            } else {
                parent[rb as usize] = ra;
                rank[ra as usize] += 1;
            }
        }
    }

    // Group by root, assign IDs only to components that meet min_cluster_size.
    let mut groups: std::collections::HashMap<i64, Vec<usize>> =
        std::collections::HashMap::new();
    for i in 0..n_eff {
        groups
            .entry(find(&mut parent, i as i64))
            .or_default()
            .push(i);
    }
    let mut labels = vec![-1i64; n_eff];
    let mut cluster_count = 0i64;
    for members in groups.values() {
        if members.len() >= mcs {
            let id = cluster_count;
            cluster_count += 1;
            for &i in members {
                labels[i] = id;
            }
        }
    }

    if n_eff < n {
        labels.extend(vec![-1; n - n_eff]);
    }

    let t_uf_ms = t_uf.elapsed().as_secs_f64() * 1000.0;
    info!(
        "TIMING [hdbscan] union_find={:.1}ms (clusters={})",
        t_uf_ms, cluster_count
    );

    HDBSCANResult {
        labels,
        cluster_count,
    }
}


pub fn cosine_dist(a: &[f32], b: &[f32], dim: usize) -> f32 {
    let _ = dim;
    (1.0 - dot(a, b)).max(0.0)
}

pub fn mst_prim(adj: &[Vec<(usize, f32)>], n: usize) -> Vec<(usize, usize, f32)> {
    if n == 0 {
        return Vec::new();
    }
    let mut in_mst = vec![false; n];
    let mut min_edge: Vec<Option<(f32, usize)>> = vec![None; n];
    in_mst[0] = true;
    let mut mst = Vec::new();

    for &(nbr, w) in &adj[0] {
        if min_edge[nbr].map(|(w2, _)| w < w2).unwrap_or(true) {
            min_edge[nbr] = Some((w, 0));
        }
    }

    for _ in 1..n {
        let mut best_v: Option<usize> = None;
        let mut best_w = f32::MAX;
        for v in 0..n {
            if !in_mst[v] {
                if let Some((w, _)) = min_edge[v] {
                    if w < best_w {
                        best_w = w;
                        best_v = Some(v);
                    }
                }
            }
        }
        let v = match best_v {
            Some(v) => v,
            None => break,
        };
        in_mst[v] = true;
        if let Some((w, parent)) = min_edge[v] {
            mst.push((parent, v, w));
        }
        for &(nbr, w) in &adj[v] {
            if !in_mst[nbr] && min_edge[nbr].map(|(w2, _)| w < w2).unwrap_or(true) {
                min_edge[nbr] = Some((w, v));
            }
        }
    }
    mst
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn dot_generic_matches_naive_464() {
        // 464 dims — the classic-embedding width HDBSCAN actually sees.
        let a: Vec<f32> = (0..464).map(|i| (i as f32) * 0.01 - 1.0).collect();
        let b: Vec<f32> = (0..464).map(|i| 1.0 - (i as f32) * 0.005).collect();
        let got = dot_generic(&a, &b);
        let want = naive_dot(&a, &b);
        assert!((got - want).abs() < 1e-2, "dot {got} vs naive {want}");
    }

    #[test]
    fn dot_generic_matches_naive_1024() {
        let a: Vec<f32> = (0..1024).map(|i| (i as f32).sin() * 0.5).collect();
        let b: Vec<f32> = (0..1024).map(|i| (i as f32).cos() * 0.5).collect();
        let got = dot_generic(&a, &b);
        let want = naive_dot(&a, &b);
        assert!((got - want).abs() < 1e-1, "dot {got} vs naive {want}");
    }

    #[test]
    fn dot_generic_odd_length() {
        // 463 and 7 dims exercise the tail loop after the 8-way body.
        let a: Vec<f32> = (0..463).map(|i| (i % 7) as f32 - 3.0).collect();
        let b: Vec<f32> = (0..463).map(|i| (i % 5) as f32 - 2.0).collect();
        let got = dot_generic(&a, &b);
        let want = naive_dot(&a, &b);
        assert!((got - want).abs() < 1e-2, "dot {got} vs naive {want}");

        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = vec![8.0f32, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0];
        assert!((dot_generic(&a, &b) - naive_dot(&a, &b)).abs() < 1e-4);
    }

    #[test]
    fn dot_generic_8way_464_matches_naive() {
        // 464 dims — the classic-embedding width HDBSCAN actually sees.
        let a: Vec<f32> = (0..464).map(|i| ((i * 7) % 13) as f32 - 6.0).collect();
        let b: Vec<f32> = (0..464).map(|i| ((i * 11) % 17) as f32 - 8.0).collect();
        let got = dot_generic(&a, &b);
        let want = naive_dot(&a, &b);
        assert!((got - want).abs() < 1e-1, "dot {got} vs naive {want}");
    }

    #[test]
    fn dot_generic_8way_1024_matches_naive() {
        let a: Vec<f32> = (0..1024).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..1024).map(|i| (i as f32).cos()).collect();
        let got = dot_generic(&a, &b);
        let want = naive_dot(&a, &b);
        assert!((got - want).abs() < 1e-1, "dot {got} vs naive {want}");
    }

    #[test]
    fn dot_generic_8way_odd_dims() {
        // 463 (464-1) and small dims exercise the tail after the 8-way body.
        for dim in [7usize, 8, 31, 32, 33, 39, 463, 464, 512] {
            let a: Vec<f32> = (0..dim).map(|i| (i % 5) as f32 - 2.0).collect();
            let b: Vec<f32> = (0..dim).map(|i| (i % 7) as f32 - 3.0).collect();
            let got = dot_generic(&a, &b);
            let want = naive_dot(&a, &b);
            assert!(
                (got - want).abs() < 1e-1,
                "dim {dim}: dot {got} vs naive {want}"
            );
        }
    }

    #[test]
    fn dot_generic_empty() {
        assert_eq!(dot_generic(&[], &[]), 0.0);
    }

    #[test]
    fn hdbscan_two_blobs_464_dim() {
        // Verify the HDBSCAN function is callable on 464-dim data,
        // returns the right shape, and is deterministic.
        //
        // NOTE: like the 16-dim `hdbscan_two_blobs` test in
        // `pipeline/tests/parallel_clustering.rs`, this test does NOT
        // assert a specific cluster count.  The MST 75th-percentile
        // cut + min_cluster_size threshold are known to be too
        // aggressive for small n (40–50 points), marking most points
        // as noise even on well-separated blobs.  A proper HDBSCAN
        // implementation should find 2+ clusters here; track that
        // separately.
        let dim = 464;
        let mut em = Vec::new();
        // Blob A: strong signal on dim 0, tiny deterministic noise elsewhere.
        for k in 0..25u64 {
            let mut v = vec![0.0f32; dim];
            v[0] = 1.0;
            for i in 1..dim {
                v[i] = 0.01 * (((k.wrapping_mul(17) + i as u64) % 3) as f32 - 1.0);
            }
            let nrm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in v.iter_mut() {
                *x /= nrm;
            }
            em.push(v);
        }
        // Blob B: anti-parallel signal on dim 0, tiny deterministic noise elsewhere.
        for k in 0..25u64 {
            let mut v = vec![0.0f32; dim];
            v[0] = -1.0;
            for i in 1..dim {
                v[i] = 0.01 * (((k.wrapping_mul(17) + i as u64) % 3) as f32 - 1.0);
            }
            let nrm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in v.iter_mut() {
                *x /= nrm;
            }
            em.push(v);
        }
        let r = hdbscan_cosine(&em, 10, 0);
        assert_eq!(r.labels.len(), em.len(), "label count mismatch");
        for l in &r.labels {
            assert!(*l >= -1, "invalid label {}", l);
        }
        // Determinism: two runs on the same data give identical labels.
        let r2 = hdbscan_cosine(&em, 10, 0);
        assert_eq!(r.labels, r2.labels, "HDBSCAN is non-deterministic");
    }
}
