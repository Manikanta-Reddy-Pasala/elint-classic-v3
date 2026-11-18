//! ONNX per-pulse predictions: centre label + similarity.
//!
//! The model service's `/predict` endpoint returns, for every pulse, the
//! nearest learned centre (a model-side label id) and the similarity to
//! it. A pulse whose similarity clears `KNOWN_THRESHOLD` belongs to a
//! known emitter and is written as an emitter + signal directly; the rest
//! go through the usual unknown-pulse clustering.
//!
//! The labels are part of the EMBEDDING call: a batch's embeddings and
//! its labels are produced by the same request, so they must stay
//! together. Keeping them in one struct (carried with the buffered
//! embeddings) is what stops a later batch's labels being applied to an
//! earlier batch's pulses.
//!
//! Classic never builds this value (`predictions: None`).

use tracing::warn;

use crate::global_match::AiPulseMeta;

/// Per-pulse model predictions for one buffered batch, in batch order.
#[derive(Clone, Debug, Default)]
pub struct OnnxPredictions {
    /// Model-side centre id per pulse; -1 when the model has no centre.
    pub labels: Vec<i64>,
    /// Similarity to that centre, 0.0 when unavailable.
    pub similarities: Vec<f32>,
}

impl OnnxPredictions {
    /// True when this pulse is predicted as known: a real centre id and a
    /// similarity at or above the configured threshold.
    pub fn is_known(&self, i: usize, threshold: f32) -> bool {
        match (self.labels.get(i), self.similarities.get(i)) {
            (Some(&l), Some(&s)) => l >= 0 && s >= threshold,
            _ => false,
        }
    }

    /// Number of pulses with a usable prediction row.
    pub fn len(&self) -> usize {
        self.labels.len().min(self.similarities.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Aggregate stats per model centre id, collected across batches.
pub struct PredictionGroups {
    groups: std::collections::HashMap<i64, Group>,
}

#[derive(Default)]
pub struct Group {
    pub freqs: Vec<f64>,
    pub powers: Vec<f64>,
    pub snrs: Vec<f64>,
    pub pws: Vec<f64>,
    pub bws: Vec<f64>,
    pub doas: Vec<f64>,
    pub toas: Vec<f64>,
    pub vector: Option<Vec<f32>>,
    pub name: Option<String>,
}

impl PredictionGroups {
    pub fn new() -> Self {
        Self {
            groups: std::collections::HashMap::new(),
        }
    }

    fn push(&mut self, id: i64, m: &AiPulseMeta, emb: Option<&Vec<f32>>) {
        let g = self.groups.entry(id).or_default();
        g.freqs.push(m.center_frequency_hz);
        g.powers.push(m.power_dbm);
        g.snrs.push(m.snr_db);
        g.pws.push(m.duration_ns / 1e3);
        g.bws.push(m.bandwidth_hz);
        g.doas.push(m.direction_of_arrival_deg);
        g.toas.push(m.time_of_arrival_ns);
        if g.name.is_none() {
            g.name = m.emitter_label.clone();
        }
        if g.vector.is_none() {
            g.vector = emb.cloned();
        }
    }

    pub fn len(&self) -> usize {
        self.groups.len()
    }

    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    pub fn into_vec(self) -> Vec<(i64, Group)> {
        self.groups.into_iter().collect()
    }
}

impl Group {
    /// Emitter name, resolved the same way the classic path resolves it so
    /// the two pipelines write identical `emitters.name` values.
    ///
    /// The CSV `emitter_id` label comes first: the classic path maps the
    /// library id (1113, 1108…) through [`radar_name`] to the radar name
    /// (MF03, PSR5m1…). Labels the table does not know (the generator's
    /// unclassified / "plus" emitters) become `unknown_ai`, matching
    /// classic. Only when the CSV carries no label at all do we fall back
    /// to the model centre id.
    pub fn display_name(&self, id: i64) -> String {
        match self.name.as_deref() {
            Some(label) => crate::classic_classify::radar_name(label)
                .map(|n| n.to_string())
                .unwrap_or_else(|| "unknown_ai".to_string()),
            None => id.to_string(),
        }
    }

    pub fn medoid(&self) -> &[f32] {
        self.vector.as_deref().unwrap_or(&[])
    }
}

/// Split one batch into known-emitter predictions and the unknown rest.
///
/// Returns `(rest_embeddings, rest_meta, rest_input_ids)`. The known
/// pulses' stats are appended to `groups`.
pub fn split_batch(
    batch: &(Vec<Vec<f32>>, Vec<AiPulseMeta>, Vec<i64>),
    pred: &OnnxPredictions,
    threshold: f32,
    groups: &mut PredictionGroups,
) -> Option<(Vec<Vec<f32>>, Vec<AiPulseMeta>, Vec<i64>)> {
    if !pred.is_empty() && pred.len() != batch.1.len() {
        warn!(
            "onnx: prediction rows={} do not match batch pulses={} — clustering the batch as unknown",
            pred.len(),
            batch.1.len()
        );
    }
    let usable = pred.len() == batch.1.len();
    let (mut r_e, mut r_m, mut r_i) = (Vec::new(), Vec::new(), Vec::new());
    for (i, meta) in batch.1.iter().enumerate() {
        let emb = batch.0.get(i);
        if usable && pred.is_known(i, threshold) {
            let id = pred.labels[i];
            groups.push(id, meta, emb);
        } else if let Some(e) = emb {
            r_e.push(e.clone());
            r_m.push(meta.clone());
            r_i.push(batch.2.get(i).copied().unwrap_or(meta.id));
        }
    }
    if r_m.is_empty() {
        None
    } else {
        Some((r_e, r_m, r_i))
    }
}
