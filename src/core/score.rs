use arrow::array::RecordBatch;
use datafusion::logical_expr::ColumnarValue;
use crate::core::model::{Model, ModelParams};


#[derive(Default, Debug)]
pub struct ScoreInfo {
    pub time_stamp_ms: i64, //absolute time the model metrics were computed
    pub total_training_time_ms: i64, //total training time until this scoring event (including checkpoints)
    pub total_scoring_time_ms: i64, //total scoring time until this scoring event (including checkpoints)
    pub total_setup_time_ms: i64, //total setup time until this scoring event (including checkpoints)
    pub this_scoring_time_ms: i64, //scoring time for this scoring event (only)
    pub is_classification: bool,
    pub is_autoencoder: bool,
    pub validation: bool,
    pub cross_validation: bool,

    pub scored_train: ScoreKeeper,
    pub scored_valid: ScoreKeeper,
    pub scored_xval: ScoreKeeper,
}


#[derive(Default, Debug, Clone)]
pub struct ScoreKeeper {
    pub mean_residual_deviance: Option<f64>,
    pub mse: Option<f64>,
    pub rmse: Option<f64>,
    pub mae: Option<f64>,
    pub rmsle: Option<f64>,
    pub logloss: Option<f64>,
    pub auc: Option<f64>,
    pub pr_auc: Option<f64>,
    pub class_error: Option<f64>,
    pub mean_per_class_error: Option<f64>,
    pub custom_metric: Option<f64>,
    pub hit_ratio: Option<Vec<f64>>,
    pub lift: Option<f64>, //Lift in top group
    pub r2: Option<f64>,
    pub anomaly_score: Option<f64>,
    pub anomaly_score_normalized: Option<f64>,
    pub auuc: Option<f64>,
    pub auuc_normalized: Option<f64>,
    pub qini: Option<f64>,
    pub auuc_nbins: i32,
    pub ate: Option<f64>,
    pub att: Option<f64>,
    pub atc: Option<f64>,
}
//
// impl Default for ScoreKeeper {
//     fn default() -> Self {
//         ScoreKeeper {
//             mean_residual_deviance: None,
//             mse: None,
//             rmse: None,
//             mae: None,
//             rmsle: None,
//             logloss: None,
//             auc: None,
//             pr_auc: None,
//             class_error: None,
//             mean_per_class_error: None,
//             custom_metric: None,
//             hit_ratio: None,
//             lift: None, //Lift in top group
//             r2: None,
//             anomaly_score: None,
//             anomaly_score_normalized: None,
//             auuc: None,
//             auuc_normalized: None,
//             qini: None,
//             auuc_nbins: 0,
//             ate: None,
//             att: None,
//             atc: None,
//         }
//     }
// }

impl ScoreKeeper {
  pub fn from_model(_model_params: &ModelParams) -> ScoreKeeper {
              ScoreKeeper::default()
  }
}

