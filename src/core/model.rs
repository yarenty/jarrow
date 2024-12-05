use arrow::array::RecordBatch;
use crate::core::frame::Frame;
use datafusion_expr::ColumnarValue;


#[derive(Debug)]
pub struct Model {
    pub params: ModelParams,
    pub train: Frame,
    pub valid: Frame,
    pub output: Frame,
    // pub scoring: Arc<dyn Scoring>,

    pub is_supervised: bool,
    // pub metrics: ModelMetrics,
}


impl Default for Model {
    fn default() -> Model {
        Model {
            params: ModelParams::default(),
            train: Frame::default(),
            valid: Frame::default(),
            output: Frame::default(),
            is_supervised: false,
        }
    }
}

pub const MAX_SUPPORTED_LEVELS: i32 = 1 << 20;


#[derive(Debug)]
pub struct ModelParams {
    pub algo: String, // name of algorithm
    pub family: String,
    pub dropping_tolerance: f64, // 1e-3,,
    pub n_folds: i32,
    pub keep_cross_validation_models: bool,
    pub keep_cross_validation_predictions: bool,
    pub response: String,
    pub offset: String,
}

impl Default for ModelParams {
    fn default() -> ModelParams {
        ModelParams {
            algo: String::new(),
            family: String::new(),
            dropping_tolerance: 1e-3,
            n_folds: 0,
            keep_cross_validation_models: true,
            keep_cross_validation_predictions: false,
            response: String::new(),
            offset: String::new(),
        }
    }
}

pub enum FoldAssignmentScheme {
    AUTO,
    Random,
    Modulo,
    Stratified,
}
pub enum CategoricalEncodingScheme {
    AUTO(bool),
    OneHotInternal(bool),
    OneHotExplicit(bool),
    Enum(bool),
    Binary(bool),
    Eigen(bool),
    LabelEncoder(bool),
    SortByResponse(bool),
    EnumLimited(bool),
}

pub trait Scoring {
    async fn score(&self, batch: Vec<RecordBatch>) -> ColumnarValue;
}

pub fn score_batch(
    _batch: Vec<RecordBatch>,
    _predictions: ColumnarValue) -> ColumnarValue {
    todo!()
}
