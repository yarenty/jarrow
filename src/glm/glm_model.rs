use arrow::array::RecordBatch;
use datafusion::logical_expr::ColumnarValue;
use crate::core::model::{Model, Scoring};
use crate::core::training::Training;

#[derive(Debug, Default)]
pub struct GLMModel {
    model: Model,
}




impl GLMModel {
    pub fn new() -> GLMModel {
        GLMModel {
            model: Model::default(),
        }
    }
}

impl Training for GLMModel {
    async fn train(&mut self)  -> Result<(), Box<dyn std::error::Error>> {
        todo!()
    }
}

impl Scoring for GLMModel {
    async fn score(&self, batch: Vec<RecordBatch>) -> ColumnarValue {
        todo!()
    }
}
