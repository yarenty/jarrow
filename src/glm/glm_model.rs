// /opt/workspace/jarrow/src/glm/glm_model.rs
use arrow::array::RecordBatch;
use datafusion::logical_expr::ColumnarValue;
use crate::core::model::{Model, Scoring, ModelParams};
use crate::core::training::Training;
use crate::core::score::ScoreKeeper;
use std::sync::Arc;
use arrow::datatypes::{Schema, Field, DataType};
use datafusion::common::Result;
use arrow::array::{Float64Array, Array, Int64Array, as_primitive_array};

#[derive(Debug)]
pub struct GLMModel {
    model: Model,
    score_keeper: ScoreKeeper,
}

impl GLMModel {
    pub fn new(mut model: Model) -> GLMModel {
        GLMModel {
            model: model.clone(),
            score_keeper: ScoreKeeper::from_model(&model.params),
        }
    }

    pub fn get_score_keeper(&self) -> ScoreKeeper {
        self.score_keeper.clone()
    }
}

impl Training for GLMModel {
    async fn train(&mut self, batch: &Vec<RecordBatch>)  -> Result<(), Box<dyn std::error::Error>> {
       println!("Training with {} batches", batch.len());

        // Dummy training logic.
        // Iterate through the data and build the model.
        // In a real implementation, this would involve computing GLM coefficients,
        // handling different distributions, etc.
        // Extract response variable and features from batches.
        for b in batch.iter() {
            // Assuming that the response variable is named "y"
            // You'll need to modify this to match the name of your response column.
            if let Some(y_col) = b.column_by_name(&self.model.params.response) {
                // println!("Response Column: {:?}", y_col);

                // Process the y column, e.g., convert it to a vector of floats.
                // ... (your logic here) ...
                let y_array = y_col.as_any().downcast_ref::<Float64Array>();
               if let Some(y_array) = y_array {
                   println!("Response Column Size: {:?}", y_array.len());
                   println!("Response Column Value: {:?}", y_array.value(0));
               }
                
                // Example: Print the first value from each column
               for i in 0..b.num_columns() {
                   let column = b.column(i);
                   if column.len() > 0 {
                       if let Some(float_array) = column.as_any().downcast_ref::<Float64Array>() {
                           if float_array.len() > 0 {
                               println!("Column {} Value: {:?}", i, float_array.value(0));
                           }
                       }

                       if let Some(int_array) = column.as_any().downcast_ref::<Int64Array>() {
                           if int_array.len() > 0 {
                               println!("Column {} Value: {:?}", i, int_array.value(0));
                           }
                       }
                       // Add more types if needed
                   }
               }
            } else {
                eprintln!("Response variable column not found.");
            }
        }


        // Update score_keeper
        self.score_keeper.mse = Some(1.0);
        self.score_keeper.rmse = Some(1.0);
        self.score_keeper.mae = Some(0.5);

        Ok(())
    }
}

impl Scoring for GLMModel {
    async fn score(&self, batch: Vec<RecordBatch>) -> ColumnarValue {
        println!("Scoring with {} batches", batch.len());
        // Dummy scoring logic.
        // In a real implementation, this would involve applying the trained
        let num_rows = batch.iter().map(|b| b.num_rows()).sum::<usize>();

        // Create array with data size
        let data_array = vec![0.0; num_rows];
        let array: Arc<dyn Array> = Arc::new(Float64Array::from(data_array));

        // model to the input data and generating predictions.

        // Example: create a dummy array of predictions

        // let schema = Schema::new(vec![Field::new("prediction", DataType::Float64, false)]);
        // let array: Arc<dyn Array> = Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0,4.0]));

        ColumnarValue::Array(array)
    }
}
