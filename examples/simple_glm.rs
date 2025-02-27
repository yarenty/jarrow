// /opt/workspace/jarrow/examples/simple_glm.rs
use datafusion::error::Result;
use datafusion::prelude::*;
use jarrow::glm;
use jarrow::core::training::Training;
use jarrow::core::model::Scoring;
use jarrow::core::frame::Frame;
use jarrow::core::model::{Model, ModelParams};
use jarrow::core::score::ScoreKeeper;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let ctx = SessionContext::new();

    // Simulate some training data
    let _ = ctx
        .register_csv("train_data", "./tests/data/train.csv", CsvReadOptions::new())
        .await;

    // Simulate some test data
    let _ = ctx
        .register_csv("test_data", "./tests/data/test.csv", CsvReadOptions::new())
        .await;

    // Create data frame from SQL
    let train_df = ctx
        .sql("select * from train_data")
        .await?;

    let test_df = ctx
        .sql("select * from test_data")
        .await?;

    // For Debugging.
    println!("Train data:");
    train_df.clone().show().await?;

    println!("Test data:");
    test_df.clone().show().await?;

    // Create ModelParams (adjust as needed for your GLM)
    let model_params = ModelParams {
        algo: "glm".to_string(),
        family: "gaussian".to_string(), // Example: Gaussian for regression
        dropping_tolerance: 1e-3,
        n_folds: 0,
        keep_cross_validation_models: true,
        keep_cross_validation_predictions: false,
        response: "y".to_string(), // Example: "y" as the response variable. change it based on data
        offset: "".to_string(),
    };

    // Create a default Model and assign parameters
    let mut model = Model::default();
    model.params = model_params;

    // Create a new GLMModel
    let mut glm_model = glm::glm_model::GLMModel::new(model);
     
    //Prepare frame from dataframe.
    let train_batch = train_df.collect().await?;
    let test_batch = test_df.collect().await?;
    
    let batch_to_train = vec![train_batch.get(0).unwrap().clone()];
    let batch_to_score = vec![test_batch.get(0).unwrap().clone()];

    // Train the model
    println!("Start training");
    // glm_model.train(&train_batch).await?;
    glm_model.train(&batch_to_train).await.unwrap();
    println!("Finish training");

    // Score the model
    let score_result = glm_model.score(test_batch.clone()).await;
    println!("Score result: {:?}", score_result);

    // Extract score keeper
    // let score_keeper: ScoreKeeper = glm_model.get_score_keeper();
    let score_keeper = glm_model.get_score_keeper();
    println!("Score keeper: {:?}", score_keeper);

    Ok(())
}
