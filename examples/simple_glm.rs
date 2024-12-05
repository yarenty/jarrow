use datafusion::error::Result;
use datafusion::prelude::*;
use jarrow::glm;
use jarrow::core::training::Training;
use jarrow::core::model::Scoring;
#[tokio::main]
async fn main() -> Result<()> {
    let ctx = SessionContext::new();

    let _ = ctx
        .register_csv("logs", "./tests/data/test.csv", CsvReadOptions::new())
        .await;

    let train = ctx
        .sql("select * from logs where level='INFO' limit 2")
        .await?;

    train.show().await?;

    let test = ctx
        .sql("select * from logs where level='INFO' limit 2")
        .await?;

    test.clone().show().await?;

    let mut glm = glm::glm_model::GLMModel::new();

    let model = glm.train().await.unwrap();

    let b = test.collect().await?;

    let score = glm.score(b).await;

    println!("score: {:?}", score);

    Ok(())
}
