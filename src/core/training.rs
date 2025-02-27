use arrow::record_batch::RecordBatch;

pub trait Training {
    fn train(&mut self, batch: &Vec<RecordBatch>) -> impl std::future::Future<Output = Result<(), Box<dyn std::error::Error>>> + Send;
    // async fn train(&mut self, batch: &Vec<RecordBatch>) -> Result<(), Box<dyn std::error::Error>>;
}
