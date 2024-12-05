pub trait Training {
    async fn train(&mut self) -> Result<(), Box<dyn std::error::Error>>;
}
