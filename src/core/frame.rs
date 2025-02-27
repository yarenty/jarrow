use datafusion::prelude::DataFrame;

#[derive(Debug, Clone)]
pub struct Frame {
    data: Option<DataFrame>,
    statis: DataStatistics,
}

impl Default for Frame {
    fn default() -> Self {
        Frame {
            data: None,
            statis: DataStatistics::default(),
        }
    }
}


# [derive(Debug, Default, Clone)]
pub struct DataStatistics {
    // pub distribution: Distribution,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub null_count: Option<i64>,
    pub total_byte_size: Option<i64>,
    pub total_count: Option<i64>,
}