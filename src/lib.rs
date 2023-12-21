pub mod csv_reader {
    use csv;
    use serde::Deserialize;
    use std::error::Error;

    pub fn read_csv_to_vec(filepath: &str) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        let mut rdr = csv::Reader::from_path(filepath)?;
        let mut data = Vec::new();

        for result in rdr.records() {
            let record = result?;
            let values: Vec<f32> = record
                .iter()
                .map(|field| field.parse::<f32>().unwrap())
                .collect();
            data.push(values);
        }

        Ok(data)
    }
}

