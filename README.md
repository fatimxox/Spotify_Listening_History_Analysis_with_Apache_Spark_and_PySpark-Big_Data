
## Prerequisites

To run this project, you need to have the following installed:

*   **Python:** Version 3.6 or higher is recommended.
*   **Apache Spark:** Ensure Spark is installed and configured on your system. You can download it from the [official Apache Spark website](https://spark.apache.org/downloads.html).
*   **Java:** Spark requires a compatible Java version (usually Java 8 or Java 11).
*   **Jupyter Notebook:** To run the `.ipynb` file. Install via `pip install notebook`.
*   **Python Libraries:** Install the necessary Python libraries using pip:
    ```bash
    pip install pyspark pandas matplotlib findspark
    ```
    *   `pyspark`: The Python API for Spark.
    *   `pandas`: Used for converting Spark DataFrames to smaller Pandas DataFrames for visualization.
    *   `matplotlib`: Used for plotting the visualizations.
    *   `findspark`: Helps PySpark locate Spark installation.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
    cd YOUR_REPOSITORY_NAME](https://github.com/fatimxox/Spotify_Listening_History_Analysis_with_Apache_Spark_and_PySpark-Big_Data)
    ```
    
2.  **Install dependencies:** Ensure you have met the [Prerequisites](#prerequisites) and installed the required Python libraries.

## Data

This project requires your personal Spotify streaming history data in a CSV format.

1.  **Obtain your data:** Request your data from Spotify through their privacy settings page (usually found under your Account -> Privacy settings -> Download your data). You'll typically receive a zip file containing several JSON files. For this project, you'll need the streaming history data, which is often provided in `StreamingHistoryX.json` or similar files within the archive.
2.  **Convert to CSV:** The provided notebook assumes a CSV file named `spotify_history.csv`. You will need to convert your relevant JSON streaming history file(s) into a single CSV file with the expected columns (`ts`, `platform`, `ms_played`, `track_name`, `artist_name`, `album_name`, `reason_start`, `reason_end`, `shuffle`, `skipped`, `spotify_track_uri`). You might need a separate script or online tool to perform this conversion and aggregation if your data is split across multiple JSON files.
3.  **Place the data file:** Save your `spotify_history.csv` file in the root directory of the cloned repository.

**Note:** The structure of the Spotify data might change over time. The provided notebook is based on a specific format and might require adjustments for newer data exports.

## Usage

1.  **Start Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
2.  **Open `big_data.ipynb`:** Navigate to the repository directory in the Jupyter file browser and open the `big_data.ipynb` file.
3.  **Run the Notebook:** Execute the cells sequentially from top to bottom. The notebook contains comments explaining each step of the process.

## ETL and Analysis Steps (Notebook Walkthrough)

The `big_data.ipynb` notebook follows a structured approach:

1.  **Spark Initialization:** Sets up the SparkSession for local execution (`local[*]`).
2.  **Library Imports & Logging:** Imports necessary libraries and configures basic logging.
3.  **Data Loading:** Reads the `spotify_history.csv` file into a Spark DataFrame. Includes basic error handling.
4.  **Basic Inspection & Repartitioning:** Shows the first few rows and the schema. The DataFrame is repartitioned for potentially better performance, though this might not be necessary or optimal for very small local datasets.
5.  **Data Cleaning:**
    *   Drops rows with nulls in essential columns (`track_name`, `artist_name`, `ms_played`, `ts`).
    *   Removes duplicate rows.
    *   Trims leading/trailing whitespace from string columns (`track_name`, `artist_name`, `album_name`, `platform`).
6.  **Data Transformation:**
    *   Converts the `ts` (timestamp) column to a proper TimestampType.
    *   Calculates `minutes_played` by converting `ms_played` from milliseconds to minutes.
    *   Creates a temporary Spark view `spotify_cleaned` for potential SQL queries (though not explicitly used in the rest of this notebook, it's a common ETL step).
7.  **Preview Cleaned Data:** Displays the first few rows of the processed DataFrame including the new columns.
8.  **Platform Usage Analysis:** Groups data by `platform` and counts the number of plays, ordered by count. A bar chart is generated using Matplotlib (converted to Pandas first).
9.  **Skip Rate Analysis:** Groups data by `shuffle` mode and calculates the total plays and skipped plays to determine the skip rate for each shuffle mode. A combined bar/line chart visualizes this.
10. **Weekly Listening Patterns:** Extracts the day of the week from the timestamp and calculates the total minutes listened for each day. A line plot shows trends across the week.
11. **Top Artists by Play Count:** Groups data by `artist_name` and counts total plays, showing the top 10 in a bar chart.
12. **Top Artists by Listening Time:** Groups data by `artist_name` and sums `minutes_played`, showing the top 10 in a bar chart.
13. **Top Albums by Listening Time:** Groups data by `album_name` and `artist_name` and sums `minutes_played`, showing the top 10 in a bar chart.
14. **Top Songs by Play Count:** Groups data by `track_name` and `artist_name` and counts total plays, showing the top 10 in a bar chart.

## Recommendation System (Collaborative Filtering)

The notebook includes a basic example of building a recommendation system:

1.  **Data Preparation for ALS:**
    *   Uses `StringIndexer` from Spark MLlib to convert `artist_name` and `track_name` into numerical `artist_id` and `track_id`. This is required by the ALS algorithm.
    *   Selects the `artist_id`, `track_id`, and `minutes_played` columns, which represent user, item, and rating (or interaction strength).
2.  **Data Splitting:** Splits the prepared data into training (80%) and test (20%) sets.
3.  **ALS Model Training:**
    *   Initializes an `ALS` model instance.
    *   Configures parameters like `maxIter` (maximum iterations), `regParam` (regularization parameter to prevent overfitting), `userCol`, `itemCol`, and `ratingCol`.
    *   Sets `coldStartStrategy` to "drop" to handle cases where the model might try to predict for users/items not seen during training.
    *   Trains the model using the `training` dataset.
4.  **Model Evaluation:** Evaluates the trained model's performance on the `test` dataset using `RegressionEvaluator` with the Root Mean Squared Error (RMSE) metric. A lower RMSE indicates better prediction accuracy.
5.  **Generate Recommendations:** Demonstrates how to generate top N recommendations for a specific user (identified by `artist_id`). The output currently provides the recommended `track_id` and the predicted rating (`minutes_played`).

**Note:** To make the recommendations more user-friendly, you would typically join the recommended `track_id` back to the original data to get the `track_name` and `artist_name`. This step is outlined as a future improvement.

## Evaluation

The notebook calculates the RMSE of the ALS model on the test data. This metric provides a measure of how close the model's predicted listening times are to the actual listening times in the test set. A lower RMSE generally indicates a better-performing model for this specific task.

## Future Improvements

*   **User-Friendly Recommendations:** Implement mapping recommended `track_id`s back to actual song titles and artist names.
*   **Hyperparameter Tuning:** Experiment with different `maxIter`, `regParam`, and other ALS parameters to potentially improve the model's performance (e.g., using Cross-Validation).
*   **Alternative Recommendation Metrics:** Explore other evaluation metrics beyond RMSE, such as Mean Absolute Error (MAE) or ranking metrics like precision and recall (requires adapting the evaluator).
*   **Different Recommendation Algorithms:** Investigate other recommendation techniques available in Spark MLlib or other libraries (e.g., Weighted Alternating Least Squares for implicit feedback data).
*   **Genre Analysis:** Extract or infer genre information for tracks and analyze listening patterns by genre.
*   **Temporal Analysis:** Analyze listening patterns at finer granularities (e.g., time of day on specific days).
*   **Scalability:** Adapt the Spark configuration for running on a cluster (e.g., Hadoop YARN, Kubernetes) for processing much larger datasets.
*   **Interactive Dashboard:** Build a dashboard (e.g., using Dash, Streamlit, or Spark's built-in UI) to visualize insights and recommendations interactively.

## Contributing

Contributions are welcome! If you find issues or have ideas for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

---
