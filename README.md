# Amazon Reviews Sentiment-Based Price Optimization

This project focuses on dynamically optimizing product prices based on sentiment analysis of customer reviews. The model combines sentiment analysis with reinforcement learning to adjust prices in a way that maximizes revenue while maintaining customer satisfaction.

## Technologies Used

-   **Apache Spark**: The base engine for large-scale data processing.
-   **GCP Dataproc Clusters**: Used for processing data at a large scale on Google Cloud Platform.
-   **Databricks Workspace**: Used as an alternative environment for running the model.

## Dataset

We used the **Amazon Reviews Dataset (2023)**, available on Hugging Face, which includes customer reviews for a variety of products. The dataset provides information such as:

-   **asin**: Unique identifier for each product.
-   **rating**: Customer ratings (1-5).
-   **text**: Review text.
-   **timestamp**: Unix timestamp for the review.
-   **verified_purchase**: Indicates if the review is from a verified purchase.

You can find the dataset here: [Amazon Reviews Dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

## Documentation

-   **Apache Spark Documentation**: [Link](https://spark.apache.org/docs/latest/)
-   **GCP Documentation**: [Link](https://cloud.google.com/docs)

## Running the Code

### For Databricks Workspace

1. **Upload the Tables**:

    - First, upload the `All_Beauty.jsonl` and `meta_All_Beauty.jsonl` files to the `/FileStore/tables` directory in the Databricks.

2. **Import the Notebook**:

    - Import the `Review-based-price-optimization.ipynb` file into your Databricks workspace.

3. **Attach a Cluster**:
    - Attach a cluster to your notebook and run the code.

### For GCP Dataproc Cluster

1. **Set Up GCP Bucket**:

    - Set up a GCP bucket using the GCP Console and create two folders inside it: `/data` and `/scripts`.

2. **Upload Files**:

    - Upload the `install_text_blob.sh` file to the `/scripts` folder in the bucket using the following GCP command:

        ```bash
        gsutil cp install_textblob.sh gs://<your-bucket-name>/scripts/install_textblob.sh
        ```

    - Also upload your dataset (e.g., `Clothing_Shoes_and_Jewelry.jsonl`) to the `/data` folder using the following GCP command:

        ```bash
        gsutil cp /<file-location>/Clothing_Shoes_and_Jewelry.jsonl gs://<your-bucket-name>/data/
        ```

3. **Set Up Dataproc Cluster**:

    - Set up the GCP Dataproc cluster using GCP Console or the following GCP command, and make sure to use `install_text_blob.sh` as an initialization action.

        ```bash
        ./google-cloud-sdk/bin/gcloud dataproc clusters create <your-cluster-name> --enable-component-gateway --bucket <your-bucket-name> \
        --region us-central1 --master-machine-type n2-standard-2 --master-boot-disk-type pd-balanced \
        --initialization-actions=gs://<your-bucket-name>/scripts/install_textblob.sh \
        --master-boot-disk-size 32 --num-workers 2 --worker-machine-type n2-standard-2 --worker-boot-disk-type pd-balanced \
        --worker-boot-disk-size 32 --image-version 2.2-debian12 --project <your-project-name>
        ```

4. **Export Environment Variables**:

    - Once the cluster is running, SSH into the Master node and export the following environment variables:

        ```bash
        export DATA_BUCKET="gs://<your-bucket-name>/data"
        export file_location="Clothing_Shoes_and_Jewelry.jsonl"
        export meta_file_location="meta_Clothing_Shoes_and_Jewelry.jsonl"
        ```

5. **Upload the Python Script**:

    - Upload the `Review-based-price-optimization.py` file to the Master node.

6. **Run the Spark Job**:

    - Use the following `spark-submit` command to start the execution of the script:

        ```bash
        spark-submit Review-based-price-optimization.py \
            --cluster=my-dataproc-cluster \
            --region=us-central1 \
            --properties=DATA_BUCKET=gs://<your-bucket-name>,DATA_LOCATION=us-central1
        ```

7. **Observe the Results**:
    - Once the job has completed, observe the output and results from the model. The results will show the agent's actions for price adjustments and the corresponding rewards based on sentiment and ratings.

## Contributing

We welcome contributions! If you have ideas to improve the model or encounter any issues, feel free to fork the repository and submit a pull request.

## License

This project is open source and available under the [MIT License](LICENSE).
