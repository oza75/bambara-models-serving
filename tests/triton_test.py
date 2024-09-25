from locust import HttpUser, task, between


class TranslationTritonServiceUser(HttpUser):
    # Wait between 1 to 3 seconds between tasks
    wait_time = between(1, 3)

    # Example of test data to be sent
    test_data = {
        "inputs": [
            {
                "name": "INPUTS",
                "shape": [
                    1,
                    3
                ],
                "datatype": "BYTES",
                "data": [
                    [
                        "On devrait partir voir John chez ses parents.",
                        "fra_Latn",
                        "bam_Latn"
                    ]]
            }
        ],
        "outputs": [
            {
                "name": "OUTPUT_TEXT"
            }
        ]
    }

    @task
    def translate(self):
        """
        Task to hit the translate endpoint with a POST request.
        """
        self.client.post(
            "/v2/models/translation/infer",
            json=self.test_data,
            headers={"Authorization": "Bearer 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"}
        )
