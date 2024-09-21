from locust import HttpUser, task, between


class TranslationServiceUser(HttpUser):
    # Wait between 1 to 3 seconds between tasks
    wait_time = between(0, 1)

    # Example of test data to be sent
    test_data = {
        "text": "On devrait partir voir John chez ses parents.",
        "src_lang": "fra_Latn",
        "tgt_lang": "bam_Latn"
    }

    @task
    def translate(self):
        """
        Task to hit the translate endpoint with a POST request.
        """
        self.client.post("/translate", json=self.test_data)
