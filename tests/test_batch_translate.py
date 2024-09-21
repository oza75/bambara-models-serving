import aiohttp
import asyncio
import json

# API endpoint for translation
API_URL = 'https://translation-service-65pi-ffdf0c78.mt-guc1.bentoml.ai'

# Define the request data
requests_data = [
    {
        "text": "On devrait partir voir John chez ses parents.",
        "src_lang": "fra_Latn",
        "tgt_lang": "bam_Latn"
    },
    {
        "text": "Bonjour, comment ça va?",
        "src_lang": "fra_Latn",
        "tgt_lang": "bam_Latn"
    },
    {
        "text": "We should go to John's parents' house.",
        "src_lang": "eng_Latn",
        "tgt_lang": "bam_Latn"
    },
    {
        "text": "Je veux manger un sandwich.",
        "src_lang": "fra_Latn",
        "tgt_lang": "bam_Latn"
    },
    {
        "text": "Je veux manger un sandwich.",
        "src_lang": "fra_Latn",
        "tgt_lang": "bam_Latn"
    },
    {
        "text": "It is raining heavily today.",
        "src_lang": "eng_Latn",
        "tgt_lang": "bam_Latn"
    }
]


async def send_request(session, data):
    """
    Send a POST request to the translation API with the provided data asynchronously.
    Returns a tuple containing the input text, translated text, and language info.
    """
    headers = {
        'accept': 'text/plain',
        'Content-Type': 'application/json'
    }

    async with session.post(API_URL, headers=headers, data=json.dumps(data)) as response:
        if response.status == 200:
            result = await response.json()
            return data['text'], result, data['src_lang'], data['tgt_lang']
        else:
            return data['text'], "Error occurred", data['src_lang'], data['tgt_lang']


async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, req) for req in requests_data]
        responses = await asyncio.gather(*tasks)

        # Process the results
        for input_text, translated_text, src_lang, tgt_lang in responses:
            # Print the result in the required format
            print(
                f"---\n- Input text: {input_text}\n- Translated: {translated_text}\n- Lang: {src_lang} -> {tgt_lang}\n")


# Run the asynchronous tasks
if __name__ == "__main__":
    asyncio.run(main())
