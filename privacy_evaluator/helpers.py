import requests
import zipfile
import os


class GoogleDriveDownloader:
    """Minimal class to download shared files from Google Drive."""

    CHUNK_SIZE = 32768
    DOWNLOAD_URL = "https://docs.google.com/uc?export=download"

    @staticmethod
    def download_file_from_google_drive(
        file_id: str, output_path: str, unzip: bool = False
    ):
        """Downloads a shared file from google drive into a given folder. Optionally unzips it.

        :param file_id: The file identifier. It can be obtained from the sharable link.
        :param output_path: The output path where to save the downloaded file.
        :param unzip: If True unzips a file. If the file is not a zip file, ignores it.
        """
        output_directory = os.path.dirname(output_path)

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        if not os.path.exists(output_path):
            session = requests.Session()
            response = session.get(
                GoogleDriveDownloader.DOWNLOAD_URL, params={"id": file_id}, stream=True
            )
            token = GoogleDriveDownloader._get_confirm_token(response)

            if token:
                params = {"id": file_id, "confirm": token}
                response = session.get(
                    GoogleDriveDownloader.DOWNLOAD_URL, params=params, stream=True
                )

            GoogleDriveDownloader._save_response_content(response, output_path)

            if unzip:
                with zipfile.ZipFile(output_path, "r") as z:
                    z.extractall(output_directory)

    @staticmethod
    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    @staticmethod
    def _save_response_content(response, output_path):
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(GoogleDriveDownloader.CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
