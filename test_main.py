import unittest
from fastapi.testclient import TestClient
from main import app


# TODO: Also test response time !
class TestWebSocket(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.test_audio_file_location = "./test_data/AURORA-this_could_be_a_dream.mp3"

    # TODO: Refactor
    def test_websocket_complete_analysis(self):

        with open(self.test_audio_file_location, "rb") as audio_file:
            test_audio_file = audio_file.read()

            with self.client.websocket_connect("/ws/complete") as websocket:
                websocket.send_bytes(test_audio_file)
                data = websocket.receive_text()

                assert data == "Hiphop"

    # TODO: Refactor
    def test_websocket_fast_analysis(self):

        with open(self.test_audio_file_location, "rb") as audio_file:
            test_audio_file = audio_file.read()

            with self.client.websocket_connect("/ws/fast") as websocket:
                websocket.send_bytes(test_audio_file)
                data = websocket.receive_text()

                assert data == "Jazz"


if __name__ == "__main__":
    unittest.main(verbosity=2)
