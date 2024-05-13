import os
from fastapi import WebSocket


# TODO: Find a cleaner way to do
# TODO: Specify front prod url in .env.prod file
def is_connection_authorized(websocket: WebSocket) -> bool:
    if not (
        (
            os.environ.get("DEVMODE") == 1
            and websocket.headers.get("origin") == "http://localhost:3000"
        )
        or (
            websocket.headers.get("origin")
            == "https://s11-front-f89f0835e50e.herokuapp.com"
        )
    ):
        return True
    else:
        print("Unauthorized access => ", websocket.headers.get("origin"))
        return False
