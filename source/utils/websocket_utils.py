import os
from fastapi import WebSocket


# TODO: Find a cleaner way to do
def is_connection_authorized(websocket: WebSocket) -> bool:
    """
    TODO: Find a cleaner way to check if in test
    here : 'websocket.headers.get("origin") == None' is the case test
    """
    if websocket.headers.get("origin") == None or websocket.headers.get(
        "origin"
    ) in str(os.environ.get("FRONTURLS")).split(" "):
        return True
    else:
        print("Unauthorized access => ", websocket.headers.get("origin"))
        return False
