import os
import sys
import json
import time
import struct
import codecs

sys.path.append(os.getcwd())

from main.app.variables import translations

CLIENT_ID = "1392816674159202396"

def create_payload(opcode, payload):
    """
    Encodes a payload dictionary into JSON and packs it into a binary frame 
    following the Discord IPC protocol specification.

    Args:
        opcode (int): The operation code (e.g., 0 for Handshake, 1 for Frame).
        payload (dict): The dictionary containing data/command arguments.

    Returns:
        bytes: The fully structured binary payload ready to be sent over the pipe.
    """

    # Serialize the payload dictionary into a UTF-8 encoded bytes sequence
    data = json.dumps(payload).encode("utf-8")

    # Structure: 
    # - "<I": Little-endian 32-bit unsigned integer for Opcode
    # - "<I": Little-endian 32-bit unsigned integer for Payload Length
    # - data: The actual JSON body data trailing right after
    return struct.pack(
        "<I", 
        opcode
    ) + struct.pack(
        "<I", 
        len(data)
    ) + data

def get_discord_ipc_path():
    """
    Locates the active Discord IPC socket path on Linux/macOS.

    Returns:
        Optional[str]: Full path to the active IPC socket, or None if not found.
    """

    env_vars = ["XDG_RUNTIME_DIR", "TMPDIR", "TMP", "TEMP"]
    base_dirs = [os.environ.get(var) for var in env_vars if os.environ.get(var)]
    base_dirs.append("/tmp")

    for base_dir in base_dirs:
        for i in range(10):
            path = os.path.join(base_dir, f"discord-ipc-{i}")
            if os.path.exists(path): return path

            app_path = os.path.join(base_dir, "app", "com.discordapp.Discord", f"discord-ipc-{i}")
            if os.path.exists(app_path): return app_path

    return None

def connect_discord_ipc():
    """
    Attempts to establish a connection to the local Discord desktop client 
    via Windows Named Pipes or Linux Unix Domain Sockets automatically based on OS.

    Returns:
        Optional[Union[BinaryIO, socket.socket]]: Connected pipe/socket object if successful, otherwise None.
    """

    if sys.platform == "win32":
        for i in range(10):
            try:
                return open(f"\\\\?\\pipe\\discord-ipc-{i}", "r+b", buffering=0)
            except Exception:
                continue

        return None

    import socket

    ipc_path = get_discord_ipc_path()
    if not ipc_path: return None

    try:
        client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client_socket.connect(ipc_path)
        return client_socket
    except Exception:
        return None

def send_discord_rpc(pipe):
    """
    Executes the initial handshake authorization sequence and pushes a fresh 
    Rich Presence activity update to the connected Discord client stream.

    Args:
        pipe (Union[BinaryIO, socket.socket]): Active open binary stream or socket linked to Discord's IPC.
    """

    def write_data(data):
        pipe.write(data) if hasattr(pipe, "write") else pipe.sendall(data)

    def read_data(length):
        return pipe.read(length) if hasattr(pipe, "read") else pipe.recv(length)

    # Send Opcode 0 (Handshake) along with our application client registration ID
    write_data(
        create_payload(
            0, {
                "v": 1, 
                "client_id": CLIENT_ID
            }
        )
    )

    # Read and process incoming response header (first 8 bytes: Opcode & Length)
    header = read_data(8)
    if len(header) == 8:
        _, length = struct.unpack("<II", header)
        if length > 0: read_data(length)

    # Send Opcode 1 (Frame) passing the SET_ACTIVITY command arguments
    write_data(
        create_payload(
            1, {
                "cmd": "SET_ACTIVITY",
                "args": {
                    "pid": os.getpid(), # Pass current process ID to bind lifecycle
                    "activity": {
                        "buttons": [{
                            "label": "Github", 
                            # Decode hidden project URL using ROT13 cipher variant
                            "url": codecs.decode(
                                "uggcf://tvguho.pbz/CunzUhlauNau16/Ivrganzrfr-EIP", 
                                "rot13"
                            )
                        }],
                        # Pull localized details and usage status text strings
                        "details": translations["details"],
                        "timestamps": {
                            "start": int(
                                time.time() # Sets the running elapsed activity timer
                            )
                        },
                        "state": translations["use"]
                    }
                },
                "nonce": str(
                    time.time() # Unique tracking transaction marker required by RPC
                )
            }
        )
    )