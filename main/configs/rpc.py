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

def connect_discord_ipc():
    """
    Attempts to establish a connection to the local Discord desktop client 
    via Windows Named Pipes (IPC slot 0).

    Returns:
        Optional[BinaryIO]: A file-like stream object mapped to the pipe if successful, otherwise None.
    """

    try:
        # Open Discord's default local named pipe in read/write binary mode
        # Setting buffering=0 ensures unbuffered instantaneous transmission

        return open(
            r"\\?\pipe\discord-ipc-0", 
            "r+b", 
            buffering=0
        )
    except Exception:
        # Fallback if Discord isn't running or the pipe slot is busy/unavailable
        return None

def send_discord_rpc(pipe):
    """
    Executes the initial handshake authorization sequence and pushes a fresh 
    Rich Presence activity update to the connected Discord client stream.

    Args:
        pipe (BinaryIO): Active open binary stream linked to Discord's IPC.
    """

    # Send Opcode 0 (Handshake) along with our application client registration ID
    pipe.write(
        create_payload(
            0, {
                "v": 1, 
                "client_id": CLIENT_ID
            }
        )
    )

    # Read and discard the incoming response header (first 8 bytes contains Opcode & Length)
    pipe.read(8)
    # Dynamically extract and read the remaining body data based on incoming length
    # Note: original sequential code read header info inline to uncover payload size safely
    pipe.read(
        struct.unpack(
            "<I", 
            pipe.read(4)
        )[0]
    )

    # Send Opcode 1 (Frame) passing the SET_ACTIVITY command arguments
    pipe.write(
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