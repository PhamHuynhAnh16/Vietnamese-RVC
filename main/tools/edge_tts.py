import re
import ssl
import json
import time
import uuid
import codecs
import certifi
import aiohttp

from io import TextIOWrapper
from dataclasses import dataclass
from contextlib import nullcontext
from xml.sax.saxutils import escape

@dataclass
class TTSConfig:
    def __init__(self, voice, rate, volume, pitch):
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.pitch = pitch

    @staticmethod
    def validate_string_param(param_name, param_value, pattern):
        if re.match(pattern, param_value) is None: raise ValueError(f"{param_name} '{param_value}'.")
        return param_value

    def __post_init__(self):
        match = re.match(r"^([a-z]{2,})-([A-Z]{2,})-(.+Neural)$", self.voice)
        if match is not None:
            region = match.group(2)
            name = match.group(3)

            if name.find("-") != -1:
                region = region + "-" + name[: name.find("-")]
                name = name[name.find("-") + 1 :]

            self.voice = ("Microsoft Server Speech Text to Speech Voice" + f" ({match.group(1)}-{region}, {name})")

        self.validate_string_param("voice", self.voice, r"^Microsoft Server Speech Text to Speech Voice \(.+,.+\)$")
        self.validate_string_param("rate", self.rate, r"^[+-]\d+%$")
        self.validate_string_param("volume", self.volume, r"^[+-]\d+%$")
        self.validate_string_param("pitch", self.pitch, r"^[+-]\d+Hz$")

def get_headers_and_data(data, header_length):
    headers = {}

    for line in data[:header_length].split(b"\r\n"):
        key, value = line.split(b":", 1)
        headers[key] = value

    return headers, data[header_length + 2 :]

def date_to_string():
    return time.strftime("%a %b %d %Y %H:%M:%S GMT+0000 (Coordinated Universal Time)", time.gmtime())

def mkssml(tc, escaped_text):
    if isinstance(escaped_text, bytes): escaped_text = escaped_text.decode("utf-8")
    return (f"<speak version='1.0' xmlns='{codecs.decode('uggc://jjj.j3.bet/2001/10/flagurfvf', 'rot13')}' xml:lang='en-US'>" f"<voice name='{tc.voice}'>" f"<prosody pitch='{tc.pitch}' rate='{tc.rate}' volume='{tc.volume}'>" f"{escaped_text}" "</prosody>" "</voice>" "</speak>")

def connect_id():
    return str(uuid.uuid4()).replace("-", "")

def ssml_headers_plus_data(request_id, timestamp, ssml):
    return (f"X-RequestId:{request_id}\r\n" "Content-Type:application/ssml+xml\r\n" f"X-Timestamp:{timestamp}Z\r\n"  "Path:ssml\r\n\r\n" f"{ssml}")

def remove_incompatible_characters(string):
    if isinstance(string, bytes): string = string.decode("utf-8")
    chars = list(string)

    for idx, char in enumerate(chars):
        code = ord(char)
        if (0 <= code <= 8) or (11 <= code <= 12) or (14 <= code <= 31): chars[idx] = " "

    return "".join(chars)

def split_text_by_byte_length(text, byte_length):
    if isinstance(text, str): text = text.encode("utf-8")
    if byte_length <= 0: raise ValueError("byte_length > 0")

    while len(text) > byte_length:
        split_at = text.rfind(b" ", 0, byte_length)
        split_at = split_at if split_at != -1 else byte_length

        while b"&" in text[:split_at]:
            ampersand_index = text.rindex(b"&", 0, split_at)
            if text.find(b";", ampersand_index, split_at) != -1: break

            split_at = ampersand_index - 1
            if split_at == 0: break

        new_text = text[:split_at].strip()

        if new_text: yield new_text
        if split_at == 0: split_at = 1

        text = text[split_at:]

    new_text = text.strip()
    if new_text: yield new_text

class Communicate:
    def __init__(self, text, voice, *, rate="+0%", volume="+0%", pitch="+0Hz", proxy=None, connect_timeout=10, receive_timeout=60):
        self.tts_config = TTSConfig(voice, rate, volume, pitch)
        self.texts = split_text_by_byte_length(escape(remove_incompatible_characters(text)), 2**16 - (len(ssml_headers_plus_data(connect_id(), date_to_string(), mkssml(self.tts_config, ""))) + 50))
        self.proxy = proxy
        self.session_timeout = aiohttp.ClientTimeout(total=None, connect=None, sock_connect=connect_timeout, sock_read=receive_timeout)
        self.state = {"partial_text": None, "offset_compensation": 0, "last_duration_offset": 0, "stream_was_called": False}

    def __parse_metadata(self, data):
        for meta_obj in json.loads(data)["Metadata"]:
            meta_type = meta_obj["Type"]
            if meta_type == "WordBoundary": return {"type": meta_type, "offset": (meta_obj["Data"]["Offset"] + self.state["offset_compensation"]), "duration": meta_obj["Data"]["Duration"], "text": meta_obj["Data"]["text"]["Text"]}
            if meta_type in ("SessionEnd",): continue

    async def __stream(self):
        async def send_command_request():
            await websocket.send_str(f"X-Timestamp:{date_to_string()}\r\n" "Content-Type:application/json; charset=utf-8\r\n" "Path:speech.config\r\n\r\n" '{"context":{"synthesis":{"audio":{"metadataoptions":{' '"sentenceBoundaryEnabled":false,"wordBoundaryEnabled":true},' '"outputFormat":"audio-24khz-48kbitrate-mono-mp3"' "}}}}\r\n")

        async def send_ssml_request():
            await websocket.send_str(ssml_headers_plus_data(connect_id(), date_to_string(), mkssml(self.tts_config, self.state["partial_text"])))

        audio_was_received = False
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())

        async with aiohttp.ClientSession(trust_env=True, timeout=self.session_timeout) as session, session.ws_connect(f"wss://speech.platform.bing.com/consumer/speech/synthesize/readaloud/edge/v1?TrustedClientToken=6A5AA1D4EAFF4E9FB37E23D68491D6F4&ConnectionId={connect_id()}", compress=15, proxy=self.proxy, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" " (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36" " Edg/130.0.0.0", "Accept-Encoding": "gzip, deflate, br", "Accept-Language": "en-US,en;q=0.9", "Pragma": "no-cache", "Cache-Control": "no-cache", "Origin": "chrome-extension://jdiccldimpdaibmpdkjnbmckianbfold"}, ssl=ssl_ctx) as websocket:
            await send_command_request()
            await send_ssml_request()

            async for received in websocket:
                if received.type == aiohttp.WSMsgType.TEXT:
                    encoded_data: bytes = received.data.encode("utf-8")
                    parameters, data = get_headers_and_data(encoded_data, encoded_data.find(b"\r\n\r\n"))
                    path = parameters.get(b"Path", None)

                    if path == b"audio.metadata":
                        parsed_metadata = self.__parse_metadata(data)
                        yield parsed_metadata
                        self.state["last_duration_offset"] = (parsed_metadata["offset"] + parsed_metadata["duration"])
                    elif path == b"turn.end":
                        self.state["offset_compensation"] = self.state["last_duration_offset"]
                        self.state["offset_compensation"] += 8_750_000
                        break
                elif received.type == aiohttp.WSMsgType.BINARY:
                    if len(received.data) < 2: raise Exception("received.data < 2")

                    header_length = int.from_bytes(received.data[:2], "big")
                    if header_length > len(received.data): raise Exception("header_length > received.data")

                    parameters, data = get_headers_and_data(received.data, header_length)
                    if parameters.get(b"Path") != b"audio": raise Exception("Path != audio")

                    content_type = parameters.get(b"Content-Type", None)
                    if content_type not in [b"audio/mpeg", None]: raise Exception("content_type != audio/mpeg")

                    if content_type is None:
                        if len(data) == 0: continue

                    if len(data) == 0: raise Exception("data = 0")
                    audio_was_received = True
                    yield {"type": "audio", "data": data}

            if not audio_was_received: raise Exception("!audio_was_received")

    async def stream(self):
        if self.state["stream_was_called"]: raise RuntimeError("stream_was_called")
        self.state["stream_was_called"] = True

        for self.state["partial_text"] in self.texts:
            async for message in self.__stream():
                yield message

    async def save(self, audio_fname, metadata_fname = None):
        metadata = (open(metadata_fname, "w", encoding="utf-8") if metadata_fname is not None else nullcontext())
        with metadata, open(audio_fname, "wb") as audio:
            async for message in self.stream():
                if message["type"] == "audio": audio.write(message["data"])
                elif (isinstance(metadata, TextIOWrapper) and message["type"] == "WordBoundary"):
                    json.dump(message, metadata)
                    metadata.write("\n")