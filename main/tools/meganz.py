import os
import re
import sys
import json
import tqdm
import codecs
import random
import base64
import struct
import shutil
import requests
import tempfile

from Crypto.Cipher import AES
from Crypto.Util import Counter

sys.path.append(os.getcwd())

from main.app.variables import translations

def makebyte(x):
    """
    Encodes a string using Latin-1 to map it straight into raw byte buffers.

    Args:
        x (str): Input string to encode.

    Returns:
        bytes: Raw bytes encoded via Latin-1.
    """

    return codecs.latin_1_encode(x)[0]

def a32_to_str(a):
    """
    Converts an array of 32-bit unsigned integers into a big-endian byte string.

    Args:
        a (Union[List[int], Tuple[int, ...]]): Iterable collection of 32-bit integers.

    Returns:
        bytes: Big-endian formatted binary byte layout.
    """

    return struct.pack('>%dI' % len(a), *a)

def get_chunks(size):
    """
    Generates sequential file chunk boundaries based on MEGA's progressive streaming algorithm.

    Args:
        size (int): Total size of the file package payload.

    Yields:
        Generator[Tuple[int, int], None, None]: Tuples containing (chunk_start_offset, chunk_size).
    """

    p, s = 0, 0x20000 # Start offset, dynamic initial chunk chunking window (128KB)

    while p + s < size:
        yield(p, s)
        p += s
        # Incrementally expand buffer windows up to 1MB bounds to scale connection bandwidth
        if s < 0x100000: s += 0x20000

    yield (p, size - p)

def aes_cbc_decrypt(data, key):
    """
    Decrypts binary data using the AES-CBC block cipher protocol with a null Initialization Vector.

    Args:
        data (bytes): Ciphertext sequence block.
        key (bytes): Private symmetric encryption key payload.

    Returns:
        bytes: Decrypted plaintext payload byte array.
    """

    aes_cipher = AES.new(key, AES.MODE_CBC, makebyte('\0' * 16))

    return aes_cipher.decrypt(data)

def decrypt_attr(attr, key):
    """
    Decrypts and parses MEGA node serialization objects into standard attribute dictionaries.

    Args:
        attr (bytes): Raw obfuscated attributes block.
        key (Union[List[int], Tuple[int, ...]]): Extracted 32-bit symmetric decryption key parts.

    Returns:
        Union[Dict[str, Any], bool]: Decoded JSON attribute map if successful, else False.
    """

    attr = codecs.latin_1_decode(aes_cbc_decrypt(attr, a32_to_str(key)))[0].rstrip('\0')

    return json.loads(attr[4:]) if attr[:6] == 'MEGA{"' else False

def _api_request(data):
    """
    Dispatches a payload request packet directly to the MEGA decentralized CS gateway infrastructure.

    Args:
        data (Union[Dict[str, Any], List[Dict[str, Any]]]): Raw request command objects or list of actions.

    Returns:
        Dict[str, Any]: Parsed backend server confirmation message map.

    Raises:
        Exception: If the server passes back a negative application exception error integer.
    """

    sequence_num = random.randint(0, 0xFFFFFFFF)
    params = {'id': sequence_num}

    sequence_num += 1
    if not isinstance(data, list): data = [data]

    response = requests.post(
        '{0}://g.api.{1}/cs'.format('https', 'mega.co.nz'), 
        params=params, 
        data=json.dumps(data), 
        timeout=160
    )

    json_resp = json.loads(response.text)
    if isinstance(json_resp, int): raise Exception(json_resp)

    return json_resp[0]

def base64_url_decode(data):
    """
    Decodes non-standard variants of Base64 strings safely with padding restorations.

    Args:
        data (str): Unpadded, safe url token Base64 string.

    Returns:
        bytes: Resolved binary byte payload array.
    """

    data += '=='[(2 - len(data) * 3) % 4:]

    for search, replace in (('-', '+'), ('_', '/'), (',', '')):
        data = data.replace(search, replace)

    return base64.b64decode(data)

def str_to_a32(b):
    """
    Unpacks alphanumeric or byte strings out into groups of 32-bit unsigned integers.

    Args:
        b (Union[str, bytes]): Plain text or byte array sequence.

    Returns:
        Tuple[int, ...]: Unpacked Big-Endian structured 32-bit integer array.
    """

    if isinstance(b, str): b = makebyte(b)
    if len(b) % 4: b += b'\0' * (4 - len(b) % 4)

    return struct.unpack('>%dI' % (len(b) / 4), b)

def base64_to_a32(s):
    """
    Wraps URL-safe Base64 token translation directly into parsed 32-bit unsigned integer arrays.

    Args:
        s (str): Obfuscated source Base64 asset token string.

    Returns:
        Tuple[int, ...]: Transformed 32-bit data block maps.
    """

    return str_to_a32(
        base64_url_decode(s)
    )

def mega_download_file(file_handle, file_key, dest_path=None):
    """
    Downloads, decrypts, and verifies file entities mapped out inside the MEGA storage grids.

    Args:
        file_handle (str): Target unique resource node reference token identifier.
        file_key (str): base64 key parameter string linked to metadata node definitions.
        dest_path (str, optional): Safe destination folder reference path. Defaults to None.

    Returns:
        str: Absolute or localized target landing path of the decrypted asset file.

    Raises:
        Exception: If connection node parameters fail access authorizations.
        ValueError: If file verification calculations fail internal MAC parity validation checks.
    """

    # Parse out cryptographic descriptors from base64 arrays
    file_key = base64_to_a32(file_key)
    file_data = _api_request({'a': 'g', 'g': 1, 'p': file_handle})

    # Isolate sub-key transformations using XOR operations over the key array layout split
    k = (
        file_key[0] ^ file_key[4], 
        file_key[1] ^ file_key[5], 
        file_key[2] ^ file_key[6], 
        file_key[3] ^ file_key[7]
    )

    iv = file_key[4:6] + (0, 0)
    if 'g' not in file_data: raise Exception(translations["file_not_access"])

    file_size = file_data['s']
    attribs = decrypt_attr(base64_url_decode(file_data['at']), k)
    # Initialize standard multi-stream response handles to temporary cache endpoints
    input_file = requests.get(file_data['g'], stream=True).raw
    temp_output_file = tempfile.NamedTemporaryFile(mode='w+b', prefix='megapy_', delete=False)

    k_str = a32_to_str(k)
    aes = AES.new(
        k_str, 
        AES.MODE_CTR, 
        counter=Counter.new(
            128, 
            initial_value=(
                (iv[0] << 32) + iv[1]
            ) << 64
        )
    )

    # Prepare Message Authentication Code (MAC) verifying structures
    mac_str = b'\0' * 16
    mac_encryptor = AES.new(k_str, AES.MODE_CBC, mac_str)
    iv_str = a32_to_str([iv[0], iv[1], iv[0], iv[1]])

    # Read fragments sequentially to stream large items comfortably within RAM allocations
    with tqdm.tqdm(total=file_size, ncols=100, unit="byte") as pbar:
        for _, chunk_size in get_chunks(file_size):
            chunk = aes.decrypt(input_file.read(chunk_size))
            temp_output_file.write(chunk)

            pbar.update(len(chunk))
            encryptor = AES.new(k_str, AES.MODE_CBC, iv_str)

            for i in range(0, len(chunk) - 16, 16):
                block = chunk[i:i + 16]
                encryptor.encrypt(block)

            i = (i + 16) if file_size > 16 else 0
            block = chunk[i:i + 16]

            if len(block) % 16: block += b'\0' * (16 - (len(block) % 16))
            mac_str = mac_encryptor.encrypt(encryptor.encrypt(block))

    file_mac = str_to_a32(mac_str)
    temp_output_file.close()

    # Step 4: Validate computed MAC integrity signatures with data structure headers
    if (file_mac[0] ^ file_mac[1], file_mac[2] ^ file_mac[3]) != file_key[6:8]: raise ValueError(translations["mac_not_match"])

    # Finalize target directory bindings and execute clean cache swaps
    file_path = os.path.join(dest_path, attribs['n'])
    if os.path.exists(file_path): os.remove(file_path)

    shutil.move(temp_output_file.name, file_path)
    return file_path

def mega_download_url(url, dest_path=None):
    """
    Normalizes and routes public MEGA uniform resource locator parameters down to target extractors.

    Args:
        url (str): Target web link parameter addressing shared MEGA file instances.
        dest_path (str, optional): Target localized disk landing directory destination. Defaults to None.

    Returns:
        str: Absolute or relative output location referencing the downloaded file object.

    Raises:
        Exception: If URL format layout structure misses expected validation criteria.
    """

    if '/file/' in url:
        url = url.replace(' ', '')
        # Isolate node handle structures via standard boundaries regex matches
        file_id = re.findall(r'\W\w\w\w\w\w\w\w\w\W', url)[0][1:-1]
        path = f'{file_id}!{url[re.search(file_id, url).end() + 1:]}'.split('!')
    elif '!' in url: path = re.findall(r'/#!(.*)', url)[0].split('!')
    else: raise Exception(translations["missing_url"])

    return mega_download_file(path[0], path[1], dest_path)