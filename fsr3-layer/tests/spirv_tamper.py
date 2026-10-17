"""SPIR-V edits for tests: change one member offset or binding, as a future X-Plane might."""
import struct

OP_MEMBER_DECORATE, OP_DECORATE, DEC_OFFSET, DEC_BINDING = 72, 71, 35, 33


def instructions(words):
    i = 5
    while i < len(words):
        n, op = words[i] >> 16, words[i] & 0xFFFF
        yield i, n, op
        i += n


def member_struct(words, block):
    """Struct id of the uniform block named `block` (OpName on the struct)."""
    for i, n, op in instructions(words):
        if op == 5 and struct.pack('<%dI' % (n - 2), *words[i + 2:i + n]).split(b'\0')[0].decode() == block:
            yield words[i + 1]


def tamper_offset(data, block, member_index, delta):
    words = list(struct.unpack('<%dI' % (len(data) // 4), data))
    ids = set(member_struct(words, block))
    for i, n, op in instructions(words):
        if op == OP_MEMBER_DECORATE and words[i + 1] in ids and words[i + 2] == member_index and words[i + 3] == DEC_OFFSET:
            words[i + 4] += delta
            return struct.pack('<%dI' % len(words), *words)
    raise SystemExit(f'tamper: no offset for {block}[{member_index}]')


def tamper_binding(data, old, new):
    words = list(struct.unpack('<%dI' % (len(data) // 4), data))
    for i, n, op in instructions(words):
        if op == OP_DECORATE and words[i + 2] == DEC_BINDING and words[i + 3] == old:
            words[i + 3] = new
            return struct.pack('<%dI' % len(words), *words)
    raise SystemExit(f'tamper: no binding {old}')
