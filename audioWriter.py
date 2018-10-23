import wave, struct, math


def write_audio(path, PCM):
    sampleRate = 44100.0  # hertz
    wavef = wave.open(path, 'w')
    wavef.setnchannels(1)  # mono
    wavef.setsampwidth(2)
    wavef.setframerate(sampleRate)

    for i in PCM:
        value = int(32767 * i)  # 2 bytes, 16 bit
        data = struct.pack('<h', value)
        wavef.writeframesraw(data)

    wavef.writeframes('')
    wavef.close()
