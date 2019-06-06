from arus.core.stream import ActigraphFileStream


if __name__ == "__main__":
    stream = ActigraphFileStream(
        obj_toload='/mnt/d/data/spades-2day/SPADES_12/OriginalRaw/Spades_12_dominant_ankle (2015-12-11)RAW.csv', sr=80, name='SPADES_12_DA')
    stream.start()
    for chunk in stream.get_iterator():
        print(chunk.iloc[:, 1].mean())
