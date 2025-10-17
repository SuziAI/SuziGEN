import pretty_midi

from music import relative_pitch_to_absolute_pitch, GongcheMelodySymbol, SuzipuSecondarySymbol


def realize_midi(lyrics, mode, relative_pitch, secondary, meter, filename):
    absolute_pitch = relative_pitch_to_absolute_pitch(mode, relative_pitch)
    pitch_number = [pretty_midi.note_name_to_number(GongcheMelodySymbol.to_pitch_name(p)) for p in absolute_pitch]

    xiao_sequence = []
    paiban_sequence = []
    for idx, (lyric, pitch, sec, met) in enumerate(zip(lyrics, pitch_number, secondary, meter)):
        pause = 0.0
        if met == "ju" or met == "pian":
            pause = 0.5
        elif met == "dou":
            pause = 0.25

        if sec == SuzipuSecondarySymbol.ADD_XIAO_ZHU:
            xiao_sequence.append((pitch, 2-pause, lyric))
            xiao_sequence.append(("REST", pause, ""))
            paiban_sequence.append(("REST", 1.0, ""))
            paiban_sequence.append((75, 1.0, ""))
        elif sec == SuzipuSecondarySymbol.ADD_DING_ZHU:
            xiao_sequence.append((pitch, 1.0, lyric))
            xiao_sequence.append((pitch, 1-pause, ""))
            xiao_sequence.append(("REST", pause, ""))
            paiban_sequence.append(("REST", 1.0, ""))
            paiban_sequence.append((75, 1.0, ""))
        elif sec == SuzipuSecondarySymbol.ADD_ZHE:
            xiao_sequence.append((pitch, 1.0, lyric))
            xiao_sequence.append((pitch+1, 1.0-pause, ""))
            xiao_sequence.append(("REST", pause, ""))
            paiban_sequence.append(("REST", 2.0, ""))
        elif sec == SuzipuSecondarySymbol.ADD_YE:
            xiao_sequence.append((pitch, 1.0, lyric))
            xiao_sequence.append((pitch+2, 1.0-pause, ""))
            xiao_sequence.append(("REST", pause, ""))
            paiban_sequence.append(("REST", 2.0, ""))
        elif sec == SuzipuSecondarySymbol.ADD_DA_ZHU:
            xiao_sequence.append((pitch, 2.5, lyric))
            xiao_sequence.append(("REST", 0.5, ""))
            paiban_sequence.append(("REST", 2.0, ""))
            paiban_sequence.append((75, 1.0, ""))
        elif sec == SuzipuSecondarySymbol.ADD_DA_DUN:
            xiao_sequence.append((pitch, 2.0, lyric))
            xiao_sequence.append(("REST", 1.0, ""))
            paiban_sequence.append(("REST", 2.0, ""))
            paiban_sequence.append((75, 1.0, ""))
        else:
            xiao_sequence.append((pitch, 1.-pause, lyric))
            xiao_sequence.append(("REST", pause, ""))
            paiban_sequence.append(("REST", 1.0, ""))

    xiao_sequence.append(("REST", 1.0, ""))
    paiban_sequence.append(("REST", 1.0, ""))

    guzheng_sequence = []
    for pitch, duration, lyric in xiao_sequence:
        if pitch == "REST":
            guzheng_sequence.append(("REST", duration, ""))
            continue
        elif isinstance(pitch, str):
            pitch_num = pretty_midi.note_name_to_number(pitch)
        else:
            pitch_num = pitch
        guzheng_sequence.append((pitch_num - 12, duration, ""))

    def make_instrument(program, name, is_drum=False):
        return pretty_midi.Instrument(program=program, name=name, is_drum=is_drum)

    def note_name_to_number(name):
        return pretty_midi.note_name_to_number(name)

    def add_notes(instrument, note_sequence, start_time=0.0):
        current_time = start_time
        lyrics = []
        for pitch, duration, lyric in note_sequence:
            if pitch == "REST":
                current_time += duration  # Skip ahead
                continue
            p_num = note_name_to_number(pitch) if isinstance(pitch, str) else pitch
            note = pretty_midi.Note(
                velocity=100, pitch=p_num, start=current_time, end=current_time + duration
            )

            lyric_bytes = lyric.encode("utf-8")
            l = pretty_midi.Lyric(text=lyric_bytes.decode("latin-1", "replace"), time=current_time)

            instrument.notes.append(note)
            lyrics.append(l)
            current_time += duration
        return lyrics


    # Pan Flute (Melody) — Program 75
    xiao = make_instrument(program=75, name="(represents Xiao)")
    # Harp (Arpeggios) — Program 46
    guzheng = make_instrument(program=46, name="(represents Guzheng)")
    # Claves (Percussion) — use MIDI percussion note 75 on channel 9 (drum channel)
    paiban = make_instrument(program=0, name="(represents Paiban)", is_drum=True)

    lyrics = add_notes(xiao, xiao_sequence)
    add_notes(guzheng, guzheng_sequence)
    add_notes(paiban, paiban_sequence)

    # Combine into one MIDI file
    midi = pretty_midi.PrettyMIDI()
    midi.instruments.extend([xiao, guzheng, paiban])
    midi.lyrics.extend(lyrics)
    midi.write(f"{filename}.mid")