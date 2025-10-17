import argparse
import dataclasses
import chinese_converter
import text_resources
from filter_and_conversion import filter_by_mode, filter_by_final

from midi import realize_midi
from mode import generate_mode
from music import absolute_pitch_to_function, relative_pitch_to_absolute_pitch, GongdiaoModeList
from pitch_and_secondary import generate_pitch, generate_secondary
from repetitions import generate_repetition, remove_repetition_for_secondary
from save_and_load import load_17_pieces_data, load_probabilities, cipai_string_to_properties
from text_resources import EnglishTexts


def lyrics_string_to_list(lyrics_string: str):
    punctuation = (":;!?<>=！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝"
                   "～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.")

    lyrics_string = "".join(lyrics_string.split())  # remove white space characters

    if lyrics_string[0] in punctuation:
        raise ValueError(
            f"Lyrics string has punctuation mark '{lyrics_string[0]}' as first character. This is invalid."
        )

    lyrics_list = []
    for character in lyrics_string:
        if character in punctuation:
            lyrics_list[-1] += character
        else:
            lyrics_list.append(character)

    return lyrics_list


def generate(lyrics, mode, cipai, pieces, probabilities, filename):
    pitch_contour_distributions = probabilities["pitch_contour_distributions"]
    pitch_initial_state_distributions = probabilities["pitch_initial_state_distributions"]
    pitch_function_distributions = probabilities["pitch_function_distributions"]
    pitch_transition_probabilities = probabilities["pitch_transition_probabilities"]

    secondary_group_distributions = probabilities["secondary_group_distributions"]
    secondary_zhe_ye_distributions = probabilities["secondary_zhe_ye_distributions"]
    secondary_initial_state_distributions = probabilities["secondary_initial_state_distributions"]
    secondary_transition_probabilities = probabilities["secondary_transition_probabilities"]

    total_probability = 1.

    description_string = "\n*SUZIGEN GENERATION REPORT*\n\n"
    pitch_repetition = generate_repetition(cipai)
    description_string += pitch_repetition["description"] + "\n\n"
    #total_probability *= repetition["probability"] ## maybe TODO?

    secondary_repetition = {"repetition": remove_repetition_for_secondary(cipai, pitch_repetition["repetition"])}

    if mode is None:
        mode = generate_mode(pieces)
        description_string += mode["description"] + "\n\n"
        total_probability *= mode["probability"]
    else:
        description_string += (EnglishTexts.mode_already_given.format(
            chinese_name=mode["chinese_name"],
            name=mode["name"]
        )) + " "
        real_pieces_with_same_mode = filter_by_mode(pieces["all"], mode_properties = {"final_note": mode["mfinal"], "gong_lvlv": mode["mgong"]})
        real_pieces_with_same_final = filter_by_final(pieces["all"], mode["mfinal"])
        if len(real_pieces_with_same_mode):
            piece_list_string = "".join(
                f"{piece['title']} ({piece['latinized_title']}), " for piece in real_pieces_with_same_mode)
            piece_list_string = piece_list_string[:-2]  # remove last comma
            description_string += EnglishTexts.mode_in_baishi.format(piece_list=piece_list_string)
        else:
            description_string += EnglishTexts.mode_not_in_baishi
        final_note_string = "".join(
            f"{piece['title']} ({piece['latinized_title']}), " for piece in real_pieces_with_same_final)
        final_note_string = final_note_string[:-2]  # remove last comma
        description_string += EnglishTexts.mode_final_note.format(final_note=mode["mfinal"], final_note_list=final_note_string)
        description_string += "\n\n"

    pitch = generate_pitch(
        initial_state_distributions=pitch_initial_state_distributions,
        contour_distributions=pitch_contour_distributions,
        function_distributions=pitch_function_distributions,
        pitch_transition_probabilities=pitch_transition_probabilities,
        cipai=cipai,
        mode=mode,
        repetition=pitch_repetition
    )
    description_string += pitch["description"] + "\n\n"
    total_probability *= pitch["probability"]

    absolute_pitch = relative_pitch_to_absolute_pitch({"gong_lvlv": mode["mgong"]}, pitch["pitch_list"])
    pitch_functions = absolute_pitch_to_function({"gong_lvlv": mode["mgong"]}, absolute_pitch)

    secondary = generate_secondary(
        initial_state_distributions=secondary_initial_state_distributions,
        zhe_ye_distributions = secondary_zhe_ye_distributions,
        secondary_group_distributions = secondary_group_distributions,
        secondary_transition_probabilities=secondary_transition_probabilities,
        cipai=cipai,
        repetition=secondary_repetition,
        pitch_functions=pitch_functions
    )
    description_string += secondary["description"] + "\n\n"
    total_probability *= secondary["probability"]

    description_string += text_resources.EnglishTexts.final_text.format(
        int_probability=int(1/total_probability),
        total_probability=total_probability
    ) + "\n"

    print(description_string)

    #mode = pieces["all"][2]["mode_properties"]
    #mode = {"mgong": mode["gong_lvlv"], "mfinal": mode["final_note"]}
    #pitch = {"pitch_list": [x["pitch"] for x in pieces["all"][2]["music"]["suzipu"]]}
    #secondary = {"secondary_list": [x["secondary"] for x in pieces["all"][2]["music"]["suzipu"]]}

    with open(filename+".txt", "w") as file:
        file.write(description_string)

    realize_midi(
        lyrics=lyrics,
        mode={"gong_lvlv": mode["mgong"], "final_pitch": mode["mfinal"]},
        relative_pitch=pitch["pitch_list"],
        secondary=secondary["secondary_list"],
        meter=cipai["meter"],
        filename=filename
    )

    #for m, s in zip(cipai["meter"], secondary["secondary_list"]):
    #    print(m, s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python3 main.py",
        description="Stylistic Composition in Jiang Kui's Ci Style",
        epilog="https://github.com/SuziAI/SuziGEN"
    )

    parser.add_argument(
        "--kuiscima-repo-dir",
        required=True,
        help="Path to the folder containing the cloned KuiSCIMA repository (https://github.com/SuziAI/KuiSCIMA)"
    )

    parser.add_argument(
        "--cipai",
        required=True,
        help="The cipai string of the the cipai that should be generated. The string must be enclosed by quotation marks.\n"
             "\n"
             "'p' indicates ping, 'z' indicates ze.\n"
             "'P' indicates rhyme position with ping, 'Z' indicates rhyme position with ze.\n"
             "'.', ',', ':', ';', '!', '?', '。', '，', '！', '？', '：', '；' correspond to ju pause.\n"
             "'、' corresponds to dou pause. '/' indicates the stanzaic division.\n"
             "\n"
             'Example: "zzzpp，zzzppZ。zppZ。/zzppZ，zzpppZ。ppzZ。"'
    )

    parser.add_argument(
        "--lyrics",
        required=False,
        help="The lyrics accompanying the cipai. The string must be enclosed by quotation marks."
             "(For display purpose in the MIDI file only.)"
    )

    parser.add_argument(
        "--mode",
        required=False,
        help="The name of the mode as indicated in class music.GongdiaoModeList. Capitalization, whitespaces, and the "
             "corresponding simplified and traditional Chinese characters do not affect the result. If not supplied, "
             "the mode will be sampled randomly. Pinyin 'ü' can also be supplied as 'v'."
    )

    parser.add_argument(
        "--output-file-name",
        required=True,
        help="Path to the output file name. The software generates two files, one .txt (containing the verbal "
             "explanation of the generation process) and a .mid containing a MIDI rendition of the generated piece."
    )

    args = parser.parse_args()

    def preprocess_mode_name(mode_name: str):
        return chinese_converter.to_simplified("".join(mode_name.lower().split()).replace("ü", "v"))

    mode_name = args.mode
    mode = None
    if mode_name is not None:
        mode_name = preprocess_mode_name(mode_name)
        for gongdiaomode in dataclasses.astuple(GongdiaoModeList()):
            if mode_name in (preprocess_mode_name(gongdiaomode.name), preprocess_mode_name(gongdiaomode.chinese_name)):
                mode = {
                    "mgong": gongdiaomode.gong_lvlv,
                    "mfinal": gongdiaomode.final_note,
                    "name": gongdiaomode.name,
                    "chinese_name": gongdiaomode.chinese_name
                }
                break

    raw_cipai = args.cipai
    cipai = cipai_string_to_properties(raw_cipai)

    raw_lyrics = args.lyrics
    if raw_lyrics is None:
        lyrics = [""] * len(cipai["meter"])
    else:
        lyrics = lyrics_string_to_list(raw_lyrics)

    # build and load precomputed data
    pieces = load_17_pieces_data(kuiscima_repo_dir=args.kuiscima_repo_dir)

    probabilities = load_probabilities(kuiscima_repo_dir=args.kuiscima_repo_dir)

    generate(
        lyrics=lyrics,
        mode=mode,
        cipai=cipai,
        pieces=pieces,
        probabilities=probabilities,
        filename=args.output_file_name
    )