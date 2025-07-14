import csv
import json
import numpy as np
import os
import pickle

from music import get_tone_inventory, tone_inventory_convert_pitch, GongcheMelodySymbol, \
    relative_pitch_to_absolute_pitch, absolute_pitch_to_interval, absolute_pitch_to_function
from pitch_and_secondary import *


def build_17_pieces_data():
    def load_latinized_titles():
        with open('./cipai/repetition.txt') as csvfile:
            cipai_file = csv.reader(csvfile, delimiter=';')
            full_titles = [row for row in cipai_file]

            title_list = []

            for piece in full_titles:
                title_list.append(piece[0])
            return title_list

    def load_cipai_file():
        with open('./cipai/cipai.txt') as csvfile:
            cipai_file = csv.reader(csvfile, delimiter=';')
            full_cipai = [row for row in cipai_file]

            cipai_list = []

            for piece in full_cipai:
                stripped_content = piece[2].replace(" ", "")

                piece_tones = []
                for character in stripped_content:
                    if character in ("z", "Z"):
                        piece_tones.append("ze")
                    elif character in ("p", "P"):
                        piece_tones.append("ping")

                piece_meter = []
                idx = 0
                while idx < len(stripped_content):
                    character = stripped_content[idx]
                    next_character = stripped_content[idx + 1]
                    if idx < len(stripped_content) - 2 and stripped_content[idx + 2] == "/":
                        piece_meter.append("pian")
                        idx += 3
                    elif next_character in ("。", "？", "，"):
                        piece_meter.append("ju")
                        idx += 2
                    elif next_character == "、":
                        piece_meter.append("dou")
                        idx += 2
                    else:
                        piece_meter.append("")
                        idx += 1

                cipai_list.append({"tones": piece_tones, "meter": piece_meter})

            return cipai_list

    def load_repetition_file():
        with open('./cipai/repetition.txt') as csvfile:
            repetition_file = csv.reader(csvfile, delimiter=';')
            full_repetition = [row for row in repetition_file]

            repetition_list = []

            for piece in full_repetition:
                repetition_list.append(list(piece[2].strip().replace("/", "")))

            return repetition_list

    def get_folder_contents(path, extension=None):
        file_list = []
        try:
            for file_path in sorted(os.listdir(path)):
                file_path = os.path.join(path, file_path)
                if os.path.isdir(file_path):
                    file_list += get_folder_contents(file_path, extension)
                if not extension or file_path.lower().endswith(f'.{extension}'):
                    file_list.append(file_path)
        except Exception as e:
            print(f"Could not read files from directory {path}. {e}")
        return file_list

    def get_title_list(content):
        title_list = []
        for box in content:
            if box["box_type"] == "Title":
                title_list.append(box["text_content"])
        return title_list

    def get_lyrics_list(content):
        lyrics_list = []
        for box in content:
            if box["box_type"] == "Music" and box["text_content"] != "":
                lyrics_list.append(box["text_content"])
        return lyrics_list

    def get_suzipu_list(content):
        suzipu_list = []
        for box in content:
            if box["box_type"] == "Music" and box["notation_content"]["pitch"] is not None:
                suzipu_list.append(box["notation_content"])
        return suzipu_list

    def relative_pitch_to_index(mode_properties, pitch_list, pitch_to_index=None):
        tone_inventory = get_tone_inventory(mode_properties["gong_lvlv"])
        index_list = []
        for pitch in pitch_list:
            index_list.append(pitch_to_index[pitch])
        return index_list

    FULL_PAUSE = [".", "。", "?", "？", ":", "："]
    PAUSE = [",", "，", "、"]

    def open_piece(path):
        with open(path, "r") as file:
            piece = json.load(file)
            del piece["version"]
            del piece["composer"]

            if piece["notation_type"] == "Suzipu":
                piece["title"] = {}
                piece["title"] = get_title_list(piece["content"])[0]

                piece["lyrics"] = {}
                piece["lyrics"]["list"] = get_lyrics_list(piece["content"])
                piece["lyrics"]["string"] = "".join(
                    [char if not (len(char) > 1 and char[1] in FULL_PAUSE) else "".join([char, "\n"]) for char in
                     piece["lyrics"]["list"]])

                piece["music"] = {}
                piece["music"]["suzipu"] = get_suzipu_list(piece["content"])
                piece["music"]["relative_pitch"] = [note["pitch"] for note in piece["music"]["suzipu"]]
                piece["music"]["secondary"] = [str(note["secondary"]) for note in piece["music"]["suzipu"]]
                piece["music"]["absolute_pitch"] = relative_pitch_to_absolute_pitch(piece["mode_properties"],
                                                                                    piece["music"]["relative_pitch"])
                piece["music"]["function"] = absolute_pitch_to_function(piece["mode_properties"],
                                                                        piece["music"]["absolute_pitch"])
                piece["music"]["interval"] = absolute_pitch_to_interval(piece["mode_properties"],
                                                                        piece["music"]["absolute_pitch"])

                piece["music"]["retrograde_suzipu"] = piece["music"]["suzipu"][::-1]
                piece["music"]["retrograde_relative_pitch"] = piece["music"]["relative_pitch"][::-1]
                piece["music"]["retrograde_secondary"] = piece["music"]["secondary"][::-1]
                piece["music"]["retrograde_absolute_pitch"] = piece["music"]["absolute_pitch"][::-1]
                piece["music"]["retrograde_function"] = piece["music"]["function"][::-1]
                piece["music"]["retrograde_interval"] = -piece["music"]["interval"][::-1]

            del piece["content"]
            return piece

    corpus_dir = "./KuiSCIMA/KuiSCIMA/symbolic_dataset/02_normalized_edition"
    json_files = get_folder_contents(corpus_dir, "json")

    all_cipai = load_cipai_file()
    all_repetitions = load_repetition_file()
    all_latinized_titles = load_latinized_titles()

    pieces = []
    for path in json_files:
        piece = open_piece(path)
        if piece["notation_type"] == "Suzipu" and piece["title"]:
            piece["cipai"] = all_cipai[len(pieces)]
            piece["latinized_title"] = all_latinized_titles[len(pieces)]
            piece["retrograde_cipai"] = {}
            piece["retrograde_cipai"]["tones"] = piece["cipai"]["tones"][::-1]
            piece["retrograde_cipai"]["meter"] = piece["cipai"]["meter"][::-1]
            piece["repetitions"] = all_repetitions[len(pieces)]
            piece["retrograde_repetitions"] = piece["repetitions"][::-1]
            pieces.append(piece)
            #print(piece["title"], len(piece["music"]["relative_pitch"]), len(piece["cipai"]["tones"]),
            #      len(piece["repetitions"]))

    with open("17_pieces_data.pkl", "wb") as file_handle:
        pickle.dump(pieces, file_handle)


def load_17_pieces_data():
    if not os.path.exists("17_pieces_data.pkl"):
        build_17_pieces_data()

    with open("17_pieces_data.pkl", "rb") as file_handle:
        all_pieces = pickle.load(file_handle)
        all_pieces_yanyue = [piece for piece in all_pieces if piece["title"] not in ("角招", "徴招")]
        return {"all": all_pieces, "yanyue": all_pieces_yanyue}

def load_probabilities():
    pieces = load_17_pieces_data()
    if not os.path.exists("probabilities.pkl"):
        pitch_contour_distributions = get_pitch_contour_distributions(pieces)
        pitch_initial_state_distributions = get_pitch_initial_state_distributions(pieces)
        pitch_function_distributions = get_pitch_function_distributions(pieces)
        pitch_transition_probabilities = get_pitch_transition_probabilities(pieces)

        secondary_group_distributions = get_secondary_group_distributions(pieces)
        secondary_zhe_ye_distributions = get_secondary_zhe_ye_distributions(pieces)
        secondary_initial_state_distributions = get_secondary_initial_state_distributions(pieces)
        secondary_transition_probabilities = get_secondary_transition_probabilities(pieces)

        with open("probabilities.pkl", "wb") as file_handle:
            pickle.dump({
                "pitch_contour_distributions": pitch_contour_distributions,
                "pitch_initial_state_distributions": pitch_initial_state_distributions,
                "pitch_function_distributions": pitch_function_distributions,
                "pitch_transition_probabilities": pitch_transition_probabilities,

                "secondary_initial_state_distributions": secondary_initial_state_distributions,
                "secondary_group_distributions": secondary_group_distributions,
                "secondary_zhe_ye_distributions": secondary_zhe_ye_distributions,
                "secondary_transition_probabilities": secondary_transition_probabilities
            }, file_handle)

    with open("probabilities.pkl", "rb") as file_handle:
        probabilities = pickle.load(file_handle)
        return probabilities