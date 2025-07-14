import text_resources
from midi import realize_midi
from mode import generate_mode
from music import absolute_pitch_to_function, get_absolute_tone_inventory, relative_pitch_to_absolute_pitch
from pitch_and_secondary import generate_pitch, generate_secondary
from repetitions import generate_repetition, remove_repetition_for_secondary
from save_and_load import load_17_pieces_data, load_probabilities

import numpy as np


def generate(cipai, pieces):
    # load precomputed calculations
    probabilities = load_probabilities()

    pitch_contour_distributions = probabilities["pitch_contour_distributions"]
    pitch_initial_state_distributions = probabilities["pitch_initial_state_distributions"]
    pitch_function_distributions = probabilities["pitch_function_distributions"]
    pitch_transition_probabilities = probabilities["pitch_transition_probabilities"]

    secondary_group_distributions = probabilities["secondary_group_distributions"]
    secondary_zhe_ye_distributions = probabilities["secondary_zhe_ye_distributions"]
    secondary_initial_state_distributions = probabilities["secondary_initial_state_distributions"]
    secondary_transition_probabilities = probabilities["secondary_transition_probabilities"]

    total_probability = 1.

    description_string = ""
    pitch_repetition = generate_repetition(cipai)
    description_string += pitch_repetition["description"] + "\n\n"
    #total_probability *= repetition["probability"] ## maybe TODO?

    # DEBUG TODO
    #pitch_repetition["repetition"] = ['r', 'r', 'r', '.', '.', '.', '.', '.', '.', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', '.', '.', '.', '.', '.', '.', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r']
    #pitch_repetition["repetition"] = ['r', 'r', 'r', 'r', '.', '.', '.', '.', '.', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', '.', '.', '.', '.', '.', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r']
    #pitch_repetition["repetition"] = ['r', '.', '.', '.', '.', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', '.', '.', '.', '.', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r']
    #pitch_repetition["repetition"] = ['r', 'r', '.', '.', '.', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', '.', '.', '.', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r']
    #pitch_repetition["repetition"] = ['r', 'r', '.', '.', '.', 'r', 'r', 'r', 'r', '.', '.', '.', '.', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', '.', '.', '.', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', '.', '.', '.', '.', 'r', 'r', 'r', 'r', 'r', 'r']
    #pitch_repetition["repetition"] = ['.', '.', '.', '.', '.', 'r', 'r', 'r', 'r', '.', '.', '.', '.', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', '.', '.', '.', '.', '.', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', '.', '.', '.', '.', 'r', 'r', 'r', 'r', 'r', 'r']
    #pitch_repetition["repetition"] = ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.']

    # DEBUG TODO
    #pitch_repetition["repetition"] = ['.']*len(cipai["meter"])

    secondary_repetition = {"repetition": remove_repetition_for_secondary(cipai, pitch_repetition["repetition"])}

    mode = generate_mode(pieces)
    description_string += mode["description"] + "\n\n"
    total_probability *= mode["probability"]

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
    )

    print(description_string)

    print(pitch_repetition["repetition"])
    print(secondary_repetition["repetition"])

    with open("output.txt", "w") as file:
        file.write(description_string)

    realize_midi({"gong_lvlv": mode["mgong"], "final_pitch": mode["mfinal"]}, pitch["pitch_list"], secondary["secondary_list"], cipai["meter"], "output")

    #for m, s in zip(cipai["meter"], secondary["secondary_list"]):
    #    print(m, s)

if __name__ == "__main__":
    pieces = load_17_pieces_data()
    generate(pieces["all"][0]["cipai"], pieces)