from mode import generate_mode
from pitch_and_secondary import generate_pitch
from repetitions import generate_repetition
from save_and_load import load_17_pieces_data, load_probabilities

import numpy as np

def generate(cipai, pieces):
    # load precomputed calculations
    probabilities = load_probabilities()
    contour_distributions = probabilities["contour_distributions"]
    initial_state_distributions = probabilities["initial_state_distributions"]
    function_distributions = probabilities["function_distributions"]
    pitch_transition_probabilities = probabilities["pitch_transition_probabilities"]

    total_probability = 1.

    description_string = ""
    repetition = generate_repetition(cipai)
    description_string += repetition["description"] + "\n\n"
    #total_probability *= repetition["probability"] ## maybe TODO?

    # DEBUG TODO
    repetition["repetition"] = ['r', 'r', 'r', '.', '.', '.', '.', '.', '.', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', '.', '.', '.', '.', '.', '.', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r']
    repetition["repetition"] = ['r', 'r', 'r', 'r', '.', '.', '.', '.', '.', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', '.', '.', '.', '.', '.', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r']
    repetition["repetition"] = ['r', '.', '.', '.', '.', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', '.', '.', '.', '.', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r']
    repetition["repetition"] = ['r', 'r', '.', '.', '.', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', '.', '.', '.', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r']
    repetition["repetition"] = ['r', 'r', '.', '.', '.', 'r', 'r', 'r', 'r', '.', '.', '.', '.', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', '.', '.', '.', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', '.', '.', '.', '.', 'r', 'r', 'r', 'r', 'r', 'r']

    mode = generate_mode(pieces)
    description_string += mode["description"] + "\n\n"
    total_probability *= mode["probability"]

    print(repetition["repetition"])

    pitch = generate_pitch(
        initial_state_distributions=initial_state_distributions,
        contour_distributions=contour_distributions,
        function_distributions=function_distributions,
        pitch_transition_probabilities=pitch_transition_probabilities,
        cipai=cipai,
        mode=mode,
        repetition=repetition
    )
    description_string += pitch["description"] + "\n\n"
    total_probability *= pitch["probability"]

    print(description_string)
    print(pitch["pitch_list"])
    print(f"Wow, your personal piece is generated with a chance of around 1 : 1{int(1/total_probability):,} ({total_probability:,.5g})!")


if __name__ == "__main__":
    pieces = load_17_pieces_data()
    generate(pieces["all"][0]["cipai"], pieces)