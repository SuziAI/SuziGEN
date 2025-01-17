from filter_and_conversion import filter_by_mode
from mode import generate_mode
from repetitions import generate_repetition
from save_and_load import load_17_pieces_data


def generate(cipai, pieces):

    description_string = ""
    repetition = generate_repetition(cipai)
    description_string += repetition["description"] + "\n\n"

    mode = generate_mode(pieces)
    description_string += mode["description"] + "\n\n"

    print(description_string)


if __name__ == "__main__":
    pieces = load_17_pieces_data()
    generate(pieces["all"][14]["cipai"], pieces)