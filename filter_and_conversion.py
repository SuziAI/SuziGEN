import dataclasses

import numpy as np

from music import get_absolute_tone_inventory, GongcheMelodySymbol, GongdiaoStep, get_relative_tone_inventory, \
    absolute_pitch_to_interval, relative_pitch_to_absolute_pitch


def filter_by_gong(pieces, gong_lvlv_array):
    return_pieces = []
    if isinstance(gong_lvlv_array, str):
        gong_lvlv_array = [gong_lvlv_array]
    for gong_lvlv in gong_lvlv_array:
        return_pieces += [piece for piece in pieces if piece["mode_properties"]["gong_lvlv"] == gong_lvlv]
    return return_pieces


def filter_by_final(pieces, final_degree_array):
    return_pieces = []
    if isinstance(final_degree_array, str):
        final_degree_array = [final_degree_array]
    for final_degree in final_degree_array:
        return_pieces += [piece for piece in pieces if piece["mode_properties"]["final_note"] == final_degree]
    return return_pieces


def filter_by_mode(pieces, mode_properties):
    return_pieces = []
    return_pieces += [piece for piece in pieces if piece["mode_properties"]["final_note"] == mode_properties["final_note"] and piece["mode_properties"]["gong_lvlv"] == mode_properties["gong_lvlv"]]
    return return_pieces


def filter_by_final_secondary(pieces, final_secondary_array):
    return_pieces = []
    if isinstance(final_secondary_array, str):
        final_secondary_array = [final_secondary_array]
    for final_secondary in final_secondary_array:
        return_pieces += [piece for piece in pieces if piece["music"]["secondary"][-1] == final_secondary]
    return return_pieces


def f_S_to_N(s, mgong): # A function can have multiple suzipu pitches
    all_functions = get_absolute_tone_inventory(mgong)
    suzipu = [GongcheMelodySymbol.to_simple(dataclasses.astuple(GongcheMelodySymbol())[idx]) for idx, function in enumerate(all_functions) if function == s]
    return suzipu


def f_N_to_S(n, mgong): # A function can have multiple suzipu pitches
    for function in dataclasses.astuple(GongdiaoStep()):
        if n in f_S_to_N(function, mgong):
            return function


def get_suzipu_pitches(gong_lvlv):
    return [pitch for pitch in get_relative_tone_inventory(gong_lvlv)]


def add_mgong(mgong, n_pitch, index_array):
    pitch_inventory = get_suzipu_pitches(mgong)

    single_value = isinstance(index_array, int)

    if single_value:
        index_array = [index_array]
    else:
        index_array = [0] + list(index_array)  # since n_pitch + (i_1, i_2) := (n_pitch, n_pitch + i_1, n_pitch + i_1 + i_2)

    try:
        output_array = []
        n_idx = pitch_inventory.index(n_pitch)
        for i in index_array:
            n_idx += i
            output_array.append(pitch_inventory[n_idx])
        if None in output_array:  # for LINZHONG we can have None as result! This must be filtered out
            return None
        return output_array[0] if single_value else output_array
    except (ValueError, IndexError):
        return None


def relative_pitch_to_interval(mode_properties, relative_pitch_list):
    return absolute_pitch_to_interval(mode_properties,
                                          relative_pitch_to_absolute_pitch(mode_properties, relative_pitch_list))


def relative_pitch_to_contour(mode_properties, relative_pitch_list):
    return np.sign(relative_pitch_to_interval(mode_properties, relative_pitch_list))