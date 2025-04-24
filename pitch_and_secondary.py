import copy
import dataclasses
import itertools
import random
import scipy.sparse as sp

import text_resources
from distribution import *
from filter_and_conversion import filter_by_final, relative_pitch_to_contour, f_S_to_N, add_mgong, get_suzipu_pitches, \
    f_N_to_S, relative_pitch_to_interval, filter_by_final_secondary
from music import GongdiaoStep, Lvlv, SimplePitchList, SuzipuSecondarySymbol, \
    SecondaryFunctionList, ReducedSecondaryList
from text_resources import EnglishTexts


def get_k_grams(base_list, k, ngram_property=None):
    ngrams = []
    for idx in range(len(base_list)-(k-1)):
        triple = tuple(base_list[idx:idx+k])
        if ngram_property is None or (ngram_property and ngram_property(triple)):
            ngrams.append(triple)
    return ngrams


def get_pitch_contour_distributions(pieces):
    def get_tone_contour_distributions(pieces, retrograde=True):
        tone_dict = {}
        for piece in pieces:
            tone_trigrams = get_k_grams(piece["retrograde_cipai"]["tones"], 3)
            interval_trigrams = get_k_grams(piece["music"]["retrograde_interval"], 2)
            contour_trigrams = [np.sign(interval) for interval in interval_trigrams]
            for tone, contour in zip(tone_trigrams, contour_trigrams):
                if 0 not in contour:
                    if tone not in tone_dict:
                        tone_dict[tone] = [contour]
                    else:
                        tone_dict[tone].append(contour)

        for key in tone_dict.keys():
            # print(tone_dict[key])
            state_space, absolute_counts = np.unique(
                sorted(tone_dict[key], key=lambda arr: 2 * (0.5 * arr[1] + 0.5) + 0.5 * arr[0] + 0.5),
                return_counts=True,
                axis=0)
            state_space = [[int(element) for element in state] for state in state_space]
            tone_dict[key] = Distribution(state_space, absolute_counts)
        return tone_dict

    def get_tone_contours_one_step(pieces, c_1, retrograde=True):
        full_tone_dict = get_tone_contour_distributions(pieces, retrograde)
        tone_dict = {}
        for tone_triple, contour_distribution in zip(full_tone_dict.keys(), full_tone_dict.values()):
            filtered_contour_pairs = [pair for pair in contour_distribution.sample_space() if pair[0] == c_1]
            filtered_contour_probabilities = [contour_distribution[filtered_pair] for filtered_pair in
                                              filtered_contour_pairs]
            tone_dict[tone_triple] = Distribution.from_dict(
                {contour_pair[1]: normalized_prob for contour_pair, normalized_prob in
                 zip(filtered_contour_pairs, filtered_contour_probabilities)})
        return tone_dict

    contour_distributions = {
        "all": get_tone_contour_distributions(pieces["all"]),
        -1: get_tone_contours_one_step(pieces["all"], -1),
        1: get_tone_contours_one_step(pieces["all"], 1),
        0: get_tone_contours_one_step(pieces["all"], np.random.choice([-1, 1], 1)[0])  # this should never be invoked
    }
    return contour_distributions


def get_pitch_function_distributions(pieces):
    def get_beginning_function_distribution(pieces, mfinal):
        traverse_pieces = filter_by_final(pieces, mfinal)
        count_list = []
        for P in traverse_pieces:
            piece_count_list = []
            pian_idx = P["cipai"]["meter"].index("pian")
            for idx, (f, meter) in enumerate(zip(P["music"]["function"], P["cipai"]["meter"])):
                if idx == 0 or idx == pian_idx+1:
                    piece_count_list.append(f)
            count_list += list(
                np.unique(piece_count_list))  # only take both beginning of first and second stanza if different!
        return Distribution(*np.unique(count_list, return_counts=True)).extend_space(dataclasses.astuple(GongdiaoStep()))


    def get_ju_function_distribution(pieces, mfinal):
        traverse_pieces = filter_by_final(pieces, mfinal)
        count_list = []
        for P in traverse_pieces:
            for f, meter in zip(P["music"]["function"], P["cipai"]["meter"]):
                if meter in ("ju", "pian"):
                    count_list.append(f)
        return Distribution(*np.unique(count_list, return_counts=True)).extend_space(dataclasses.astuple(GongdiaoStep()))


    def get_after_ju_function_distribution(pieces, mfinal):
        traverse_pieces = filter_by_final(pieces, mfinal)
        count_list = []
        for P in traverse_pieces:
            for idx, f in enumerate(P["music"]["function"]):
                if idx > 1 and P["cipai"]["meter"][idx - 1] in ("ju", "pian"):
                    count_list.append(f)
        return Distribution(*np.unique(count_list, return_counts=True)).extend_space(dataclasses.astuple(GongdiaoStep()))


    def get_dou_function_distribution(pieces, mfinal):
        traverse_pieces = filter_by_final(pieces, mfinal)
        count_list = []
        for P in traverse_pieces:
            for f, meter in zip(P["music"]["function"], P["cipai"]["meter"]):
                if meter == "dou":
                    count_list.append(f)
        return Distribution(*np.unique(count_list, return_counts=True)).extend_space(dataclasses.astuple(GongdiaoStep()))


    def get_after_dou_function_distribution(pieces, mfinal):
        traverse_pieces = filter_by_final(pieces, mfinal)
        count_list = []
        for P in traverse_pieces:
            for idx, f in enumerate(P["music"]["function"]):
                if idx > 1 and P["cipai"]["meter"][idx - 1] == "dou":
                    count_list.append(f)
        return Distribution(*np.unique(count_list, return_counts=True)).extend_space(dataclasses.astuple(GongdiaoStep()))

    output_dict = {}
    for mfinal in [GongdiaoStep.GONG, GongdiaoStep.SHANG, GongdiaoStep.YU]:
        output_dict[mfinal] = {
            "beginning": get_beginning_function_distribution(pieces["all"], mfinal=mfinal),
            "ju": get_ju_function_distribution(pieces["all"], mfinal=mfinal),
            "after_ju": get_after_ju_function_distribution(pieces["all"], mfinal=mfinal),
            "dou": get_dou_function_distribution(pieces["all"], mfinal=mfinal),
            "after_dou": get_after_dou_function_distribution(pieces["all"], mfinal=mfinal),
        }

    return output_dict


def get_secondary_group_distributions(pieces):
    def get_beginning_function_distribution(pieces, final_secondary):
        traverse_pieces = filter_by_final_secondary(pieces, final_secondary)
        count_list = []
        for P in traverse_pieces:
            piece_count_list = []
            pian_idx = P["cipai"]["meter"].index("pian")
            for idx, (f, meter) in enumerate(zip(P["music"]["secondary"], P["cipai"]["meter"])):
                if idx == 0 or idx == pian_idx+1:
                    piece_count_list.append(SuzipuSecondarySymbol.to_function(f))
            count_list += list(
                np.unique([piece_count_list]))  # only take both beginning of first and second stanza if different!
        return Distribution(*np.unique(count_list, return_counts=True)).extend_space(SecondaryFunctionList)


    def get_ju_function_distribution(pieces, final_secondary):
        traverse_pieces = filter_by_final_secondary(pieces, final_secondary)
        count_list = []
        for P in traverse_pieces:
            for f, meter in zip(P["music"]["secondary"], P["cipai"]["meter"]):
                if meter in ("ju", "pian"):
                    count_list.append(SuzipuSecondarySymbol.to_function(f))
        return Distribution(*np.unique(count_list, return_counts=True)).extend_space(SecondaryFunctionList)


    def get_after_ju_function_distribution(pieces, final_secondary):
        traverse_pieces = filter_by_final_secondary(pieces, final_secondary)
        count_list = []
        for P in traverse_pieces:
            for idx, f in enumerate(P["music"]["secondary"]):
                if idx > 1 and P["cipai"]["meter"][idx - 1] in ("ju", "pian"):
                    count_list.append(SuzipuSecondarySymbol.to_function(f))
        return Distribution(*np.unique(count_list, return_counts=True)).extend_space(SecondaryFunctionList)


    def get_dou_function_distribution(pieces, final_secondary):
        traverse_pieces = filter_by_final_secondary(pieces, final_secondary)
        count_list = []
        for P in traverse_pieces:
            for f, meter in zip(P["music"]["secondary"], P["cipai"]["meter"]):
                if meter == "dou":
                    count_list.append(SuzipuSecondarySymbol.to_function(f))
        return Distribution(*np.unique(count_list, return_counts=True)).extend_space(SecondaryFunctionList)


    def get_after_dou_function_distribution(pieces, final_secondary):
        traverse_pieces = filter_by_final_secondary(pieces, final_secondary)
        count_list = []
        for P in traverse_pieces:
            for idx, f in enumerate(P["music"]["secondary"]):
                if idx > 1 and P["cipai"]["meter"][idx - 1] == "dou":
                    count_list.append(SuzipuSecondarySymbol.to_function(f))
        return Distribution(*np.unique(count_list, return_counts=True)).extend_space(SecondaryFunctionList)

    output_dict = {}
    for final_secondary in [SuzipuSecondarySymbol.ADD_XIAO_ZHU, SuzipuSecondarySymbol.ADD_DA_DUN]:
        output_dict[final_secondary] = {
            "beginning": get_beginning_function_distribution(pieces["all"], final_secondary=final_secondary),
            "ju": get_ju_function_distribution(pieces["all"], final_secondary=final_secondary),
            "after_ju": get_after_ju_function_distribution(pieces["all"], final_secondary=final_secondary),
            "dou": get_dou_function_distribution(pieces["all"], final_secondary=final_secondary),
            "after_dou": get_after_dou_function_distribution(pieces["all"], final_secondary=final_secondary),
        }

    return output_dict


def get_secondary_zhe_ye_distributions(pieces):
    def mapping(symbol):
        if symbol == "ZHE":
            return "ZHE"
        elif symbol == "YE":
            return "YE"
        else:
            return "any"

    def get_scale_degreee_distributions(pieces, retrograde=True):
        secondary_dict = {}
        for piece in pieces:
            scale_degree_tuples = get_k_grams(piece["music"]["retrograde_function"], 2)
            secondary_list = [mapping(symbol) for symbol in piece["music"]["retrograde_secondary"][::-1]]
            for secondary_group, scales in zip(secondary_list, scale_degree_tuples):
                if secondary_group not in secondary_dict:
                    secondary_dict[scales] = [secondary_group]
                else:
                    secondary_dict[scales].append(secondary_group)
                secondary_dict[scales] = Distribution.from_sample(secondary_dict[scales])
        return secondary_dict

    contour_distributions = {
        "base": get_scale_degreee_distributions(pieces["all"]),
    }
    return contour_distributions


def igram(P, k):
    return get_k_grams(P["music"]["retrograde_interval"], k - 1)


def sgram(P, k):
    # do not differentiate between Zhe and Ye
    return get_k_grams([symbol if symbol not in ("ZHE", "YE") else "ZheYe" for symbol in P["music"]["retrograde_secondary"]], k)


def get_pitch_cadential_phrases(P):
    igrams = igram(P, 3)
    pian_idx = P["retrograde_cipai"]["meter"].index("pian")
    phrase_list = [igrams[0], igrams[pian_idx]]  # final cadential phrase and pian cadential phrase
    unq = np.unique(phrase_list, axis=0)  # only take the pian cadential phrase if it is different from the final one
    return list(unq)


def get_secondary_cadential_phrases(P):
    sgrams = sgram(P, 3)
    pian_idx = P["retrograde_cipai"]["meter"].index("pian")
    phrase_list = [sgrams[0], sgrams[pian_idx]]  # final cadential phrase and pian cadential phrase
    unq = [phrase_list[0]]  # only take the pian cadential phrase if it is different from the final one
    if phrase_list[1] != phrase_list[0]:
        unq.append(phrase_list[1])
    return list(unq)


def pitch_triple_to_contour(mgong_0):
    def inner(pitch_list):
        return tuple(relative_pitch_to_contour({"gong_lvlv": mgong_0}, pitch_list))

    return inner


def get_pitch_initial_state_distributions(pieces):
    initial_state_distributions = {}

    for mgong_0 in [Lvlv.HUANGZHONG, Lvlv.DALV, Lvlv.JIAZHONG, Lvlv.ZHONGLV, Lvlv.LINZHONG, Lvlv.YIZE, Lvlv.WUYI]:
        initial_state_distributions[mgong_0] = {}
        for mfinal_0 in [GongdiaoStep.GONG, GongdiaoStep.SHANG, GongdiaoStep.YU]:
            final_0 = [mfinal_0]
            final_1 = [mgong for mgong in dataclasses.astuple(GongdiaoStep()) if mgong != mfinal_0]

            P_0 = filter_by_final(pieces["yanyue"], final_0)  # pieces with same mfinal_0
            P_1 = filter_by_final(pieces["yanyue"], final_1)  # pieces not with mfinal_0

            I_0 = []
            for p in P_0:
                I_0 += get_pitch_cadential_phrases(p)
            I_1 = []
            for p in P_1:
                I_1 += get_pitch_cadential_phrases(p)

            N_0 = []
            N_1 = []
            for final_n in f_S_to_N(mfinal_0, mgong_0):
                N_0 += [tuple(add_mgong(mgong_0, final_n, interval_trigram)) for interval_trigram in I_0 if
                        add_mgong(mgong_0, final_n, interval_trigram)]
                N_1 += [tuple(add_mgong(mgong_0, final_n, interval_trigram)) for interval_trigram in I_1 if
                        add_mgong(mgong_0, final_n, interval_trigram)]

            N_0_distribution = Distribution.from_sample(N_0)
            N_1_distribution = Distribution.from_sample(N_1)

            N_0_distribution, N_1_distribution = Distribution.extend_to_same_space(N_0_distribution, N_1_distribution)
            # with 70% chance take the initial state from N_0
            initial_state_distributions[mgong_0][mfinal_0] = Distribution.convex_combination([N_0_distribution, N_1_distribution], [0.7, 0.3])
    return initial_state_distributions


def get_secondary_initial_state_distributions(pieces):
    initial_state_distributions = {}

    for final_secondary in [SuzipuSecondarySymbol.ADD_XIAO_ZHU, SuzipuSecondarySymbol.ADD_DA_DUN]:
        forbidden_secondary = [] if final_secondary == SuzipuSecondarySymbol.ADD_DA_DUN else [SuzipuSecondarySymbol.ADD_DA_DUN, SuzipuSecondarySymbol.ADD_DA_ZHU]
        A_0 = []
        A_1 = []
        for piece in pieces["all"]:
            if piece["title"] == "醉吟商小品":
                continue

            if piece["music"]["secondary"][-1] == final_secondary:
                A_0 += get_secondary_cadential_phrases(piece)
            else:
                A_1 += [[final_secondary, triple[1], triple[2]] for triple in get_secondary_cadential_phrases(piece) if triple[0] not in forbidden_secondary and triple[1] not in forbidden_secondary and triple[2] not in forbidden_secondary]

        A_0_distribution = Distribution.from_sample(A_0)
        A_1_distribution = Distribution.from_sample(A_1)

        A_0_distribution, A_1_distribution = Distribution.extend_to_same_space(A_0_distribution, A_1_distribution)

        initial_state_distributions[final_secondary] = Distribution.convex_combination([A_0_distribution, A_1_distribution], [0.7, 0.3])

    return initial_state_distributions


def get_pitch_initial_state(initial_state_distributions, contour_distributions, cipai, mode, start_idx=0):
    mgong_0, mfinal_0 = mode["mgong"], mode["mfinal"]

    retrograde_cipai = cipai["tones"][::-1]

    try:
        tone_dependent_distribution = initial_state_distributions[mgong_0][mfinal_0].get_conditioned_on_Q(
            contour_distributions["all"][tuple(retrograde_cipai[start_idx:start_idx+3])],
            pitch_triple_to_contour(mgong_0)
        )
    except ZeroDivisionError:
        tone_dependent_distribution = initial_state_distributions[mgong_0][mfinal_0]

    initial_state = tone_dependent_distribution.sample()

    return_string = EnglishTexts.second_stanza_pitch_initial_state.format(
        final_note=initial_state[0],
        cadential_phrase=[str(pitch) for pitch in initial_state][::-1],
        probability=tone_dependent_distribution[initial_state] * 100
    )

    return_string_first_stanza = EnglishTexts.first_stanza_pitch_initial_state_with_repetition.format(
        cadential_phrase=[str(pitch) for pitch in initial_state][::-1],
        probability=tone_dependent_distribution[initial_state] * 100
    )

    return {"initial_state": [str(entry) for entry in initial_state],
            "description": return_string,
            "description_first": return_string_first_stanza,
            "probability": tone_dependent_distribution[initial_state]}


def get_pitch_initial_state_probabilities(initial_state_distributions, contour_distributions, cipai, mode):
    mgong_0, mfinal_0 = mode["mgong"], mode["mfinal"]

    retrograde_cipai = cipai["tones"][::-1]

    tone_dependent_distribution = initial_state_distributions[mgong_0][mfinal_0].get_conditioned_on_Q(contour_distributions["all"][tuple(retrograde_cipai[0:3])],
                                                                                                      pitch_triple_to_contour(mgong_0))
    return tone_dependent_distribution


def get_secondary_initial_state(initial_state_distributions, final_secondary, start_idx=0):
    initial_state = initial_state_distributions[final_secondary].sample()

    return_string = EnglishTexts.second_stanza_secondary_initial_state.format(
        cadential_phrase=[str(secondary) for secondary in initial_state][::-1],
        probability=initial_state_distributions[initial_state[0]][initial_state] * 100
    )

    return_string_first_stanza = EnglishTexts.first_stanza_secondary_initial_state_with_repetition.format(
        cadential_phrase=[str(secondary) for secondary in initial_state][::-1],
        probability=initial_state_distributions[initial_state[0]][initial_state] * 100
    )

    return {"initial_state": [str(entry) for entry in initial_state],
            "description": return_string,
            "description_first": return_string_first_stanza,
            "probability": initial_state_distributions[initial_state[0]][initial_state]}


def get_secondary_initial_state_probabilities(initial_state_distributions, final_secondary):
    return initial_state_distributions[final_secondary]


def get_interval_n_grams(gong_lvlv, base_list, n):
    ngrams = []
    interval_list = base_list["interval"]
    suzipu_pitches = get_suzipu_pitches(gong_lvlv)

    for idx in range(len(interval_list) - (n - 1)):
        current_base = base_list["function"][idx]
        current_base_suzipu = f_S_to_N(current_base, gong_lvlv)
        current_base_suzipu_indices = [i for i, x in enumerate(suzipu_pitches) if x in current_base_suzipu]

        current_intervals = interval_list[idx:idx + n]

        for base_index in current_base_suzipu_indices:
            current_ngram = [suzipu_pitches[base_index]]
            current_position = base_index
            try:
                for interval in current_intervals:
                    current_position += interval
                    current_suzipu_pitch = suzipu_pitches[current_position]
                    if current_position < 0:
                        raise IndexError
                    current_ngram.append(current_suzipu_pitch)
            except IndexError:
                current_ngram = [suzipu_pitches[base_index]]
                current_position = base_index

            if None not in current_ngram and len(current_ngram) == n + 1:
                ngrams.append(current_ngram)

    return ngrams


def get_secondary_n_grams(base_list, n):
    ngrams = []
    secondary_list = [symbol if symbol not in ("ZHE", "YE") else "ZheYe" for symbol in base_list["secondary"]]

    for idx in range(len(secondary_list) - n):
        current_ngram = secondary_list[idx:idx+n+1]
        ngrams.append(current_ngram)

    return ngrams


def get_pitch_inverted_n_step_markov_chain(base_pieces, gong_lvlv, n):
    total_list = []
    for piece in base_pieces:
        total_list += get_interval_n_grams(gong_lvlv, piece["music"], n=n)
    markov_chain = {}
    for combination in itertools.product(*[SimplePitchList for idx in range(n)]):
        markov_chain[combination] = {}
        for symbol in SimplePitchList:
            markov_chain[combination][symbol] = 0

    # add absolute count
    for ngram in total_list:
        ngram.reverse()  # we must consider the inverted ngrams since we generate starting from the end
        triple = tuple(ngram[0:n])
        next_step = ngram[n]

        markov_chain[triple][next_step] += 1

    zero_rows = []
    # scale to probability vector
    for triple in itertools.product(*[SimplePitchList for idx in range(n)]):
        row_sum = sum(markov_chain[triple].values())
        if row_sum < 0.01:
            zero_rows.append(triple)
        else:
            for symbol in SimplePitchList:
                markov_chain[triple][symbol] /= row_sum

    # incorporate n-1 step probabilites with 5%, except for zero rows, which take the n-1 steps completely
    if n > 1:
        n_minus_one_chain = get_pitch_inverted_n_step_markov_chain(base_pieces, gong_lvlv, n - 1)

        for triple in itertools.product(*[SimplePitchList for idx in range(n)]):
            if triple in zero_rows:
                for symbol in SimplePitchList:
                    markov_chain[triple][symbol] = n_minus_one_chain[triple[1:n]][symbol]
            else:
                for symbol in SimplePitchList:
                    markov_chain[triple][symbol] = 0.95 * markov_chain[triple][symbol] + 0.05 * \
                                                   n_minus_one_chain[triple[1:n]][symbol]
                # scale to probability vector
                row_sum = sum(markov_chain[triple].values())
                for symbol in SimplePitchList:
                    markov_chain[triple][symbol] /= row_sum
    else:  # if zero, choose some random pitch (belonging to the mode!) to fill the empty row
        for zero_row in zero_rows:
            rands = get_suzipu_pitches(gong_lvlv=gong_lvlv)
            for r in rands:
                markov_chain[zero_row][r] = 1. / len(rands)

    return markov_chain


def get_secondary_inverted_n_step_markov_chain(base_pieces, final_secondary, n):
    total_list = []
    forbidden_secondary = [] if final_secondary == SuzipuSecondarySymbol.ADD_DA_DUN else [SuzipuSecondarySymbol.ADD_DA_DUN, SuzipuSecondarySymbol.ADD_DA_ZHU]

    for piece in base_pieces:
        total_list += get_secondary_n_grams(piece["music"], n=n)
    markov_chain = {}
    for combination in itertools.product(*[ReducedSecondaryList for idx in range(n)]):
        markov_chain[combination] = {}
        for symbol in ReducedSecondaryList:
            markov_chain[combination][symbol] = 0

    # add absolute count
    for ngram in total_list:
        ngram.reverse()  # we must consider the inverted ngrams since we generate starting from the end
        triple = tuple(ngram[0:n])
        next_step = ngram[n]

        markov_chain[triple][next_step] += 1

    zero_rows = []
    # scale to probability vector
    for triple in itertools.product(*[ReducedSecondaryList for idx in range(n)]):
        row_sum = sum(markov_chain[triple].values())
        if row_sum < 0.01:
            zero_rows.append(triple)
        else:
            for symbol in ReducedSecondaryList:
                markov_chain[triple][symbol] /= row_sum

    # incorporate n-1 step probabilites with 5%, except for zero rows, which take the n-1 steps completely
    if n > 1:
        n_minus_one_chain = get_secondary_inverted_n_step_markov_chain(base_pieces, final_secondary, n - 1)

        for triple in itertools.product(*[ReducedSecondaryList for idx in range(n)]):
            if triple in zero_rows:
                for symbol in ReducedSecondaryList:
                    markov_chain[triple][symbol] = n_minus_one_chain[triple[1:n]][symbol]
            else:
                for symbol in ReducedSecondaryList:
                    markov_chain[triple][symbol] = 0.95 * markov_chain[triple][symbol] + 0.05 * \
                                                   n_minus_one_chain[triple[1:n]][symbol]
                # scale to probability vector
                row_sum = sum(markov_chain[triple].values())
                for symbol in ReducedSecondaryList:
                    markov_chain[triple][symbol] /= row_sum
    else:  # if zero, choose some random allowed secondary to fill the empty row
        for zero_row in zero_rows:
            rands = [secondary for secondary in ReducedSecondaryList if secondary not in forbidden_secondary]
            for r in rands:
                markov_chain[zero_row][r] = 1. / len(rands)

    return markov_chain


def get_pitch_transition_probabilities(pieces):
    n = 3
    base_probabilities = {}
    for final in (GongdiaoStep.GONG, GongdiaoStep.SHANG, GongdiaoStep.YU):
        base_probabilities[final] = {}
        for gong_lvlv in dataclasses.astuple(Lvlv()):
            markov_chain = get_pitch_inverted_n_step_markov_chain(
                filter_by_final(pieces["all"], final),
                n=n,
                gong_lvlv=gong_lvlv
            )
            base_probabilities[final][gong_lvlv] = {key: Distribution.from_dict(markov_chain[key]) for key in markov_chain.keys()}
            for triple in itertools.product(*[SimplePitchList for idx in range(n)]):
                row_sum = sum(base_probabilities[final][gong_lvlv][triple].probabilities())
                if abs(row_sum - 1) > 1e-10:
                    raise ValueError("The generation process yielded no valid Markov chain. Some row sum is not equal to 1.")

    # The original probabilities lead to the Markov chain to oscillate between pairs often,
    # e.g., 'FAN', 'WU', 'FAN', 'WU', 'FAN', 'WU', 'FAN', and we do not want this behavior.
    # Therefore, let's reduce the probability of (a, b, a) |-> (b, a, b) to 0.1 of the original probability
    for final in (GongdiaoStep.GONG, GongdiaoStep.SHANG, GongdiaoStep.YU):
        for gong_lvlv in dataclasses.astuple(Lvlv()):
            current_keys = base_probabilities[final][gong_lvlv].keys()
            for key in current_keys:
                if key[0] == key[2]:
                    current_distribution = base_probabilities[final][gong_lvlv][key].distribution
                    current_distribution[key[1]] *= 0.25
                    base_probabilities[final][gong_lvlv][key].from_dict(current_distribution)

    # for the no repetition probabilities, we exclude that (a, b, c) gets mapped to (b, c, c)
    less_repetition_probabilities = copy.deepcopy(base_probabilities)
    for final in (GongdiaoStep.GONG, GongdiaoStep.SHANG, GongdiaoStep.YU):
        for gong_lvlv in dataclasses.astuple(Lvlv()):
            current_keys = less_repetition_probabilities[final][gong_lvlv].keys()
            for key in current_keys:
                current_distribution = less_repetition_probabilities[final][gong_lvlv][key].distribution
                current_distribution[key[2]] *= 0.1
                less_repetition_probabilities[final][gong_lvlv][key].from_dict(current_distribution)

    return {
        "base": base_probabilities,
        "less_repetition": less_repetition_probabilities
    }


def get_secondary_transition_probabilities(pieces):
    n = 3
    base_probabilities = {}
    for final_secondary in (SuzipuSecondarySymbol.ADD_XIAO_ZHU, SuzipuSecondarySymbol.ADD_DA_DUN):
        base_probabilities[final_secondary] = {}
        for gong_lvlv in dataclasses.astuple(Lvlv()):
            markov_chain = get_secondary_inverted_n_step_markov_chain(
                filter_by_final_secondary(pieces["all"], final_secondary),
                n=n,
                final_secondary=final_secondary
            )
            base_probabilities[final_secondary] = {key: Distribution.from_dict(markov_chain[key]) for key in
                                                    markov_chain.keys()}
            for triple in itertools.product(*[ReducedSecondaryList for idx in range(n)]):
                row_sum = sum(base_probabilities[final_secondary][triple].probabilities())
                if abs(row_sum - 1) > 1e-10:
                    raise ValueError(
                        "The generation process yielded no valid Markov chain. Some row sum is not equal to 1.")

    return {
        "base": base_probabilities,
    }


def generate_pitch(initial_state_distributions, contour_distributions, function_distributions, pitch_transition_probabilities, cipai, mode, repetition):
    cipai_meter = cipai["meter"]
    generated_piece = [None] * len(cipai_meter)

    retrograde_meter = cipai["meter"][::-1]
    retrograde_tones = cipai["tones"][::-1]
    retrograde_repetition = repetition["repetition"][::-1]

    mfinal = mode["mfinal"]
    mgong = mode["mgong"]

    pitch_initial_state = get_pitch_initial_state(
        initial_state_distributions=initial_state_distributions,
        contour_distributions=contour_distributions,
        cipai=cipai,
        mode=mode
    )
    description_string = pitch_initial_state["description"] + " "
    probability = pitch_initial_state["probability"]
    pitch_initial_state = pitch_initial_state["initial_state"]

    pian_idx = retrograde_meter.index("pian")

    def get_contour(current_triple):
        def inner(pitch):
            try:
                val = int(np.sign(
                    relative_pitch_to_interval({"gong_lvlv": mode["mgong"]}, [current_triple[2], pitch])[
                        0]))
                if val == 0:  # 0 must not occur in the dicts, so choose some value
                    val = -1
                return val
            except TypeError:  # can occur when interval of nonexistent pitch is calculated
                return None
        return inner

    def get_current_pitch_distribution(idx, current_triple):
        # get the correct distribution of pitches for generating the idx-th position
        def get_function_from_pitch(pitch):
            return f_N_to_S(pitch, mode["mgong"])

        if idx in (len(generated_piece) - 1, pian_idx - 1):  # First note of each stanza
            next_pitch_distribution = pitch_transition_probabilities["less_repetition"][mfinal][mgong][current_triple]
            try:
                distr = next_pitch_distribution.get_conditioned_on_Q(
                    function_distributions[mfinal]["beginning"],
                    get_function_from_pitch
                )
            except ZeroDivisionError:
                distr = next_pitch_distribution
        elif cipai["meter"][::-1][idx] == "ju": # ju position
            next_pitch_distribution = pitch_transition_probabilities["less_repetition"][mfinal][mgong][current_triple]
            try:
                distr = next_pitch_distribution.get_conditioned_on_Q(
                    function_distributions[mfinal]["ju"],
                    get_function_from_pitch
                )
            except ZeroDivisionError:
                distr = next_pitch_distribution
        elif idx < len(generated_piece) - 1 and cipai["meter"][::-1][idx + 1] == "ju": # one after ju position
            next_pitch_distribution = pitch_transition_probabilities["base"][mfinal][mgong][current_triple]
            try:
                distr = next_pitch_distribution.get_conditioned_on_Q(
                    function_distributions[mfinal]["after_ju"],
                    get_function_from_pitch
                )
            except ZeroDivisionError:
                distr = next_pitch_distribution
        elif cipai["meter"][::-1][idx] == "dou": # dou position
            next_pitch_distribution = pitch_transition_probabilities["less_repetition"][mfinal][mgong][current_triple]
            try:
                distr = next_pitch_distribution.get_conditioned_on_Q(
                    function_distributions[mfinal]["dou"],
                    get_function_from_pitch
                )
            except ZeroDivisionError:
                distr = next_pitch_distribution
        elif idx < len(generated_piece) - 1 and cipai["meter"][::-1][idx + 1] == "dou": # one after dou position
            next_pitch_distribution = pitch_transition_probabilities["less_repetition"][mfinal][mgong][current_triple]
            try:
                distr = next_pitch_distribution.get_conditioned_on_Q(
                    function_distributions[mfinal]["after_dou"],
                    get_function_from_pitch
                )
            except ZeroDivisionError:
                distr = next_pitch_distribution
        else:
            distr = pitch_transition_probabilities["less_repetition"][mfinal][mgong][current_triple]

        last_contour = int(np.sign(relative_pitch_to_interval({"gong_lvlv": mode["mgong"]}, [current_triple[1], current_triple[2]])[0]))
        if last_contour == 0:  # if 0 contour, i.e., tone repetition, use the previous contour!
            last_contour = int(np.sign(relative_pitch_to_interval({"gong_lvlv": mode["mgong"]}, [current_triple[0], current_triple[1]])[0]))
        try:
            distr = distr.get_conditioned_on_Q(
                contour_distributions[last_contour][tuple(retrograde_tones[idx-3:idx])],
                get_contour(current_triple)
            )
        except ZeroDivisionError:
            pass

        return distr

    def get_allowed_suzipu_list():
        allowed_suzipu_list = get_suzipu_pitches(mode["mgong"])
        allowed_suzipu_list = [s for s in allowed_suzipu_list if s is not None]
        return allowed_suzipu_list

    # now, get all possibilities for the first stanza, calculate the likelihood and sample accordingly
    allowed_suzipu_list = get_allowed_suzipu_list()

    def get_transition_matrix(idx):
        def get_function_from_pitch(pitch):
            return f_N_to_S(pitch, mode["mgong"])

        distr_dict = {}

        allowed_suzipu_list = get_allowed_suzipu_list()

        for triple in itertools.product(*[allowed_suzipu_list for idx in range(3)]):
            if idx in (len(generated_piece) - 1, pian_idx - 1):  # First note of stanza
                next_pitch_distribution = pitch_transition_probabilities["less_repetition"][mfinal][mgong][triple]
                distr = next_pitch_distribution.get_conditioned_on_Q(
                    function_distributions[mfinal]["beginning"],
                    get_function_from_pitch
                )
            elif cipai["meter"][::-1][idx] == "ju": # Generate a ju
                next_pitch_distribution = pitch_transition_probabilities["less_repetition"][mfinal][mgong][triple]
                distr = next_pitch_distribution.get_conditioned_on_Q(
                    function_distributions[mfinal]["ju"],
                    get_function_from_pitch
                )
            elif idx < len(generated_piece) - 1 and cipai["meter"][::-1][idx + 1] == "ju": # Generate one after a ju
                next_pitch_distribution = pitch_transition_probabilities["base"][mfinal][mgong][triple]
                distr = next_pitch_distribution.get_conditioned_on_Q(
                    function_distributions[mfinal]["after_ju"],
                    get_function_from_pitch
                )
            elif cipai["meter"][::-1][idx] == "dou": # Generate dou
                next_pitch_distribution = pitch_transition_probabilities["less_repetition"][mfinal][mgong][triple]
                distr = next_pitch_distribution.get_conditioned_on_Q(
                    function_distributions[mfinal]["dou"],
                    get_function_from_pitch
                )
            elif idx < len(generated_piece) - 1 and cipai["meter"][::-1][idx + 1] == "dou": # Generate one after dou
                next_pitch_distribution = pitch_transition_probabilities["less_repetition"][mfinal][mgong][triple]
                distr = next_pitch_distribution.get_conditioned_on_Q(
                    function_distributions[mfinal]["after_dou"],
                    get_function_from_pitch
                )
            else:
                distr = pitch_transition_probabilities["less_repetition"][mfinal][mgong][triple]

            last_contour = int(np.sign(
                relative_pitch_to_interval({"gong_lvlv": mode["mgong"]}, [triple[1], triple[2]])[0]))
            if last_contour == 0:  # if 0 contour, i.e., tone repetition, use the previous contour!
                last_contour = new_last_contour = int(np.sign(
                    relative_pitch_to_interval({"gong_lvlv": mode["mgong"]}, [triple[0], triple[1]])[
                        0]))
            distr = distr.get_conditioned_on_Q(
                contour_distributions[last_contour][tuple(retrograde_tones[idx-3:idx])],
                get_contour(triple)
            )
            distr_dict[triple] = distr.restrict_space(allowed_suzipu_list)

        return distr_dict

    def encode_triple(allowed_suzipu_list, triple):
        # e.g., 9 states means 9^2 * a + 9 * b + c for a triple (a, b, c)
        return len(allowed_suzipu_list) * len(allowed_suzipu_list) * allowed_suzipu_list.index(triple[0]) + len(
            allowed_suzipu_list) * allowed_suzipu_list.index(triple[1]) + allowed_suzipu_list.index(triple[2])

    # returns column vector
    def get_sparse_unit_vector(dim, indices):
        return sp.csr_matrix(([1]*len(indices), (indices, [0]*len(indices))), shape=(dim, 1))

    def transition_matrix_to_sparse(transition_matrix):
        allowed_suzipu_list = get_allowed_suzipu_list()

        rows, cols, values = [], [], []
        for start_state in itertools.product(*[allowed_suzipu_list for idx in range(3)]):
            start_state_encoding = encode_triple(allowed_suzipu_list, start_state)
            for final_state in transition_matrix[start_state].sample_space():
                state_probability = transition_matrix[start_state][final_state]
                final_state_encoding = encode_triple(allowed_suzipu_list, [start_state[1], start_state[2], final_state])
                if state_probability > 1e-10:
                    rows.append(start_state_encoding)
                    cols.append(final_state_encoding)
                    values.append(state_probability)

        return sp.csr_matrix((values, (rows, cols)), shape=(len(allowed_suzipu_list)*len(allowed_suzipu_list)*len(allowed_suzipu_list), len(allowed_suzipu_list)*len(allowed_suzipu_list)*len(allowed_suzipu_list)))

    def fill_by_using_matrix_products(gap_list):
        final_index = gap_list[-1]
        final_condition = None
        p = 1.

        # check if we have a final state we need to condition to
        final_distribution = None
        # case 1: we condition on full final triple
        if final_index < len(generated_piece) - 3 and None not in generated_piece[final_index + 1:final_index + 4]:
            final_condition = generated_piece[final_index + 3]
            three_after = get_transition_matrix(final_index + 3)
            for key in three_after.keys():
                try:
                    three_after[key] = three_after[key].get_conditioned_on_Q(
                        Distribution.from_dict(
                            {final_val: 1. if final_val == generated_piece[final_index + 3] else 0. for
                             final_val in allowed_suzipu_list}),
                        lambda final_val: final_val
                    )
                except ZeroDivisionError:
                    for final_key in three_after[key].distribution:
                        three_after[key].distribution[
                            final_key] = 0.  # this is no probability matrix, so if we calculate it later and the result is zero we can discard the result

            two_after = get_transition_matrix(final_index + 2)
            for key in two_after.keys():
                try:
                    two_after[key] = two_after[key].get_conditioned_on_Q(
                        Distribution.from_dict(
                            {final_val: 1. if final_val == generated_piece[final_index + 2] else 0. for
                             final_val in allowed_suzipu_list}),
                        lambda final_val: final_val
                    )
                except ZeroDivisionError:
                    for final_key in two_after[key].distribution:
                        two_after[key].distribution[
                            final_key] = 0.  # this is no probability matrix, so if we calculate it later and the result is zero we can discard the result

            one_after = get_transition_matrix(final_index + 1)
            for key in one_after.keys():
                try:
                    one_after[key] = one_after[key].get_conditioned_on_Q(
                        Distribution.from_dict(
                            {final_val: 1. if final_val == generated_piece[final_index + 1] else 0.
                             for final_val in allowed_suzipu_list}),
                        lambda final_val: final_val
                    )
                except ZeroDivisionError:
                    for final_key in one_after[key].distribution:
                        one_after[key].distribution[
                            final_key] = 0.  # this is no probability matrix, so if we calculate it later and the result is zero we can discard the result

            final_distribution = transition_matrix_to_sparse(one_after) * transition_matrix_to_sparse(
                two_after) * transition_matrix_to_sparse(three_after)
        # case 2: we condition on two final values
        elif final_index < len(generated_piece) - 2 and None not in generated_piece[final_index + 1:final_index + 3]:
            final_condition = generated_piece[final_index + 2]
            two_after = get_transition_matrix(final_index + 2)
            for key in two_after.keys():
                try:
                    two_after[key] = two_after[key].get_conditioned_on_Q(
                        Distribution.from_dict(
                            {final_val: 1. if final_val == generated_piece[final_index + 2] else 0. for
                             final_val in allowed_suzipu_list}),
                        lambda final_val: final_val
                    )
                except ZeroDivisionError:
                    for final_key in two_after[key].distribution:
                        two_after[key].distribution[
                            final_key] = 0.  # this is no probability matrix, so if we calculate it later and the result is zero we can discard the result

            one_after = get_transition_matrix(final_index + 1)
            for key in one_after.keys():
                try:
                    one_after[key] = one_after[key].get_conditioned_on_Q(
                        Distribution.from_dict(
                            {final_val: 1. if final_val == generated_piece[final_index + 1] else 0.
                             for final_val in allowed_suzipu_list}),
                        lambda final_val: final_val
                    )
                except ZeroDivisionError:
                    for final_key in one_after[key].distribution:
                        one_after[key].distribution[
                            final_key] = 0.  # this is no probability matrix, so if we calculate it later and the result is zero we can discard the result

            final_distribution = transition_matrix_to_sparse(one_after) * transition_matrix_to_sparse(two_after)

        # case 3: we only condition on one final value
        elif final_index < len(generated_piece) - 1 and None not in generated_piece[final_index + 1:final_index + 2]:
            final_condition = generated_piece[final_index + 1]
            final_distribution = get_transition_matrix(final_index + 1)
            for key in final_distribution.keys():
                try:
                    final_distribution[key] = final_distribution[key].get_conditioned_on_Q(
                        Distribution.from_dict(
                            {final_val: 1. if final_val == generated_piece[final_index + 1] else 0. for final_val in
                             allowed_suzipu_list}),
                        lambda final_val: final_val
                    )
                except ZeroDivisionError:
                    for final_key in final_distribution[key].distribution:
                        final_distribution[key].distribution[
                            final_key] = 0.  # this is no probability matrix, so if we calculate it later and the result is zero we can discard the result

            final_distribution = transition_matrix_to_sparse(final_distribution)

        else:
            idx = gap_list[0]
            while idx < final_index + 1:
                current_triple = tuple(generated_piece[idx - 3:idx])
                distr = get_current_pitch_distribution(idx=idx, current_triple=current_triple)
                generated_piece[idx] = distr.sample()
                p *= distr[generated_piece[idx]]
                idx += 1
            return p

        # we calculate the matrix products we need for the generation later
        precalculated_matrices = [final_distribution]
        for idx, gap_idx in enumerate(reversed(gap_list[1:])):  # we only need n-1 matrices (n is the len of the gap)
            precalculated_matrices.append(
                transition_matrix_to_sparse(get_transition_matrix(gap_idx)) * precalculated_matrices[-1])

        precalculated_matrices.reverse()  # the first pitch needs the largest product

        final_condition_triples = []
        for state in itertools.product(*[allowed_suzipu_list for idx in range(2)]):
            final_condition_triples.append(encode_triple(allowed_suzipu_list, [state[0], state[1], final_condition]))
        final_condition_vector = get_sparse_unit_vector(len(allowed_suzipu_list) ** 3, final_condition_triples)

        # now, sample a suitable pitch for each gap_idx to fill the gap!
        for idx, gap_idx in enumerate(gap_list):
            pitch_likelihoods = []
            for new_pitch in allowed_suzipu_list:
                start_encoding = encode_triple(allowed_suzipu_list, generated_piece[gap_idx - 3:gap_idx])
                new_state_encoding = encode_triple(allowed_suzipu_list, tuple(
                    [generated_piece[gap_idx - 2], generated_piece[gap_idx - 1], new_pitch]))
                pitch_likelihoods.append(
                    transition_matrix_to_sparse(get_transition_matrix(gap_idx))[start_encoding, new_state_encoding] * (
                                get_sparse_unit_vector(len(allowed_suzipu_list) ** 3,
                                                       [new_state_encoding]).transpose() * precalculated_matrices[
                                    idx] * final_condition_vector)[0, 0])
            pitch_distribution = Distribution(allowed_suzipu_list, pitch_likelihoods)
            generated_piece[gap_idx] = pitch_distribution.sample()
            p *= pitch_distribution[generated_piece[gap_idx]]
        return p

    def fill_by_sampling_all_trajectories(gap_list):
        probable_trajectories = []
        probable_likelihoods = []

        def recursive_traverse(current_trajectory, depth, start_probability):
            current_prob = start_probability

            if depth > 0:
                index = gap_list[depth - 1]
                current_prob *= \
                get_current_pitch_distribution(idx=index, current_triple=tuple(generated_piece[index - 3:index]))[
                    generated_piece[index]]

            if depth == len(gap_list):
                index = gap_list[depth - 1]
                if index < len(generated_piece) - 1 and None not in generated_piece[index - 2:index + 1] and \
                        generated_piece[
                            index + 1] is not None:  # also incorporate final state into probability calculation
                    try:
                        current_prob *= get_current_pitch_distribution(idx=index + 1, current_triple=tuple(
                            generated_piece[index - 2:index + 1]))[generated_piece[index + 1]]
                    except ZeroDivisionError:
                        current_prob = 0.
                if index < len(generated_piece) - 2 and None not in generated_piece[index - 1:index + 2] and \
                        generated_piece[
                            index + 2] is not None:  # also incorporate final state into probability calculation
                    try:
                        current_prob *= get_current_pitch_distribution(idx=index + 2, current_triple=tuple(
                            generated_piece[index - 1:index + 2]))[generated_piece[index + 2]]
                    except ZeroDivisionError:
                        current_prob = 0.
                if current_prob > 1e-10:
                    probable_trajectories.append(current_trajectory)
                    probable_likelihoods.append(current_prob)
            else:
                if current_prob > 1e-10:
                    for option in allowed_suzipu_list:
                        new_trajectory = current_trajectory + (option,)
                        generated_piece[gap_list[depth]] = option
                        recursive_traverse(new_trajectory, depth + 1, start_probability=current_prob)  # Recurse further

        # Start recursion with an empty trajectory and depth=0
        recursive_traverse((), 0, start_probability=1.)

        trajectory_distribution = Distribution(probable_trajectories, probable_likelihoods)
        current_trajectory = trajectory_distribution.sample()
        p = trajectory_distribution[current_trajectory]
        for i, fill_idx in enumerate(gap_list):
            generated_piece[fill_idx] = current_trajectory[i]

        return p

    if not "1" in repetition["repetition"] and not "r" in repetition["repetition"]: # no-repetition case
        stanza_probability = 1.

        # Generate second stanza
        idx = 0
        while idx < pian_idx:
            if idx == 0:  # ending of piece
                generated_piece[0:3] = pitch_initial_state
                idx += 3
            else:
                if retrograde_meter[
                    idx] == "pian":  # we reset the piece for the first stanza after generating the second one
                    break
                else:
                    current_triple = tuple(generated_piece[idx - 3:idx])
                    distr = get_current_pitch_distribution(idx=idx, current_triple=current_triple)
                    generated_piece[idx] = distr.sample()
                    stanza_probability *= distr[generated_piece[idx]]
                    idx += 1

        first_stanza_initial = get_pitch_initial_state(
            initial_state_distributions=initial_state_distributions,
            contour_distributions=contour_distributions,
            cipai=cipai,
            mode=mode,
            start_idx=pian_idx
        )

        description_string += first_stanza_initial["description_first"] + " "
        probability *= first_stanza_initial["probability"]
        first_stanza_initial = first_stanza_initial["initial_state"]
        # Generate first stanza
        while idx < len(generated_piece):
            if idx == pian_idx:  # ending of stanza
                generated_piece[pian_idx:pian_idx+3] = first_stanza_initial
                idx += 3
            else:
                current_triple = tuple(generated_piece[idx - 3:idx])
                distr = get_current_pitch_distribution(idx=idx, current_triple=current_triple)
                generated_piece[idx] = distr.sample()
                stanza_probability *= distr[generated_piece[idx]]
                idx += 1

        description_string += text_resources.EnglishTexts.both_stanzas_pitch_no_repetition_case.format(
            probability=stanza_probability
        ) + "\n\n"

        probability *= stanza_probability
    elif not "1" in repetition["repetition"]:  # inter-strophal repetitions case. the intra-strophal repetitions must be treated separately
        second_stanza_probability = 1.
        idx = 0
        while idx < pian_idx:
            ######### FIRST, GENERATE LAST STANZA!
            if idx == 0:  # ending of piece
                generated_piece[0:3] = pitch_initial_state
                idx += 3
            else:
                if retrograde_meter[idx] == "pian":  # we reset the piece for the first stanza after generating the second one
                    break
                else:
                    current_triple = tuple(generated_piece[idx - 3:idx])
                    distr = get_current_pitch_distribution(idx=idx, current_triple=current_triple)
                    generated_piece[idx] = distr.sample()
                    second_stanza_probability *= distr[generated_piece[idx]]
                    idx += 1

        probability *= second_stanza_probability
        description_string += text_resources.EnglishTexts.second_stanza_pitch_normal_case.format(
            probability=second_stanza_probability
        ) + " "

        # FILL IN REPEATED VALUES
        second_stanza_repetitions = retrograde_repetition[:pian_idx]
        first_stanza_repetitions = retrograde_repetition[pian_idx:]

        second_stanza_repetition_idxs = [i for i, x in enumerate(second_stanza_repetitions) if x == "r"]
        first_stanza_repetition_idxs = [i+pian_idx for i, x in enumerate(first_stanza_repetitions) if x == "r"]

        generated_piece[pian_idx] = generated_piece[0]  # final note of first stanza must be mode's final note
        for second_idx, first_idx in zip(second_stanza_repetition_idxs, first_stanza_repetition_idxs):
            generated_piece[first_idx] = generated_piece[second_idx]

        # NOW, CONNECT THE EMPTY BLANKS WITH THE REPEATED VALUES
        first_stanza_need_to_fill_idxs = [i+pian_idx for i, x in enumerate(first_stanza_repetitions) if x != "r" and i not in [0, 1, 2]]  # starting triple will be available later
        first_stanza_ending = retrograde_repetition[pian_idx:pian_idx+3]

        initial_prob = get_pitch_initial_state_probabilities(
            initial_state_distributions=initial_state_distributions,
            contour_distributions=contour_distributions,
            cipai=cipai,
            mode=mode
        )
        initial_dist = initial_prob
        # first one is always final note, so look at first two of ending triple of first stanza
        if first_stanza_ending[1:] == ["r", "r"]:  # [r, r] case, we must check if the ending is compatible
            if initial_prob[tuple(generated_piece[pian_idx:pian_idx+3])] < 1e-10:  # not permissible, let's sample a new ending triple
                try:  # do not repeat everything, try out [final, r, .]
                    initial_dist = initial_prob.get_conditioned_on_Q(
                        Distribution.from_dict({suzipu: 1. if suzipu==generated_piece[pian_idx+1] else 0. for suzipu in SimplePitchList}),
                        lambda tuple: tuple[1]
                    )
                except Exception:
                    try: # do not repeat everything, try out [final, ., r]
                        initial_dist = initial_prob.get_conditioned_on_Q(
                            Distribution.from_dict(
                                {suzipu: 1. if suzipu == generated_piece[pian_idx + 2] else 0. for suzipu in
                                 SimplePitchList}),
                            lambda tuple: tuple[2]
                        )
                    except Exception:  # if nothing works, resample completely
                        initial_dist = initial_prob
        elif first_stanza_ending[1] == "r": # [final, r, .] case
            try:
                initial_dist = initial_prob.get_conditioned_on_Q(
                    Distribution.from_dict(
                        {suzipu: 1. if suzipu == generated_piece[pian_idx + 1] else 0. for suzipu in SimplePitchList}),
                    lambda tuple: tuple[1]
                )
            except Exception:  # if it doesn't work, resample completely
                initial_dist = initial_prob
        elif first_stanza_ending[2] == "r": # [final, ., r]#
            try:
                initial_dist = initial_prob.get_conditioned_on_Q(
                    Distribution.from_dict(
                        {suzipu: 1. if suzipu == generated_piece[pian_idx + 2] else 0. for suzipu in
                         SimplePitchList}),
                    lambda tuple: tuple[2]
                )
            except Exception:  # if it doesn't work, resample completely
                initial_dist = initial_prob
        else: # [., ., .] case
            initial_dist = initial_prob

        generated_piece[pian_idx:pian_idx + 3] = initial_dist.sample()
        first_init_prob = initial_dist[tuple(generated_piece[pian_idx:pian_idx + 3])]
        probability *= first_init_prob

        description_string += text_resources.EnglishTexts.first_stanza_pitch_initial_state_without_repetition.format(
            probability=first_init_prob*100,
            cadential_phrase=generated_piece[pian_idx:pian_idx + 3]
        ) + " "

        # get all gaps
        first_stanza_idx_groups = [[first_stanza_need_to_fill_idxs[0]]]
        for idx in range(len(first_stanza_need_to_fill_idxs) - 1):
            if first_stanza_need_to_fill_idxs[idx+1] != first_stanza_need_to_fill_idxs[idx] + 1:
                first_stanza_idx_groups.append([])
            first_stanza_idx_groups[-1].append(first_stanza_need_to_fill_idxs[idx+1])

        first_stanza_probability = 1.
        for gap_list in first_stanza_idx_groups:
            if len(gap_list) <= 4:
                first_stanza_probability *= fill_by_sampling_all_trajectories(gap_list)
            else:
                first_stanza_probability *= fill_by_using_matrix_products(gap_list)

        description_string += text_resources.EnglishTexts.first_stanza_pitch_normal_case.format(
            probability=first_stanza_probability
        ) + "\n\n"

        probability *= first_stanza_probability

    else:  # Qiuxiaoyin case (ABAB/CBCD structure)
        second_stanza_indices = range(len(generated_piece))[0:pian_idx]
        first_stanza_indices = range(len(generated_piece))[pian_idx:]

        half_stanza_cd = second_stanza_indices[:len(second_stanza_indices) // 2]
        half_stanza_cb = second_stanza_indices[len(second_stanza_indices)//2:]

        # Generate last CD part
        cd_probability = 1.
        idx = 0
        while idx < len(half_stanza_cd):
            if idx == 0:  # ending of piece
                generated_piece[0:3] = pitch_initial_state
                idx += 3
            else:
                current_triple = tuple(generated_piece[idx - 3:idx])
                distr = get_current_pitch_distribution(idx=idx, current_triple=current_triple)
                generated_piece[idx] = distr.sample()
                cd_probability *= distr[generated_piece[idx]]
                idx += 1

        description_string += text_resources.EnglishTexts.second_stanza_pitch_intrastrophal_case_cd.format(
            probability = cd_probability
        ) + " "

        # Fill C in the CB part
        c_idxs_cb = [idx for idx in half_stanza_cb if retrograde_repetition[idx] == "3"]
        c_idxs_cd = [idx for idx in half_stanza_cd if retrograde_repetition[idx] == "3"]
        for c_idx_cb, c_idx_cd in zip(c_idxs_cb, c_idxs_cd):
            generated_piece[c_idx_cb] = generated_piece[c_idx_cd]

        # Generate B part in CB. By construction of the repetition, the B part is at least 3 syllables long,
        # so we generate a cadential phrase for the ending of the first stanza.

        first_stanza_initial = get_pitch_initial_state(
            initial_state_distributions=initial_state_distributions,
            contour_distributions=contour_distributions,
            cipai=cipai,
            mode=mode,
            start_idx=pian_idx
        )

        b_ending_probability = first_stanza_initial["probability"]
        first_stanza_initial = first_stanza_initial["initial_state"]

        all_cb_gaps = [idx for idx in half_stanza_cb if generated_piece[idx] is None]
        b_idxs_cb = [idx for idx in half_stanza_cb if retrograde_repetition[idx] == "2"]

        try:
            # B might be a bit scattered in the second stanza!
            generated_piece[b_idxs_cb[0]] = first_stanza_initial[0]
            generated_piece[b_idxs_cb[1]] = first_stanza_initial[1]
            generated_piece[b_idxs_cb[2]] = first_stanza_initial[2]
            del all_cb_gaps[all_cb_gaps.index(b_idxs_cb[0])]
            del all_cb_gaps[all_cb_gaps.index(b_idxs_cb[1])]
            del all_cb_gaps[all_cb_gaps.index(b_idxs_cb[2])]
        except Exception:
            pass

        # all_cb_gaps possibly consists of multiple contiguous areas that must be filled, so we separate them
        cb_gaps = []
        current_contiguous_gap = [all_cb_gaps[0]]
        for gap_idx in all_cb_gaps[1:]:
            if gap_idx == current_contiguous_gap[-1] + 1:
                current_contiguous_gap.append(gap_idx)
            else:
                cb_gaps.append(current_contiguous_gap)
                current_contiguous_gap = [gap_idx]
        cb_gaps.append(current_contiguous_gap)

        cb_probability = 1.
        for gap_list in cb_gaps:
            if pian_idx-1 in gap_list:  # this means we don't have to condition on future values
                # Generate last CD part
                idx = gap_list[0]
                while idx <= gap_list[-1]:
                    current_triple = tuple(generated_piece[idx - 3:idx])
                    distr = get_current_pitch_distribution(idx=idx, current_triple=current_triple)
                    generated_piece[idx] = distr.sample()
                    cb_probability *= distr[generated_piece[idx]]
                    idx += 1
            elif len(gap_list) <= 4:
                cb_probability *= fill_by_sampling_all_trajectories(gap_list)
            else:
                cb_probability *= fill_by_using_matrix_products(gap_list)

        description_string += text_resources.EnglishTexts.second_stanza_pitch_intrastrophal_case_cb.format(
            cadential_phrase=first_stanza_initial,
            cadential_probability=b_ending_probability*100,
            pitch_probability=cb_probability
        ) + " "

        # Now, fill in the b parts in the first stanza
        half_stanza_ab_last = first_stanza_indices[:len(first_stanza_indices) // 2]
        half_stanza_ab_first = first_stanza_indices[len(first_stanza_indices) // 2:]

        # Fill C in the CB part
        b_idxs_ab_first = [idx for idx in half_stanza_ab_first if retrograde_repetition[idx] == "2"]
        b_idxs_ab_last = [idx for idx in half_stanza_ab_last if retrograde_repetition[idx] == "2"]
        for b_idx_cb, b_idx_ab_first, b_idx_ab_last in zip(b_idxs_cb, b_idxs_ab_first, b_idxs_ab_last):
            generated_piece[b_idx_ab_first] = generated_piece[b_idx_cb]
            generated_piece[b_idx_ab_last] = generated_piece[b_idx_cb]

        # Then, generate the missing A part. Due to the high tonal compatibility, we only need to take into account
        # the later occurance. This is guaranteed to be a contiguous index set by construction of the repetition
        ab_probability = 1.
        a_idxs_ab_last = [idx for idx in half_stanza_ab_last if retrograde_repetition[idx] == "1"]
        if len(a_idxs_ab_last) <= 4:
            ab_probability *= fill_by_sampling_all_trajectories(a_idxs_ab_last)
        else:
            ab_probability *= fill_by_using_matrix_products(a_idxs_ab_last)

        # Finally, fill in the first A part

        a_idxs_ab_first = [idx for idx in half_stanza_ab_first if retrograde_repetition[idx] == "1"]
        for a_idx_first, a_idx_last in zip(a_idxs_ab_first, a_idxs_ab_last):
            generated_piece[a_idx_first] = generated_piece[a_idx_last]

        description_string += text_resources.EnglishTexts.second_stanza_pitch_intrastrophal_case_ab.format(
            probability=ab_probability
        ) + "\n\n"

        probability *= cd_probability * b_ending_probability * cb_probability * ab_probability


    generated_piece.reverse()
    return {
        "pitch_list": generated_piece,
        "description": description_string,
        "probability": probability
    }


def generate_secondary(initial_state_distributions, zhe_ye_distributions, secondary_group_distributions, secondary_transition_probabilities, cipai, repetition, pitch):
    cipai_meter = cipai["meter"]
    generated_piece = [None] * len(cipai_meter)

    retrograde_meter = cipai["meter"][::-1]
    retrograde_repetition = repetition[::-1]

    if np.random.rand() < 5/17:
        final_secondary = SuzipuSecondarySymbol.ADD_XIAO_ZHU
        final_p = 5/17
    else:
        final_secondary = SuzipuSecondarySymbol.ADD_DA_DUN
        final_p = 12/17

    description_string = EnglishTexts.second_stanza_secondary_final.format(final_secondary=final_secondary, probability=final_p*100) + " "

    secondary_initial_state = get_secondary_initial_state(
        initial_state_distributions=initial_state_distributions,
        final_secondary=final_secondary
    )
    description_string += secondary_initial_state["description"] + " "
    probability = secondary_initial_state["probability"]
    secondary_initial_state = secondary_initial_state["initial_state"]

    return {"description": description_string, "probability": probability, "secondary_list": [None]*len(cipai_meter)}

    pian_idx = retrograde_meter.index("pian")

    def get_contour(current_triple):
        def inner(pitch):
            try:
                val = int(np.sign(
                    relative_pitch_to_interval({"gong_lvlv": mode["mgong"]}, [current_triple[2], pitch])[
                        0]))
                if val == 0:  # 0 must not occur in the dicts, so choose some value
                    val = -1
                return val
            except TypeError:  # can occur when interval of nonexistent pitch is calculated
                return None
        return inner

    def get_current_pitch_distribution(idx, current_triple):
        # get the correct distribution of pitches for generating the idx-th position
        def get_function_from_pitch(pitch):
            return f_N_to_S(pitch, mode["mgong"])

        if idx in (len(generated_piece) - 1, pian_idx - 1):  # First note of each stanza
            next_pitch_distribution = pitch_transition_probabilities["less_repetition"][mfinal][mgong][current_triple]
            try:
                distr = next_pitch_distribution.get_conditioned_on_Q(
                    function_distributions[mfinal]["beginning"],
                    get_function_from_pitch
                )
            except ZeroDivisionError:
                distr = next_pitch_distribution
        elif cipai["meter"][::-1][idx] == "ju": # ju position
            next_pitch_distribution = pitch_transition_probabilities["less_repetition"][mfinal][mgong][current_triple]
            try:
                distr = next_pitch_distribution.get_conditioned_on_Q(
                    function_distributions[mfinal]["ju"],
                    get_function_from_pitch
                )
            except ZeroDivisionError:
                distr = next_pitch_distribution
        elif idx < len(generated_piece) - 1 and cipai["meter"][::-1][idx + 1] == "ju": # one after ju position
            next_pitch_distribution = pitch_transition_probabilities["base"][mfinal][mgong][current_triple]
            try:
                distr = next_pitch_distribution.get_conditioned_on_Q(
                    function_distributions[mfinal]["after_ju"],
                    get_function_from_pitch
                )
            except ZeroDivisionError:
                distr = next_pitch_distribution
        elif cipai["meter"][::-1][idx] == "dou": # dou position
            next_pitch_distribution = pitch_transition_probabilities["less_repetition"][mfinal][mgong][current_triple]
            try:
                distr = next_pitch_distribution.get_conditioned_on_Q(
                    function_distributions[mfinal]["dou"],
                    get_function_from_pitch
                )
            except ZeroDivisionError:
                distr = next_pitch_distribution
        elif idx < len(generated_piece) - 1 and cipai["meter"][::-1][idx + 1] == "dou": # one after dou position
            next_pitch_distribution = pitch_transition_probabilities["less_repetition"][mfinal][mgong][current_triple]
            try:
                distr = next_pitch_distribution.get_conditioned_on_Q(
                    function_distributions[mfinal]["after_dou"],
                    get_function_from_pitch
                )
            except ZeroDivisionError:
                distr = next_pitch_distribution
        else:
            distr = pitch_transition_probabilities["less_repetition"][mfinal][mgong][current_triple]

        last_contour = int(np.sign(relative_pitch_to_interval({"gong_lvlv": mode["mgong"]}, [current_triple[1], current_triple[2]])[0]))
        if last_contour == 0:  # if 0 contour, i.e., tone repetition, use the previous contour!
            last_contour = int(np.sign(relative_pitch_to_interval({"gong_lvlv": mode["mgong"]}, [current_triple[0], current_triple[1]])[0]))
        try:
            distr = distr.get_conditioned_on_Q(
                contour_distributions[last_contour][tuple(retrograde_tones[idx-3:idx])],
                get_contour(current_triple)
            )
        except ZeroDivisionError:
            pass

        return distr

    def get_allowed_suzipu_list():
        allowed_suzipu_list = get_suzipu_pitches(mode["mgong"])
        allowed_suzipu_list = [s for s in allowed_suzipu_list if s is not None]
        return allowed_suzipu_list

    # now, get all possibilities for the first stanza, calculate the likelihood and sample accordingly
    allowed_suzipu_list = get_allowed_suzipu_list()

    def get_transition_matrix(idx):
        def get_function_from_pitch(pitch):
            return f_N_to_S(pitch, mode["mgong"])

        distr_dict = {}

        allowed_suzipu_list = get_allowed_suzipu_list()

        for triple in itertools.product(*[allowed_suzipu_list for idx in range(3)]):
            if idx in (len(generated_piece) - 1, pian_idx - 1):  # First note of stanza
                next_pitch_distribution = pitch_transition_probabilities["less_repetition"][mfinal][mgong][triple]
                distr = next_pitch_distribution.get_conditioned_on_Q(
                    function_distributions[mfinal]["beginning"],
                    get_function_from_pitch
                )
            elif cipai["meter"][::-1][idx] == "ju": # Generate a ju
                next_pitch_distribution = pitch_transition_probabilities["less_repetition"][mfinal][mgong][triple]
                distr = next_pitch_distribution.get_conditioned_on_Q(
                    function_distributions[mfinal]["ju"],
                    get_function_from_pitch
                )
            elif idx < len(generated_piece) - 1 and cipai["meter"][::-1][idx + 1] == "ju": # Generate one after a ju
                next_pitch_distribution = pitch_transition_probabilities["base"][mfinal][mgong][triple]
                distr = next_pitch_distribution.get_conditioned_on_Q(
                    function_distributions[mfinal]["after_ju"],
                    get_function_from_pitch
                )
            elif cipai["meter"][::-1][idx] == "dou": # Generate dou
                next_pitch_distribution = pitch_transition_probabilities["less_repetition"][mfinal][mgong][triple]
                distr = next_pitch_distribution.get_conditioned_on_Q(
                    function_distributions[mfinal]["dou"],
                    get_function_from_pitch
                )
            elif idx < len(generated_piece) - 1 and cipai["meter"][::-1][idx + 1] == "dou": # Generate one after dou
                next_pitch_distribution = pitch_transition_probabilities["less_repetition"][mfinal][mgong][triple]
                distr = next_pitch_distribution.get_conditioned_on_Q(
                    function_distributions[mfinal]["after_dou"],
                    get_function_from_pitch
                )
            else:
                distr = pitch_transition_probabilities["less_repetition"][mfinal][mgong][triple]

            last_contour = int(np.sign(
                relative_pitch_to_interval({"gong_lvlv": mode["mgong"]}, [triple[1], triple[2]])[0]))
            if last_contour == 0:  # if 0 contour, i.e., tone repetition, use the previous contour!
                last_contour = new_last_contour = int(np.sign(
                    relative_pitch_to_interval({"gong_lvlv": mode["mgong"]}, [triple[0], triple[1]])[
                        0]))
            distr = distr.get_conditioned_on_Q(
                contour_distributions[last_contour][tuple(retrograde_tones[idx-3:idx])],
                get_contour(triple)
            )
            distr_dict[triple] = distr.restrict_space(allowed_suzipu_list)

        return distr_dict

    def encode_triple(allowed_suzipu_list, triple):
        # e.g., 9 states means 9^2 * a + 9 * b + c for a triple (a, b, c)
        return len(allowed_suzipu_list) * len(allowed_suzipu_list) * allowed_suzipu_list.index(triple[0]) + len(
            allowed_suzipu_list) * allowed_suzipu_list.index(triple[1]) + allowed_suzipu_list.index(triple[2])

    # returns column vector
    def get_sparse_unit_vector(dim, indices):
        return sp.csr_matrix(([1]*len(indices), (indices, [0]*len(indices))), shape=(dim, 1))

    def transition_matrix_to_sparse(transition_matrix):
        allowed_suzipu_list = get_allowed_suzipu_list()

        rows, cols, values = [], [], []
        for start_state in itertools.product(*[allowed_suzipu_list for idx in range(3)]):
            start_state_encoding = encode_triple(allowed_suzipu_list, start_state)
            for final_state in transition_matrix[start_state].sample_space():
                state_probability = transition_matrix[start_state][final_state]
                final_state_encoding = encode_triple(allowed_suzipu_list, [start_state[1], start_state[2], final_state])
                if state_probability > 1e-10:
                    rows.append(start_state_encoding)
                    cols.append(final_state_encoding)
                    values.append(state_probability)

        return sp.csr_matrix((values, (rows, cols)), shape=(len(allowed_suzipu_list)*len(allowed_suzipu_list)*len(allowed_suzipu_list), len(allowed_suzipu_list)*len(allowed_suzipu_list)*len(allowed_suzipu_list)))

    def fill_by_using_matrix_products(gap_list):
        final_index = gap_list[-1]
        final_condition = None
        p = 1.

        # check if we have a final state we need to condition to
        final_distribution = None
        # case 1: we condition on full final triple
        if final_index < len(generated_piece) - 3 and None not in generated_piece[final_index + 1:final_index + 4]:
            final_condition = generated_piece[final_index + 3]
            three_after = get_transition_matrix(final_index + 3)
            for key in three_after.keys():
                try:
                    three_after[key] = three_after[key].get_conditioned_on_Q(
                        Distribution.from_dict(
                            {final_val: 1. if final_val == generated_piece[final_index + 3] else 0. for
                             final_val in allowed_suzipu_list}),
                        lambda final_val: final_val
                    )
                except ZeroDivisionError:
                    for final_key in three_after[key].distribution:
                        three_after[key].distribution[
                            final_key] = 0.  # this is no probability matrix, so if we calculate it later and the result is zero we can discard the result

            two_after = get_transition_matrix(final_index + 2)
            for key in two_after.keys():
                try:
                    two_after[key] = two_after[key].get_conditioned_on_Q(
                        Distribution.from_dict(
                            {final_val: 1. if final_val == generated_piece[final_index + 2] else 0. for
                             final_val in allowed_suzipu_list}),
                        lambda final_val: final_val
                    )
                except ZeroDivisionError:
                    for final_key in two_after[key].distribution:
                        two_after[key].distribution[
                            final_key] = 0.  # this is no probability matrix, so if we calculate it later and the result is zero we can discard the result

            one_after = get_transition_matrix(final_index + 1)
            for key in one_after.keys():
                try:
                    one_after[key] = one_after[key].get_conditioned_on_Q(
                        Distribution.from_dict(
                            {final_val: 1. if final_val == generated_piece[final_index + 1] else 0.
                             for final_val in allowed_suzipu_list}),
                        lambda final_val: final_val
                    )
                except ZeroDivisionError:
                    for final_key in one_after[key].distribution:
                        one_after[key].distribution[
                            final_key] = 0.  # this is no probability matrix, so if we calculate it later and the result is zero we can discard the result

            final_distribution = transition_matrix_to_sparse(one_after) * transition_matrix_to_sparse(
                two_after) * transition_matrix_to_sparse(three_after)
        # case 2: we condition on two final values
        elif final_index < len(generated_piece) - 2 and None not in generated_piece[final_index + 1:final_index + 3]:
            final_condition = generated_piece[final_index + 2]
            two_after = get_transition_matrix(final_index + 2)
            for key in two_after.keys():
                try:
                    two_after[key] = two_after[key].get_conditioned_on_Q(
                        Distribution.from_dict(
                            {final_val: 1. if final_val == generated_piece[final_index + 2] else 0. for
                             final_val in allowed_suzipu_list}),
                        lambda final_val: final_val
                    )
                except ZeroDivisionError:
                    for final_key in two_after[key].distribution:
                        two_after[key].distribution[
                            final_key] = 0.  # this is no probability matrix, so if we calculate it later and the result is zero we can discard the result

            one_after = get_transition_matrix(final_index + 1)
            for key in one_after.keys():
                try:
                    one_after[key] = one_after[key].get_conditioned_on_Q(
                        Distribution.from_dict(
                            {final_val: 1. if final_val == generated_piece[final_index + 1] else 0.
                             for final_val in allowed_suzipu_list}),
                        lambda final_val: final_val
                    )
                except ZeroDivisionError:
                    for final_key in one_after[key].distribution:
                        one_after[key].distribution[
                            final_key] = 0.  # this is no probability matrix, so if we calculate it later and the result is zero we can discard the result

            final_distribution = transition_matrix_to_sparse(one_after) * transition_matrix_to_sparse(two_after)

        # case 3: we only condition on one final value
        elif final_index < len(generated_piece) - 1 and None not in generated_piece[final_index + 1:final_index + 2]:
            final_condition = generated_piece[final_index + 1]
            final_distribution = get_transition_matrix(final_index + 1)
            for key in final_distribution.keys():
                try:
                    final_distribution[key] = final_distribution[key].get_conditioned_on_Q(
                        Distribution.from_dict(
                            {final_val: 1. if final_val == generated_piece[final_index + 1] else 0. for final_val in
                             allowed_suzipu_list}),
                        lambda final_val: final_val
                    )
                except ZeroDivisionError:
                    for final_key in final_distribution[key].distribution:
                        final_distribution[key].distribution[
                            final_key] = 0.  # this is no probability matrix, so if we calculate it later and the result is zero we can discard the result

            final_distribution = transition_matrix_to_sparse(final_distribution)

        else:
            idx = gap_list[0]
            while idx < final_index + 1:
                current_triple = tuple(generated_piece[idx - 3:idx])
                distr = get_current_pitch_distribution(idx=idx, current_triple=current_triple)
                generated_piece[idx] = distr.sample()
                p *= distr[generated_piece[idx]]
                idx += 1
            return p

        # we calculate the matrix products we need for the generation later
        precalculated_matrices = [final_distribution]
        for idx, gap_idx in enumerate(reversed(gap_list[1:])):  # we only need n-1 matrices (n is the len of the gap)
            precalculated_matrices.append(
                transition_matrix_to_sparse(get_transition_matrix(gap_idx)) * precalculated_matrices[-1])

        precalculated_matrices.reverse()  # the first pitch needs the largest product

        final_condition_triples = []
        for state in itertools.product(*[allowed_suzipu_list for idx in range(2)]):
            final_condition_triples.append(encode_triple(allowed_suzipu_list, [state[0], state[1], final_condition]))
        final_condition_vector = get_sparse_unit_vector(len(allowed_suzipu_list) ** 3, final_condition_triples)

        # now, sample a suitable pitch for each gap_idx to fill the gap!
        for idx, gap_idx in enumerate(gap_list):
            pitch_likelihoods = []
            for new_pitch in allowed_suzipu_list:
                start_encoding = encode_triple(allowed_suzipu_list, generated_piece[gap_idx - 3:gap_idx])
                new_state_encoding = encode_triple(allowed_suzipu_list, tuple(
                    [generated_piece[gap_idx - 2], generated_piece[gap_idx - 1], new_pitch]))
                pitch_likelihoods.append(
                    transition_matrix_to_sparse(get_transition_matrix(gap_idx))[start_encoding, new_state_encoding] * (
                                get_sparse_unit_vector(len(allowed_suzipu_list) ** 3,
                                                       [new_state_encoding]).transpose() * precalculated_matrices[
                                    idx] * final_condition_vector)[0, 0])
            pitch_distribution = Distribution(allowed_suzipu_list, pitch_likelihoods)
            generated_piece[gap_idx] = pitch_distribution.sample()
            p *= pitch_distribution[generated_piece[gap_idx]]
        return p

    def fill_by_sampling_all_trajectories(gap_list):
        probable_trajectories = []
        probable_likelihoods = []

        def recursive_traverse(current_trajectory, depth, start_probability):
            current_prob = start_probability

            if depth > 0:
                index = gap_list[depth - 1]
                current_prob *= \
                get_current_pitch_distribution(idx=index, current_triple=tuple(generated_piece[index - 3:index]))[
                    generated_piece[index]]

            if depth == len(gap_list):
                index = gap_list[depth - 1]
                if index < len(generated_piece) - 1 and None not in generated_piece[index - 2:index + 1] and \
                        generated_piece[
                            index + 1] is not None:  # also incorporate final state into probability calculation
                    try:
                        current_prob *= get_current_pitch_distribution(idx=index + 1, current_triple=tuple(
                            generated_piece[index - 2:index + 1]))[generated_piece[index + 1]]
                    except ZeroDivisionError:
                        current_prob = 0.
                if index < len(generated_piece) - 2 and None not in generated_piece[index - 1:index + 2] and \
                        generated_piece[
                            index + 2] is not None:  # also incorporate final state into probability calculation
                    try:
                        current_prob *= get_current_pitch_distribution(idx=index + 2, current_triple=tuple(
                            generated_piece[index - 1:index + 2]))[generated_piece[index + 2]]
                    except ZeroDivisionError:
                        current_prob = 0.
                if current_prob > 1e-10:
                    probable_trajectories.append(current_trajectory)
                    probable_likelihoods.append(current_prob)
            else:
                if current_prob > 1e-10:
                    for option in allowed_suzipu_list:
                        new_trajectory = current_trajectory + (option,)
                        generated_piece[gap_list[depth]] = option
                        recursive_traverse(new_trajectory, depth + 1, start_probability=current_prob)  # Recurse further

        # Start recursion with an empty trajectory and depth=0
        recursive_traverse((), 0, start_probability=1.)

        trajectory_distribution = Distribution(probable_trajectories, probable_likelihoods)
        current_trajectory = trajectory_distribution.sample()
        p = trajectory_distribution[current_trajectory]
        for i, fill_idx in enumerate(gap_list):
            generated_piece[fill_idx] = current_trajectory[i]

        return p

    if not "1" in repetition["repetition"] and not "r" in repetition["repetition"]: # no-repetition case
        stanza_probability = 1.

        # Generate second stanza
        idx = 0
        while idx < pian_idx:
            if idx == 0:  # ending of piece
                generated_piece[0:3] = secondary_initial_state
                idx += 3
            else:
                if retrograde_meter[
                    idx] == "pian":  # we reset the piece for the first stanza after generating the second one
                    break
                else:
                    current_triple = tuple(generated_piece[idx - 3:idx])
                    distr = get_current_pitch_distribution(idx=idx, current_triple=current_triple)
                    generated_piece[idx] = distr.sample()
                    stanza_probability *= distr[generated_piece[idx]]
                    idx += 1

        first_stanza_initial = get_pitch_initial_state(
            initial_state_distributions=initial_state_distributions,
            contour_distributions=contour_distributions,
            cipai=cipai,
            mode=mode,
            start_idx=pian_idx
        )

        description_string += first_stanza_initial["description_first"] + " "
        probability *= first_stanza_initial["probability"]
        first_stanza_initial = first_stanza_initial["initial_state"]
        # Generate first stanza
        while idx < len(generated_piece):
            if idx == pian_idx:  # ending of stanza
                generated_piece[pian_idx:pian_idx+3] = first_stanza_initial
                idx += 3
            else:
                current_triple = tuple(generated_piece[idx - 3:idx])
                distr = get_current_pitch_distribution(idx=idx, current_triple=current_triple)
                generated_piece[idx] = distr.sample()
                stanza_probability *= distr[generated_piece[idx]]
                idx += 1

        description_string += text_resources.EnglishTexts.both_stanzas_pitch_no_repetition_case.format(
            probability=stanza_probability
        ) + "\n\n"

        probability *= stanza_probability
    elif not "1" in repetition["repetition"]:  # inter-strophal repetitions case. the intra-strophal repetitions must be treated separately
        second_stanza_probability = 1.
        idx = 0
        while idx < pian_idx:
            ######### FIRST, GENERATE LAST STANZA!
            if idx == 0:  # ending of piece
                generated_piece[0:3] = secondary_initial_state
                idx += 3
            else:
                if retrograde_meter[idx] == "pian":  # we reset the piece for the first stanza after generating the second one
                    break
                else:
                    current_triple = tuple(generated_piece[idx - 3:idx])
                    distr = get_current_pitch_distribution(idx=idx, current_triple=current_triple)
                    generated_piece[idx] = distr.sample()
                    second_stanza_probability *= distr[generated_piece[idx]]
                    idx += 1

        probability *= second_stanza_probability
        description_string += text_resources.EnglishTexts.second_stanza_pitch_normal_case.format(
            probability=second_stanza_probability
        ) + " "

        # FILL IN REPEATED VALUES
        second_stanza_repetitions = retrograde_repetition[:pian_idx]
        first_stanza_repetitions = retrograde_repetition[pian_idx:]

        second_stanza_repetition_idxs = [i for i, x in enumerate(second_stanza_repetitions) if x == "r"]
        first_stanza_repetition_idxs = [i+pian_idx for i, x in enumerate(first_stanza_repetitions) if x == "r"]

        generated_piece[pian_idx] = generated_piece[0]  # final note of first stanza must be mode's final note
        for second_idx, first_idx in zip(second_stanza_repetition_idxs, first_stanza_repetition_idxs):
            generated_piece[first_idx] = generated_piece[second_idx]

        # NOW, CONNECT THE EMPTY BLANKS WITH THE REPEATED VALUES
        first_stanza_need_to_fill_idxs = [i+pian_idx for i, x in enumerate(first_stanza_repetitions) if x != "r" and i not in [0, 1, 2]]  # starting triple will be available later
        first_stanza_ending = retrograde_repetition[pian_idx:pian_idx+3]

        initial_prob = get_pitch_initial_state_probabilities(
            initial_state_distributions=initial_state_distributions,
            contour_distributions=contour_distributions,
            cipai=cipai,
            mode=mode
        )
        initial_dist = initial_prob
        # first one is always final note, so look at first two of ending triple of first stanza
        if first_stanza_ending[1:] == ["r", "r"]:  # [r, r] case, we must check if the ending is compatible
            if initial_prob[tuple(generated_piece[pian_idx:pian_idx+3])] < 1e-10:  # not permissible, let's sample a new ending triple
                try:  # do not repeat everything, try out [final, r, .]
                    initial_dist = initial_prob.get_conditioned_on_Q(
                        Distribution.from_dict({suzipu: 1. if suzipu==generated_piece[pian_idx+1] else 0. for suzipu in SimplePitchList}),
                        lambda tuple: tuple[1]
                    )
                except Exception:
                    try: # do not repeat everything, try out [final, ., r]
                        initial_dist = initial_prob.get_conditioned_on_Q(
                            Distribution.from_dict(
                                {suzipu: 1. if suzipu == generated_piece[pian_idx + 2] else 0. for suzipu in
                                 SimplePitchList}),
                            lambda tuple: tuple[2]
                        )
                    except Exception:  # if nothing works, resample completely
                        initial_dist = initial_prob
        elif first_stanza_ending[1] == "r": # [final, r, .] case
            try:
                initial_dist = initial_prob.get_conditioned_on_Q(
                    Distribution.from_dict(
                        {suzipu: 1. if suzipu == generated_piece[pian_idx + 1] else 0. for suzipu in SimplePitchList}),
                    lambda tuple: tuple[1]
                )
            except Exception:  # if it doesn't work, resample completely
                initial_dist = initial_prob
        elif first_stanza_ending[2] == "r": # [final, ., r]#
            try:
                initial_dist = initial_prob.get_conditioned_on_Q(
                    Distribution.from_dict(
                        {suzipu: 1. if suzipu == generated_piece[pian_idx + 2] else 0. for suzipu in
                         SimplePitchList}),
                    lambda tuple: tuple[2]
                )
            except Exception:  # if it doesn't work, resample completely
                initial_dist = initial_prob
        else: # [., ., .] case
            initial_dist = initial_prob

        generated_piece[pian_idx:pian_idx + 3] = initial_dist.sample()
        first_init_prob = initial_dist[tuple(generated_piece[pian_idx:pian_idx + 3])]
        probability *= first_init_prob

        description_string += text_resources.EnglishTexts.first_stanza_pitch_initial_state_without_repetition.format(
            probability=first_init_prob*100,
            cadential_phrase=generated_piece[pian_idx:pian_idx + 3]
        ) + " "

        # get all gaps
        first_stanza_idx_groups = [[first_stanza_need_to_fill_idxs[0]]]
        for idx in range(len(first_stanza_need_to_fill_idxs) - 1):
            if first_stanza_need_to_fill_idxs[idx+1] != first_stanza_need_to_fill_idxs[idx] + 1:
                first_stanza_idx_groups.append([])
            first_stanza_idx_groups[-1].append(first_stanza_need_to_fill_idxs[idx+1])

        first_stanza_probability = 1.
        for gap_list in first_stanza_idx_groups:
            if len(gap_list) <= 4:
                first_stanza_probability *= fill_by_sampling_all_trajectories(gap_list)
            else:
                first_stanza_probability *= fill_by_using_matrix_products(gap_list)

        description_string += text_resources.EnglishTexts.first_stanza_pitch_normal_case.format(
            probability=first_stanza_probability
        ) + "\n\n"

        probability *= first_stanza_probability

    else:  # Qiuxiaoyin case (ABAB/CBCD structure)
        second_stanza_indices = range(len(generated_piece))[0:pian_idx]
        first_stanza_indices = range(len(generated_piece))[pian_idx:]

        half_stanza_cd = second_stanza_indices[:len(second_stanza_indices) // 2]
        half_stanza_cb = second_stanza_indices[len(second_stanza_indices)//2:]

        # Generate last CD part
        cd_probability = 1.
        idx = 0
        while idx < len(half_stanza_cd):
            if idx == 0:  # ending of piece
                generated_piece[0:3] = secondary_initial_state
                idx += 3
            else:
                current_triple = tuple(generated_piece[idx - 3:idx])
                distr = get_current_pitch_distribution(idx=idx, current_triple=current_triple)
                generated_piece[idx] = distr.sample()
                cd_probability *= distr[generated_piece[idx]]
                idx += 1

        description_string += text_resources.EnglishTexts.second_stanza_pitch_intrastrophal_case_cd.format(
            probability = cd_probability
        ) + " "

        # Fill C in the CB part
        c_idxs_cb = [idx for idx in half_stanza_cb if retrograde_repetition[idx] == "3"]
        c_idxs_cd = [idx for idx in half_stanza_cd if retrograde_repetition[idx] == "3"]
        for c_idx_cb, c_idx_cd in zip(c_idxs_cb, c_idxs_cd):
            generated_piece[c_idx_cb] = generated_piece[c_idx_cd]

        # Generate B part in CB. By construction of the repetition, the B part is at least 3 syllables long,
        # so we generate a cadential phrase for the ending of the first stanza.

        first_stanza_initial = get_pitch_initial_state(
            initial_state_distributions=initial_state_distributions,
            contour_distributions=contour_distributions,
            cipai=cipai,
            mode=mode,
            start_idx=pian_idx
        )

        b_ending_probability = first_stanza_initial["probability"]
        first_stanza_initial = first_stanza_initial["initial_state"]

        all_cb_gaps = [idx for idx in half_stanza_cb if generated_piece[idx] is None]
        b_idxs_cb = [idx for idx in half_stanza_cb if retrograde_repetition[idx] == "2"]

        try:
            # B might be a bit scattered in the second stanza!
            generated_piece[b_idxs_cb[0]] = first_stanza_initial[0]
            generated_piece[b_idxs_cb[1]] = first_stanza_initial[1]
            generated_piece[b_idxs_cb[2]] = first_stanza_initial[2]
            del all_cb_gaps[all_cb_gaps.index(b_idxs_cb[0])]
            del all_cb_gaps[all_cb_gaps.index(b_idxs_cb[1])]
            del all_cb_gaps[all_cb_gaps.index(b_idxs_cb[2])]
        except Exception:
            pass

        # all_cb_gaps possibly consists of multiple contiguous areas that must be filled, so we separate them
        cb_gaps = []
        current_contiguous_gap = [all_cb_gaps[0]]
        for gap_idx in all_cb_gaps[1:]:
            if gap_idx == current_contiguous_gap[-1] + 1:
                current_contiguous_gap.append(gap_idx)
            else:
                cb_gaps.append(current_contiguous_gap)
                current_contiguous_gap = [gap_idx]
        cb_gaps.append(current_contiguous_gap)

        cb_probability = 1.
        for gap_list in cb_gaps:
            if pian_idx-1 in gap_list:  # this means we don't have to condition on future values
                # Generate last CD part
                idx = gap_list[0]
                while idx <= gap_list[-1]:
                    current_triple = tuple(generated_piece[idx - 3:idx])
                    distr = get_current_pitch_distribution(idx=idx, current_triple=current_triple)
                    generated_piece[idx] = distr.sample()
                    cb_probability *= distr[generated_piece[idx]]
                    idx += 1
            elif len(gap_list) <= 4:
                cb_probability *= fill_by_sampling_all_trajectories(gap_list)
            else:
                cb_probability *= fill_by_using_matrix_products(gap_list)

        description_string += text_resources.EnglishTexts.second_stanza_pitch_intrastrophal_case_cb.format(
            cadential_phrase=first_stanza_initial,
            cadential_probability=b_ending_probability*100,
            pitch_probability=cb_probability
        ) + " "

        # Now, fill in the b parts in the first stanza
        half_stanza_ab_last = first_stanza_indices[:len(first_stanza_indices) // 2]
        half_stanza_ab_first = first_stanza_indices[len(first_stanza_indices) // 2:]

        # Fill C in the CB part
        b_idxs_ab_first = [idx for idx in half_stanza_ab_first if retrograde_repetition[idx] == "2"]
        b_idxs_ab_last = [idx for idx in half_stanza_ab_last if retrograde_repetition[idx] == "2"]
        for b_idx_cb, b_idx_ab_first, b_idx_ab_last in zip(b_idxs_cb, b_idxs_ab_first, b_idxs_ab_last):
            generated_piece[b_idx_ab_first] = generated_piece[b_idx_cb]
            generated_piece[b_idx_ab_last] = generated_piece[b_idx_cb]

        # Then, generate the missing A part. Due to the high tonal compatibility, we only need to take into account
        # the later occurance. This is guaranteed to be a contiguous index set by construction of the repetition
        ab_probability = 1.
        a_idxs_ab_last = [idx for idx in half_stanza_ab_last if retrograde_repetition[idx] == "1"]
        if len(a_idxs_ab_last) <= 4:
            ab_probability *= fill_by_sampling_all_trajectories(a_idxs_ab_last)
        else:
            ab_probability *= fill_by_using_matrix_products(a_idxs_ab_last)

        # Finally, fill in the first A part

        a_idxs_ab_first = [idx for idx in half_stanza_ab_first if retrograde_repetition[idx] == "1"]
        for a_idx_first, a_idx_last in zip(a_idxs_ab_first, a_idxs_ab_last):
            generated_piece[a_idx_first] = generated_piece[a_idx_last]

        description_string += text_resources.EnglishTexts.second_stanza_pitch_intrastrophal_case_ab.format(
            probability=ab_probability
        ) + "\n\n"

        probability *= cd_probability * b_ending_probability * cb_probability * ab_probability


    generated_piece.reverse()
    return {
        "pitch_list": generated_piece,
        "description": description_string,
        "probability": probability
    }