import copy
import dataclasses
import itertools
import random

from distribution import *
from filter_and_conversion import filter_by_final, relative_pitch_to_contour, f_S_to_N, add_mgong, get_suzipu_pitches, \
    f_N_to_S, relative_pitch_to_interval
from music import GongdiaoStep, Lvlv, SimpleSuzipuList
from text_resources import EnglishTexts


def get_k_grams(base_list, k, ngram_property=None):
    ngrams = []
    for idx in range(len(base_list)-(k-1)):
        triple = tuple(base_list[idx:idx+k])
        if ngram_property is None or (ngram_property and ngram_property(triple)):
            ngrams.append(triple)
    return ngrams


def get_contour_distributions(pieces):
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


def get_function_distributions(pieces):
    def get_beginning_function_distribution(pieces, mfinal):
        traverse_pieces = filter_by_final(pieces, mfinal)
        count_list = []
        for P in traverse_pieces:
            piece_count_list = []
            for idx, (f, meter) in enumerate(zip(P["music"]["function"], P["cipai"]["meter"])):
                if meter == "pian" or idx == 0:
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
                if idx < len(P["music"]["function"]) - 1 and P["cipai"]["meter"][idx + 1] in ("ju", "pian"):
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
                if idx < len(P["music"]["function"]) - 1 and P["cipai"]["meter"][idx + 1] == "dou":
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


def igram(P, k):
    return get_k_grams(P["music"]["retrograde_interval"], k - 1)


def get_cadential_phrases(P):
    igrams = igram(P, 3)
    pian_idx = P["retrograde_cipai"]["meter"].index("pian")
    phrase_list = [igrams[0], igrams[pian_idx]]  # final cadential phrase and pian cadential phrase
    unq = np.unique(phrase_list, axis=0)  # only take the pian cadential phrase if it is not the same as the final one
    return list(unq)


def pitch_triple_to_contour(mgong_0):
    def inner(pitch_list):
        return tuple(relative_pitch_to_contour({"gong_lvlv": mgong_0}, pitch_list))

    return inner


def get_initial_state_distributions(pieces):
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
                I_0 += get_cadential_phrases(p)
            I_1 = []
            for p in P_1:
                I_1 += get_cadential_phrases(p)

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


def get_pitch_initial_state(initial_state_distributions, contour_distributions, cipai, mode):
    mgong_0, mfinal_0 = mode["mgong"], mode["mfinal"]

    retrograde_cipai = cipai["tones"][::-1]

    tone_dependent_distribution = initial_state_distributions[mgong_0][mfinal_0].get_conditioned_on_Q(contour_distributions["all"][tuple(retrograde_cipai[0:3])],
                                                                                                      pitch_triple_to_contour(mgong_0))

    initial_state = tone_dependent_distribution.sample()

    return_string = EnglishTexts.initial_state.format(final_note=initial_state[0],
                                                      cadential_phrase=[str(pitch) for pitch in initial_state][::-1],
                                                      probability=tone_dependent_distribution[initial_state] * 100)

    return {"initial_state": [str(entry) for entry in initial_state],
            "description": return_string,
            "probability": tone_dependent_distribution[initial_state]}


def get_pitch_initial_state_probabilities(initial_state_distributions, contour_distributions, cipai, mode):
    mgong_0, mfinal_0 = mode["mgong"], mode["mfinal"]

    retrograde_cipai = cipai["tones"][::-1]

    tone_dependent_distribution = initial_state_distributions[mgong_0][mfinal_0].get_conditioned_on_Q(contour_distributions["all"][tuple(retrograde_cipai[0:3])],
                                                                                                      pitch_triple_to_contour(mgong_0))
    return tone_dependent_distribution


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


def get_inverted_n_step_markov_chain(base_pieces, gong_lvlv, n):
    total_list = []
    for piece in base_pieces:
        total_list += get_interval_n_grams(gong_lvlv, piece["music"], n=n)
    markov_chain = {}
    for combination in itertools.product(*[SimpleSuzipuList for idx in range(n)]):
        markov_chain[combination] = {}
        for symbol in SimpleSuzipuList:
            markov_chain[combination][symbol] = 0

    # add absolute count
    for ngram in total_list:
        ngram.reverse()  # we must consider the inverted ngrams since we generate starting from the end
        triple = tuple(ngram[0:n])
        next_step = ngram[n]

        markov_chain[triple][next_step] += 1

    zero_rows = []
    # scale to probability vector
    for triple in itertools.product(*[SimpleSuzipuList for idx in range(n)]):
        row_sum = sum(markov_chain[triple].values())
        if row_sum < 0.01:
            zero_rows.append(triple)
        else:
            for symbol in SimpleSuzipuList:
                markov_chain[triple][symbol] /= row_sum

    # incorporate n-1 step probabilites with 5%, except for zero rows, which take the n-1 steps completely
    if n > 1:
        n_minus_one_chain = get_inverted_n_step_markov_chain(base_pieces, gong_lvlv, n - 1)

        for triple in itertools.product(*[SimpleSuzipuList for idx in range(n)]):
            if triple in zero_rows:
                for symbol in SimpleSuzipuList:
                    markov_chain[triple][symbol] = n_minus_one_chain[triple[1:n]][symbol]
            else:
                for symbol in SimpleSuzipuList:
                    markov_chain[triple][symbol] = 0.95 * markov_chain[triple][symbol] + 0.05 * \
                                                   n_minus_one_chain[triple[1:n]][symbol]
                # scale to probability vector
                row_sum = sum(markov_chain[triple].values())
                for symbol in SimpleSuzipuList:
                    markov_chain[triple][symbol] /= row_sum
    else:  # if zero, choose some random pitch (belonging to the mode!) to fill the empty row
        for zero_row in zero_rows:
            rands = random.sample([pitch for pitch in get_suzipu_pitches(gong_lvlv=gong_lvlv) if pitch], 3)
            for r in rands:
                markov_chain[zero_row][r] = 1. / len(rands)

    return markov_chain


def get_pitch_transition_probabilities(pieces):
    n = 3
    base_probabilities = {}
    for final in (GongdiaoStep.GONG, GongdiaoStep.SHANG, GongdiaoStep.YU):
        base_probabilities[final] = {}
        for gong_lvlv in dataclasses.astuple(Lvlv()):
            markov_chain = get_inverted_n_step_markov_chain(
                filter_by_final(pieces["all"], final),
                n=n,
                gong_lvlv=gong_lvlv
            )
            base_probabilities[final][gong_lvlv] = {key: Distribution.from_dict(markov_chain[key]) for key in markov_chain.keys()}
            for triple in itertools.product(*[SimpleSuzipuList for idx in range(n)]):
                row_sum = sum(base_probabilities[final][gong_lvlv][triple].probabilities())
                if abs(row_sum - 1) > 1e-10:
                    raise ValueError("The generation process yielded no valid Markov chain. Some row sum is not equal to 1.")

    no_repetition_probabilities = copy.deepcopy(base_probabilities)
    for final in (GongdiaoStep.GONG, GongdiaoStep.SHANG, GongdiaoStep.YU):
        for gong_lvlv in dataclasses.astuple(Lvlv()):
            current_keys = no_repetition_probabilities[final][gong_lvlv].keys()
            for key in current_keys:
                current_distribution = no_repetition_probabilities[final][gong_lvlv][key].distribution
                current_distribution[key[2]] = 0
                no_repetition_probabilities[final][gong_lvlv][key].from_dict(current_distribution)

    return {
        "base": base_probabilities,
        "no_repetition": no_repetition_probabilities
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
    description_string = pitch_initial_state["description"] + "\n\n"
    probability = pitch_initial_state["probability"]
    pitch_initial_state = pitch_initial_state["initial_state"]

    pian_idx = retrograde_meter.index("pian")

    def get_contour(pitch):
        try:
            val = int(np.sign(
                relative_pitch_to_interval({"gong_lvlv": mode["mgong"]}, [current_triple[2], pitch])[
                    0]))
            if val == 0:  # 0 must not occur in the dicts, so choose some value
                val = -1
            return val
        except TypeError:  # can occur when interval of nonexistent pitch is calculated
            return None

    def get_current_pitch_distribution(idx, current_triple):
        def get_function_from_pitch(pitch):
            return f_N_to_S(pitch, mode["mgong"])

        if idx in (len(generated_piece) - 1, pian_idx - 1):  # First note of stanza
            next_pitch_distribution = pitch_transition_probabilities["no_repetition"][mfinal][mgong][current_triple]
            distr = next_pitch_distribution.get_conditioned_on_Q(
                function_distributions[mfinal]["beginning"],
                get_function_from_pitch
            )
        elif cipai["meter"][::-1][idx] == "ju":
            next_pitch_distribution = pitch_transition_probabilities["no_repetition"][mfinal][mgong][current_triple]
            distr = next_pitch_distribution.get_conditioned_on_Q(
                function_distributions[mfinal]["ju"],
                get_function_from_pitch
            )
        elif idx < len(generated_piece) - 1 and cipai["meter"][::-1][idx + 1] == "ju":
            next_pitch_distribution = pitch_transition_probabilities["base"][mfinal][mgong][current_triple]
            distr = next_pitch_distribution.get_conditioned_on_Q(
                function_distributions[mfinal]["after_ju"],
                get_function_from_pitch
            )
        elif cipai["meter"][::-1][idx] == "dou":
            next_pitch_distribution = pitch_transition_probabilities["no_repetition"][mfinal][mgong][current_triple]
            distr = next_pitch_distribution.get_conditioned_on_Q(
                function_distributions[mfinal]["dou"],
                get_function_from_pitch
            )
        elif idx < len(generated_piece) - 1 and cipai["meter"][::-1][idx + 1] == "dou":
            next_pitch_distribution = pitch_transition_probabilities["no_repetition"][mfinal][mgong][current_triple]
            distr = next_pitch_distribution.get_conditioned_on_Q(
                function_distributions[mfinal]["after_dou"],
                get_function_from_pitch
            )
        else:
            distr = pitch_transition_probabilities["no_repetition"][mfinal][mgong][current_triple]

        return distr

    if not "1" in repetition["repetition"] and not "r" in repetition["repetition"]: # no-repetition case
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
                    distr = get_current_pitch_distribution(idx=idx, current_triple=tuple(generated_piece[idx - 3:idx]))

                    new_last_contour = int(np.sign(relative_pitch_to_interval({"gong_lvlv": mode["mgong"]},
                                                                              [current_triple[0], current_triple[1]])[
                                                       0]))
                    if new_last_contour != 0:  # if 0 contour, i.e., tone repetition, use the previous contour!
                        last_contour = new_last_contour
                    distr = distr.get_conditioned_on_Q(
                        contour_distributions[last_contour][tuple(retrograde_tones[0:3])],
                        get_contour
                    )
                    generated_piece[idx] = distr.sample()
                    probability *= distr[generated_piece[idx]]
                    idx += 1

        # Generate first stanza
        while idx < len(generated_piece):
            if idx == pian_idx:  # ending of stanza
                generated_piece[pian_idx:pian_idx+3] = pitch_initial_state
                idx += 3
            else:
                current_triple = tuple(generated_piece[idx - 3:idx])
                distr = get_current_pitch_distribution(idx=idx, current_triple=tuple(generated_piece[idx - 3:idx]))

                new_last_contour = int(np.sign(relative_pitch_to_interval({"gong_lvlv": mode["mgong"]},
                                                                          [current_triple[0], current_triple[1]])[
                                                   0]))
                if new_last_contour != 0:  # if 0 contour, i.e., tone repetition, use the previous contour!
                    last_contour = new_last_contour
                distr = distr.get_conditioned_on_Q(
                    contour_distributions[last_contour][tuple(retrograde_tones[0:3])],
                    get_contour
                )
                generated_piece[idx] = distr.sample()
                probability *= distr[generated_piece[idx]]
                idx += 1
    elif not "1" in repetition["repetition"]:  # inter-strophal repetitions case. the intra-strophal repetitions must be treated separately
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
                    distr = get_current_pitch_distribution(idx=idx, current_triple=tuple(generated_piece[idx - 3:idx]))

                    new_last_contour = int(np.sign(relative_pitch_to_interval({"gong_lvlv": mode["mgong"]},
                        [current_triple[0], current_triple[1]])[0]))
                    if new_last_contour != 0:  # if 0 contour, i.e., tone repetition, use the previous contour!
                        last_contour = new_last_contour
                    distr = distr.get_conditioned_on_Q(
                        contour_distributions[last_contour][tuple(retrograde_tones[0:3])],
                        get_contour
                    )
                    generated_piece[idx] = distr.sample()
                    probability *= distr[generated_piece[idx]]
                    idx += 1

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
                        Distribution.from_dict({suzipu: 1. if suzipu==generated_piece[pian_idx+1] else 0. for suzipu in SimpleSuzipuList}),
                        lambda tuple: tuple[1]
                    )
                except Exception:
                    try: # do not repeat everything, try out [final, ., r]
                        initial_dist = initial_prob.get_conditioned_on_Q(
                            Distribution.from_dict(
                                {suzipu: 1. if suzipu == generated_piece[pian_idx + 2] else 0. for suzipu in
                                 SimpleSuzipuList}),
                            lambda tuple: tuple[2]
                        )
                    except Exception:  # if nothing works, resample completely
                        initial_dist = initial_prob
        elif first_stanza_ending[1] == "r": # [final, r, .] case
            try:
                initial_dist = initial_prob.get_conditioned_on_Q(
                    Distribution.from_dict(
                        {suzipu: 1. if suzipu == generated_piece[pian_idx + 1] else 0. for suzipu in SimpleSuzipuList}),
                    lambda tuple: tuple[1]
                )
            except Exception:  # if it doesn't work, resample completely
                initial_dist = initial_prob
        elif first_stanza_ending[2] == "r": # [final, ., r]#
            try:
                initial_dist = initial_prob.get_conditioned_on_Q(
                    Distribution.from_dict(
                        {suzipu: 1. if suzipu == generated_piece[pian_idx + 2] else 0. for suzipu in
                         SimpleSuzipuList}),
                    lambda tuple: tuple[2]
                )
            except Exception:  # if it doesn't work, resample completely
                initial_dist = initial_prob
        else: # [., ., .] case
            initial_dist = initial_prob

        generated_piece[pian_idx:pian_idx + 3] = initial_dist.sample()
        probability *= initial_dist[tuple(generated_piece[pian_idx:pian_idx + 3])]

        # now, get all possibilities for the first stanza, calculate the likelihood and sample accordingly
        allowed_suzipu_list = get_suzipu_pitches(mode["mgong"])
        allowed_suzipu_list = [s for s in allowed_suzipu_list if s is not None]

        # get all gaps
        first_stanza_idx_groups = [[first_stanza_need_to_fill_idxs[0]]]
        for idx in range(len(first_stanza_need_to_fill_idxs) - 1):
            if first_stanza_need_to_fill_idxs[idx+1] != first_stanza_need_to_fill_idxs[idx] + 1:
                first_stanza_idx_groups.append([])
            first_stanza_idx_groups[-1].append(first_stanza_need_to_fill_idxs[idx+1])

        # if necessary, split large gaps (> 4)
        for gap_list in first_stanza_idx_groups:
            pass

        print(first_stanza_idx_groups)
        
        def fill_by_sampling_all_trajectories(probability):
            for gap_list in first_stanza_idx_groups:
                probable_trajectories = []
                probable_likelihoods = []

                def recursive_traverse(current_trajectory, depth, start_probability):
                    current_prob = start_probability

                    if depth > 0:
                        index = gap_list[depth-1]
                        current_prob *= get_current_pitch_distribution(idx=index, current_triple=tuple(generated_piece[index - 3:index]))[generated_piece[index]]

                    if depth == len(gap_list):
                        index = gap_list[depth - 1]
                        if index < len(generated_piece)-1 and None not in generated_piece[index - 2:index+1] and generated_piece[index+1] is not None: # also incorporate final state into probability calculation
                            try:
                                current_prob *= get_current_pitch_distribution(idx=index+1, current_triple=tuple(generated_piece[index - 2:index+1]))[generated_piece[index+1]]
                            except ZeroDivisionError:
                                current_prob = 0.
                        if index < len(generated_piece)-2 and None not in generated_piece[index - 1:index+2] and generated_piece[index+2] is not None: # also incorporate final state into probability calculation
                            try:
                                current_prob *= get_current_pitch_distribution(idx=index+2, current_triple=tuple(generated_piece[index - 1:index+2]))[generated_piece[index+2]]
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
                probability *= trajectory_distribution[current_trajectory]
                for i, fill_idx in enumerate(gap_list):
                    generated_piece[fill_idx] = current_trajectory[i]
            return probability

        probability = fill_by_sampling_all_trajectories(probability)

    else:  # danhuangliu case
        return {
            "pitch_list": "Not implemented",
            "description": description_string,
            "probability": probability,
        }

    generated_piece.reverse()
    return {
        "pitch_list": generated_piece,
        "description": description_string,
        "probability": probability
    }
