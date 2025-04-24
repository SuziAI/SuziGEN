import numpy as np
import random
import pylcs

from text_resources import EnglishTexts


# calculate some statistics such as repetition share, tonal or metrical agreement inside the repetition wrt the cipai
def get_agreements(cipai, repetitions):
    first_stanza_structure = get_sentence_lengths(cipai)[0]
    second_stanza_structure = get_sentence_lengths(cipai)[1]

    meter = [symbol if symbol != "" else " " for symbol in cipai["meter"]]

    first_stanza_repetitions = repetitions[0:sum(first_stanza_structure)]
    second_stanza_repetitions = repetitions[sum(first_stanza_structure):]

    first_stanza_tones = cipai["tones"][0:sum(first_stanza_structure)]
    second_stanza_tones = cipai["tones"][sum(first_stanza_structure):]

    first_stanza_meter_less_strict = ["mark" if tone in ("ju", "pian", "dou") else " " for tone in
                                      cipai["meter"][0:sum(first_stanza_structure)]]
    second_stanza_meter_less_strict = ["mark" if tone in ("ju", "pian", "dou") else " " for tone in
                                       cipai["meter"][sum(first_stanza_structure):]]

    first_stanza_meter = ["ju" if tone == "pian" else tone for tone in meter[0:sum(first_stanza_structure)]]
    second_stanza_meter = ["ju" if tone == "pian" else tone for tone in meter[sum(first_stanza_structure):]]

    first_stanza_tones_conditioned_on_repeat = [tone for tone, repetition in
                                                zip(first_stanza_tones, first_stanza_repetitions) if repetition == "r"]
    second_stanza_tones_conditioned_on_repeat = [tone for tone, repetition in
                                                 zip(second_stanza_tones, second_stanza_repetitions) if
                                                 repetition == "r"]

    first_stanza_structure_conditioned_on_repeat = [struc for struc, repetition in
                                                    zip(first_stanza_meter, first_stanza_repetitions) if
                                                    repetition == "r"]
    second_stanza_structure_conditioned_on_repeat = [struc for struc, repetition in
                                                     zip(second_stanza_meter, second_stanza_repetitions) if
                                                     repetition == "r"]

    rep_percent = (np.array(repetitions) != ".").mean() * 100
    tone_match = (np.array(first_stanza_tones_conditioned_on_repeat) == np.array(
        second_stanza_tones_conditioned_on_repeat)).mean() * 100
    meter_match = (np.array(first_stanza_structure_conditioned_on_repeat) == np.array(
        second_stanza_structure_conditioned_on_repeat)).mean() * 100

    return rep_percent, tone_match, meter_match


# for pylcs, we later need to convert the meter and tone to a single string encoding all information
def get_combined_str(meter, tone):
    output_str = []
    for m, t in zip(meter, tone):
        if m == " ":  # no meter mark
            if t == "p":
                element = "0"
            else:
                element = "1"
        elif m == "j":  # ju
            if t == "p":
                element = "2"
            else:
                element = "3"
        else:  # dou
            if t == "p":
                element = "4"
            else:
                element = "5"

        output_str.append(element)
    return "".join(output_str)


# get the ju structure of the cipai for each stanza
def get_sentence_lengths(cipai):
    sentence_lengths = []
    counter = 0
    pian_lengths = []
    for idx, meter in enumerate(cipai["meter"]):
        if meter not in ["ju", "pian"]:
            counter += 1
        elif meter == "ju" and idx != len(cipai["meter"]) - 1:
            counter += 1
            pian_lengths.append(counter)
            counter = 0
        else:
            counter += 1
            pian_lengths.append(counter)
            counter = 0
            sentence_lengths.append(pian_lengths)
            pian_lengths = []
    return sentence_lengths


def get_subsequence_indices(subseq, sequence):
    """
    Finds the indices of subseq in sequence if subseq is a subsequence of sequence.
    If not, returns None.

    Parameters:
        subseq (list): The subsequence to find.
        sequence (list): The sequence to search in.

    Returns:
        list or None: Indices of the subsequence in the sequence, or None if subseq
                      is not a subsequence of sequence.
    """
    indices = []
    seq_iter = iter(enumerate(sequence))

    for sub_elem in subseq:
        for idx, seq_elem in seq_iter:
            if seq_elem == sub_elem:
                indices.append(idx)
                break
        else:
            # If the inner loop exits without breaking, subseq is not a subsequence
            return None

    return indices


def get_substring_indices(substring, sequence):
    """
    Finds the indices of a substring in sequence if substring is a contiguous part of sequence.
    If not, returns None.

    Parameters:
        substring (list): The substring to find.
        sequence (list): The sequence to search in.

    Returns:
        list or None: Indices of the substring in the sequence, or None if substring
                      is not a contiguous part of sequence.
    """
    n, m = len(sequence), len(substring)

    # Check for all possible start positions in the sequence
    for start in range(n - m + 1):
        if sequence[start:start + m] == substring:
            return list(range(start, start + m))

    return None


# remove some repetition signs
def erode_repetitions(first_stanza, second_stanza, num_repetitions=1, erode_front=True, choose_random=False):
    for rep in range(num_repetitions):
        first_indices = [idx for idx, r_str in enumerate(first_stanza) if r_str == "r"]
        second_indices = [idx for idx, r_str in enumerate(second_stanza) if r_str == "r"]

        if erode_front:
            delete_indices = [idx for idx, el in enumerate(first_indices) if
                              1 <= el < len(first_stanza) and first_stanza[el - 1:el + 1] == ".r"]
        else:
            delete_indices = [idx for idx, el in enumerate(first_indices) if
                              0 <= el < len(first_stanza) - 1 and first_stanza[el:el + 2] == "r."]

        if choose_random and len(delete_indices):
            delete_indices = random.sample(delete_indices, 1)

        first_indices = [el for idx, el in enumerate(first_indices) if idx not in delete_indices]
        second_indices = [el for idx, el in enumerate(second_indices) if idx not in delete_indices]

        first_stanza = "".join(["r" if idx in first_indices else "." for idx in range(len(first_stanza))])
        second_stanza = "".join(["r" if idx in second_indices else "." for idx in range(len(second_stanza))])
    return first_stanza, second_stanza


# add some repetition signs
def dilate_repetitions(first_stanza, second_stanza, num_repetitions=1, dilate_front=True, choose_random=False):
    for rep in range(num_repetitions):
        first_indices = [idx for idx, r_str in enumerate(first_stanza) if r_str == "r"]
        second_indices = [idx for idx, r_str in enumerate(second_stanza) if r_str == "r"]

        if dilate_front:
            add_indices = [idx for idx, el in enumerate(first_indices) if
                           1 <= el < len(first_stanza) and first_stanza[el - 1:el + 1] == ".r" and second_stanza[
                                                                                                   second_indices[
                                                                                                       idx] - 1:
                                                                                                   second_indices[
                                                                                                       idx] + 1] == ".r"]
            if choose_random and len(add_indices):
                add_indices = random.sample(add_indices, 1)
            first_indices += [el - 1 for idx, el in enumerate(first_indices) if idx in add_indices]
            second_indices += [el - 1 for idx, el in enumerate(second_indices) if idx in add_indices]
        else:
            add_indices = [idx for idx, el in enumerate(first_indices) if
                           0 <= el < len(first_stanza) - 1 and first_stanza[el:el + 2] == "r." and second_stanza[
                                                                                                   second_indices[idx]:
                                                                                                   second_indices[
                                                                                                       idx] + 2] == "r."]
            if choose_random and len(add_indices):
                add_indices = random.sample(add_indices, 1)
            first_indices += [el + 1 for idx, el in enumerate(first_indices) if idx in add_indices]
            second_indices += [el + 1 for idx, el in enumerate(second_indices) if idx in add_indices]

        first_stanza = "".join(["r" if idx in first_indices else "." for idx in range(len(first_stanza))])
        second_stanza = "".join(["r" if idx in second_indices else "." for idx in range(len(second_stanza))])
    return first_stanza, second_stanza


# introduce ending repetitions, i.e., in the last three characters in each stanza
def add_ending_repetition(first_stanza, second_stanza):
    if len(first_stanza) <= 3 or first_stanza[-1] != "." or second_stanza[-1] != ".":
        return first_stanza, second_stanza

    first_stanza = first_stanza[::-1]
    second_stanza = second_stanza[::-1]

    first_indices = [idx for idx, r_str in enumerate(first_stanza) if r_str == "."]
    second_indices = [idx for idx, r_str in enumerate(second_stanza) if r_str == "."]

    components = []

    previous_index = first_indices[0]
    current_component = [0]
    for ctr, index in enumerate(first_indices[1:]):
        if index == previous_index + 1:  # consecutive indices belong to a component
            current_component.append(ctr + 1)
        else:
            components.append(current_component)
            current_component = [ctr + 1]
        previous_index = index
    components.append(current_component)

    delete_indices = components[0]
    if len(delete_indices) >= 3:
        delete_indices = delete_indices[0:3]

    first_indices = [el for idx, el in enumerate(first_indices) if idx not in delete_indices]
    second_indices = [el for idx, el in enumerate(second_indices) if idx not in delete_indices]

    first_stanza = "".join(["." if idx in first_indices else "r" for idx in range(len(first_stanza))])
    second_stanza = "".join(["." if idx in second_indices else "r" for idx in range(len(second_stanza))])

    first_stanza = first_stanza[::-1]
    second_stanza = second_stanza[::-1]

    return first_stanza, second_stanza


# remove structures of the form ....r...
def remove_single_repetition(first_stanza, second_stanza):
    first_indices = [idx for idx, r_str in enumerate(first_stanza) if r_str == "r"]
    second_indices = [idx for idx, r_str in enumerate(second_stanza) if r_str == "r"]

    delete_indices = [idx for idx, el in enumerate(first_indices) if
                      1 <= el < len(first_stanza) - 1 and first_stanza[el - 1:el + 2] == ".r."]
    if first_stanza[0:2] == "r." and second_stanza[0:2] == "r.":
        delete_indices += [0]

    first_indices = [el for idx, el in enumerate(first_indices) if idx not in delete_indices]
    second_indices = [el for idx, el in enumerate(second_indices) if idx not in delete_indices]

    first_stanza = "".join(["r" if idx in first_indices else "." for idx in range(len(first_stanza))])
    second_stanza = "".join(["r" if idx in second_indices else "." for idx in range(len(second_stanza))])
    return first_stanza, second_stanza


def remove_repetition_for_secondary(input_cipai, repetition_string):
    if "1" in repetition_string:
        return ["." for r in repetition_string]

    first_stanza_structure = get_sentence_lengths(input_cipai)[0]

    first_stanza = repetition_string[:sum(first_stanza_structure)]
    second_stanza = repetition_string[sum(first_stanza_structure):]

    first_stanza_idxs = [idx for idx, r in enumerate(first_stanza) if r == "r"]
    second_stanza_idxs = [idx for idx, r in enumerate(second_stanza) if r == "r"]

    remove_idxs = [idx for idx in range(len(first_stanza)) if np.random.rand() < 0.2] # remove 20% of indices on average

    first_stanza_idxs = [idx for number, idx in enumerate(first_stanza_idxs) if number not in remove_idxs]
    second_stanza_idxs = [idx for number, idx in enumerate(second_stanza_idxs) if number not in remove_idxs]

    first_stanza = ["r" if idx in first_stanza_idxs else "." for idx in range(len(first_stanza))]
    second_stanza = ["r" if idx in second_stanza_idxs else "." for idx in range(len(second_stanza))]
    repetition_string = remove_single_repetition(first_stanza, second_stanza)
    return list(repetition_string[0] + repetition_string[1])

# randomly select some repetition components that are removed
def remove_some_components(first_stanza, second_stanza):
    first_indices = [idx for idx, r_str in enumerate(first_stanza) if r_str == "r"]
    second_indices = [idx for idx, r_str in enumerate(second_stanza) if r_str == "r"]

    if not len(first_indices):
        return first_stanza, second_stanza

    components = []

    previous_index = first_indices[0]
    current_component = [0]
    for ctr, index in enumerate(first_indices[1:]):
        if index == previous_index + 1:  # consecutive indices belong to a component
            current_component.append(ctr + 1)
        else:
            components.append(current_component)
            current_component = [ctr + 1]
        previous_index = index
    components.append(current_component)

    def flatten(xss):
        return [x for xs in xss for x in xs]

    if len(components) > 1:
        min_num_components_removed = 0
        if len(components) > 4:
            min_num_components_removed = len(components) - 4
        elif (np.array(list(first_stanza + second_stanza)) != ".").mean() > 0.95:  # too much repetition
            min_num_components_removed = 1

        max_num_components_removed = len(components) - 1

        remove_components = np.random.randint(min_num_components_removed,
                                              max_num_components_removed) if min_num_components_removed < max_num_components_removed else min_num_components_removed
        delete_indices = flatten(random.sample(components, remove_components))
        first_indices = [el for idx, el in enumerate(first_indices) if idx not in delete_indices]
        second_indices = [el for idx, el in enumerate(second_indices) if idx not in delete_indices]
        first_stanza = "".join(["r" if idx in first_indices else "." for idx in range(len(first_stanza))])
        second_stanza = "".join(["r" if idx in second_indices else "." for idx in range(len(second_stanza))])

    return first_stanza, second_stanza


def generate_repetition(input_cipai):
    first_stanza_structure = get_sentence_lengths(input_cipai)[0]
    second_stanza_structure = get_sentence_lengths(input_cipai)[1]

    first_stanza_tones = input_cipai["tones"][0:sum(first_stanza_structure)]
    second_stanza_tones = input_cipai["tones"][sum(first_stanza_structure):]

    first_stanza_meter = ["mark" if tone in ("ju", "pian", "dou") else " " for tone in
                          input_cipai["meter"][0:sum(first_stanza_structure)]]
    second_stanza_meter = ["mark" if tone in ("ju", "pian", "dou") else " " for tone in
                           input_cipai["meter"][sum(first_stanza_structure):]]

    # second_meter_idxs = [idx for idx in pylcs.lcs_sequence_idx(first_stanza_meter_str, second_stanza_meter_str) if idx != -1]

    # we have no piece where the repetition is 100% exact!
    # But the majority of pieces do feature at least some repetition.

    if np.random.rand() < 2 / 17:
        description_string = EnglishTexts.repetitions_none.format(prob=2/17*100)
        return {"repetition": ["."] * len(input_cipai["tones"]), "description": description_string}

    # Qiuxiaoyin case with intra-strophal repetition and high compatibility between the first two half-stanzas
    if first_stanza_structure[0:len(first_stanza_structure) // 2] == first_stanza_structure[
                                                                     len(first_stanza_structure) // 2:] and (
            np.array(first_stanza_tones[:len(first_stanza_tones) // 2]) == np.array(
        first_stanza_tones[len(first_stanza_tones) // 2:])).mean() > 0.85:

        tonal_agreement = (np.array(first_stanza_tones[:len(first_stanza_tones) // 2]) == np.array(first_stanza_tones[len(first_stanza_tones) // 2:])).mean()

        half_stanza_1 = first_stanza_tones[:len(first_stanza_tones) // 2]  # A B
        # half_stanza_2 = first_stanza_tones[len(first_stanza_tones)//2:]  # A B
        half_stanza_3 = second_stanza_tones[:len(second_stanza_tones) // 2]  # C B
        half_stanza_4 = second_stanza_tones[len(second_stanza_tones) // 2:]  # C D

        tones_a = half_stanza_1[:len(half_stanza_1) // 2]  # A
        tones_b = half_stanza_1[len(half_stanza_1) // 2:]  # B

        tones2_c = half_stanza_3[:len(half_stanza_3) // 2]  # C
        tones2_b = half_stanza_3[len(half_stanza_3) // 2:]  # B
        tones2_c2 = half_stanza_4[:len(half_stanza_4) // 2]  # C
        tones2_d = half_stanza_4[len(half_stanza_4) // 2:]  # D

        length_b = min(len(tones_b), len(tones2_b))
        if length_b >= 6:  # we want to have a B part that has at least 3 entries so it can form a full cadential phrase
            length_b = np.random.randint(length_b // 2, length_b - 2) if length_b > 2 else 0
            length_c = min(len(tones2_c), len(tones2_c2))
            length_c = np.random.randint(length_c // 2, length_c - 2) if length_c > 2 else 0

            first_stanza_1_a = "".join(["1" for tone in tones_a])
            first_stanza_1_b = ["1" for tone in tones_b]
            first_stanza_1_b[-length_b:] = ["2"] * length_b
            first_stanza_1_b = "".join(first_stanza_1_b)
            first_stanza = first_stanza_1_a + first_stanza_1_b + first_stanza_1_a + first_stanza_1_b

            second_stanza_1_c = ["3" for tone in tones2_c]
            num_dots = len(second_stanza_1_c) - length_c
            head_len = np.random.randint(0, num_dots - 1) if num_dots > 1 else 0
            intermediate_len = np.random.randint(0, 3)
            tail_len = num_dots - head_len - intermediate_len
            if head_len:
                second_stanza_1_c[0:head_len] = "." * head_len
            if tail_len:
                second_stanza_1_c[-tail_len:] = "." * tail_len
            if intermediate_len:
                intermediate_idx = random.sample(range(head_len, len(second_stanza_1_c) - tail_len), intermediate_len)
                for iidx in intermediate_idx:
                    second_stanza_1_c[iidx] = "."
            second_stanza_1_c = "".join(second_stanza_1_c)

            second_stanza_1_b = ["2" for tone in tones2_b]
            num_dots = len(second_stanza_1_b) - length_b
            head_len = np.random.randint(0, num_dots - 1) if num_dots > 1 else 0
            intermediate_len = 0  # np.random.randint(0, 3)
            tail_len = num_dots - head_len - intermediate_len
            if head_len:
                second_stanza_1_b[0:head_len] = "." * head_len
            if tail_len:
                second_stanza_1_b[-tail_len:] = "." * tail_len
            if intermediate_len:
                intermediate_idx = random.sample(range(head_len, len(second_stanza_1_b) - tail_len), intermediate_len)
                for iidx in intermediate_idx:
                    second_stanza_1_b[iidx] = "."
            second_stanza_1_b = "".join(second_stanza_1_b)

            second_stanza_2_c = ["3" for tone in tones2_c2]
            num_dots = len(second_stanza_2_c) - length_c
            head_len = np.random.randint(0, num_dots - 1) if num_dots > 1 else 0
            intermediate_len = 0  # np.random.randint(0, 3)
            tail_len = num_dots - head_len - intermediate_len
            if head_len:
                second_stanza_2_c[0:head_len] = "." * head_len
            if tail_len:
                second_stanza_2_c[-tail_len:] = "." * tail_len
            if intermediate_len:
                intermediate_idx = random.sample(range(head_len, len(second_stanza_2_c) - tail_len), intermediate_len)
                for iidx in intermediate_idx:
                    second_stanza_2_c[iidx] = "."
            second_stanza_2_c = "".join(second_stanza_2_c)

            second_stanza_2_d = "".join(["." for tone in tones2_d])

            first_indices_a = [idx for idx, r_str in enumerate(first_stanza[:len(first_stanza) // 2]) if r_str == "1"]
            second_indices_a = [idx for idx, r_str in enumerate(first_stanza[len(first_stanza) // 2:]) if r_str == "1"]
            first_indices_b = [idx for idx, r_str in enumerate(first_stanza[:len(first_stanza) // 2]) if r_str == "2"]
            second_indices_b = [idx for idx, r_str in enumerate(first_stanza[len(first_stanza) // 2:]) if r_str == "2"]
            first_indices_c = [idx for idx, r_str in enumerate(second_stanza_1_c) if r_str == "3"]
            second_indices_c = [idx for idx, r_str in enumerate(second_stanza_2_c) if r_str == "3"]
            first_indices_d = [idx for idx, r_str in enumerate(first_stanza[len(first_stanza) // 2:]) if r_str == "2"]
            second_indices_d = [idx for idx, r_str in enumerate(second_stanza_1_b) if r_str == "2"]
            if len(first_indices_a) != len(second_indices_a) or \
                    len(first_indices_b) != len(second_indices_b) or \
                    len(first_indices_c) != len(second_indices_c) or \
                    len(first_indices_d) != len(second_indices_d):  # make sure that the repetition structure is valid!
                return ["." for idx, t in enumerate(input_cipai["tones"])]
            description_string = EnglishTexts.repetitions_intrastrophal.format(tonal_agreement=tonal_agreement*100)
            return {"repetition": list(first_stanza + second_stanza_1_c + second_stanza_1_b + second_stanza_2_c + second_stanza_2_d), "description": description_string}

    # Geximeiling case with parallel stanzas and high degree of tone compatibility
    if first_stanza_structure == second_stanza_structure and (
            np.array(first_stanza_tones) == np.array(second_stanza_tones)).mean() > 0.9:
        cipai = ["r" for t in input_cipai["tones"]]

        tonal_agreement = (np.array(first_stanza_tones) == np.array(second_stanza_tones)).mean()

        # remove a portion of the first ju
        cipai[0:first_stanza_structure[0]] = ["."] * first_stanza_structure[0]
        cipai[len(first_stanza_tones):len(first_stanza_tones) + first_stanza_structure[0]] = ["."] * \
                                                                                             first_stanza_structure[0]
        if (first_stanza_structure[0] > 5):
            keep_portion = np.random.randint(2, first_stanza_structure[0] - 2)
            cipai[0:keep_portion] = ["r"] * keep_portion
            cipai[len(first_stanza_tones):len(first_stanza_tones) + keep_portion] = ["r"] * keep_portion
        cipai_str = "".join(cipai)

        cipai_first = cipai_str[:len(first_stanza_tones)]
        cipai_second = cipai_str[len(first_stanza_tones):]

        cipai_first, cipai_second = erode_repetitions(cipai_first, cipai_second,
                                                      num_repetitions=np.random.randint(0, 5))

        cipai_first, cipai_second = remove_single_repetition(cipai_first, cipai_second)
        cipai_str = cipai_first + cipai_second

        first_indices = [idx for idx, r_str in enumerate(cipai_first) if r_str == "r"]
        second_indices = [idx for idx, r_str in enumerate(cipai_second) if r_str == "r"]
        if len(first_indices) != len(second_indices):  # make sure that the repetition structure is valid!
            return ["." for idx, t in enumerate(cipai["tones"])]

        description_string = EnglishTexts.repetitions_similar_strophes.format(tonal_agreement=tonal_agreement*100)
        return {"repetition": list(cipai_str), "description": description_string}

    # Xinghuatianying, Zuiyinshangxiaopin case with parallel stanzas and lower degree of tone compatibility
    elif first_stanza_structure == second_stanza_structure:
        cipai = ["r" for t in input_cipai["tones"]]
        stanza_structure_cumsum = np.array([0] + first_stanza_structure).cumsum()

        tonal_agreement = (np.array(first_stanza_tones) == np.array(second_stanza_tones)).mean()

        # remove the ju with low tone compatibility (< 75%)
        for ju_idx in range(len(stanza_structure_cumsum) - 1):
            current_first_tones = first_stanza_tones[
                                  stanza_structure_cumsum[ju_idx]:stanza_structure_cumsum[ju_idx + 1]]
            current_second_tones = second_stanza_tones[
                                   stanza_structure_cumsum[ju_idx]:stanza_structure_cumsum[ju_idx + 1]]
            if (np.array(current_first_tones) == np.array(current_second_tones)).mean() < 0.75:
                cipai[stanza_structure_cumsum[ju_idx]:stanza_structure_cumsum[ju_idx + 1]] = ["."] * (
                        stanza_structure_cumsum[ju_idx + 1] - stanza_structure_cumsum[ju_idx])
                cipai[len(first_stanza_tones) + stanza_structure_cumsum[ju_idx]:len(first_stanza_tones) +
                                                                                stanza_structure_cumsum[ju_idx + 1]] = [
                                                                                                                           "."] * (
                                                                                                                               stanza_structure_cumsum[
                                                                                                                                   ju_idx + 1] -
                                                                                                                               stanza_structure_cumsum[
                                                                                                                                   ju_idx])

        cipai_str = "".join(cipai)

        # erode string randomly
        cipai_first = cipai_str[:len(first_stanza_tones)]
        cipai_second = cipai_str[len(first_stanza_tones):]

        cipai_first, cipai_second = erode_repetitions(cipai_first, cipai_second,
                                                      num_repetitions=np.random.randint(0, 5), erode_front=True)
        cipai_first, cipai_second = erode_repetitions(cipai_first, cipai_second,
                                                      num_repetitions=np.random.randint(0, 5), erode_front=False)

        cipai_first, cipai_second = remove_single_repetition(cipai_first, cipai_second)
        cipai_str = cipai_first + cipai_second

        first_indices = [idx for idx, r_str in enumerate(cipai_first) if r_str == "r"]
        second_indices = [idx for idx, r_str in enumerate(cipai_second) if r_str == "r"]
        if len(first_indices) != len(second_indices):  # make sure that the repetition structure is valid!
            return ["." for idx, t in enumerate(cipai["tones"])]

        description_string = EnglishTexts.repetitions_not_so_similar_strophes.format(tonal_agreement=tonal_agreement * 100)
        return {"repetition": list(cipai_str), "description": description_string}
    else: # non-parallel stanzas case
        def recursive_repetitions(first_cipai, second_cipai, return_raw_indices=False):
            lfc = len(first_cipai)
            lsc = len(second_cipai)

            if not return_raw_indices:
                lower_f = 2 * lfc // 5
                upper_f = 3 * lfc // 5
                lower_s = 2 * lsc // 5
                upper_s = 3 * lsc // 5
            else:
                lower_f = lfc // 4
                upper_f = 3 * lfc // 4
                lower_s = lsc // 4
                upper_s = 3 * lsc // 4

            first_longest_substring_idxs = [idx + lower_f for idx in pylcs.lcs_string_idx(second_cipai[lower_s:upper_s],
                                                                                          first_cipai[lower_f:upper_f])
                                            if idx != -1]
            if first_longest_substring_idxs:
                first_longest_substring_seq = "".join([first_cipai[idx] for idx in first_longest_substring_idxs])
                second_longest_substring_idxs = list(np.array(
                    get_substring_indices(first_longest_substring_seq, second_cipai[lower_s:upper_s])) + lower_s)

                left_first_part = first_cipai[0:first_longest_substring_idxs[0]]
                left_second_part = second_cipai[0:second_longest_substring_idxs[0]]
                right_first_part = first_cipai[first_longest_substring_idxs[-1] + 1:]
                right_second_part = second_cipai[second_longest_substring_idxs[-1] + 1:]

                if len(left_first_part) > 3 and len(left_second_part) > 3:
                    left_part_first_substring_idxs, left_part_second_substring_idxs = recursive_repetitions(
                        left_first_part, left_second_part, return_raw_indices=True)
                    if left_part_first_substring_idxs:
                        first_longest_substring_idxs = left_part_first_substring_idxs + first_longest_substring_idxs
                        second_longest_substring_idxs = left_part_second_substring_idxs + second_longest_substring_idxs

                if len(right_first_part) > 3 and len(right_second_part) > 3:
                    right_part_first_substring_idxs, right_part_second_substring_idxs = recursive_repetitions(
                        right_first_part, right_second_part, return_raw_indices=True)
                    if right_part_first_substring_idxs:
                        first_longest_substring_idxs += list(
                            np.array(right_part_first_substring_idxs) + first_longest_substring_idxs[-1] + 1)
                        second_longest_substring_idxs += list(
                            np.array(right_part_second_substring_idxs) + second_longest_substring_idxs[-1] + 1)

                return list(first_longest_substring_idxs), list(np.array(second_longest_substring_idxs) + len(
                    first_cipai)) if not return_raw_indices else second_longest_substring_idxs
            else:
                return None, None

        first_stanza_meter_str = "".join([s[0] for s in first_stanza_meter])
        second_stanza_meter_str = "".join([s[0] for s in second_stanza_meter])
        first_stanza_tone_str = "".join([s[0] for s in first_stanza_tones])
        second_stanza_tone_str = "".join([s[0] for s in second_stanza_tones])

        first_stanza_cipai = get_combined_str(first_stanza_meter_str, first_stanza_tone_str)
        second_stanza_cipai = get_combined_str(second_stanza_meter_str, second_stanza_tone_str)

        first_idxs, second_idxs = recursive_repetitions(first_stanza_cipai, second_stanza_cipai)
        if not first_idxs:
            return ["." for idx, t in enumerate(input_cipai["tones"])]

        full_seq = first_idxs + second_idxs

        cipai = ["r" if idx in full_seq else "." for idx, t in enumerate(input_cipai["tones"])]

        cipai_str = "".join(cipai)

        # manipulate indices
        cipai_first = cipai_str[:len(first_stanza_tones)]
        cipai_second = cipai_str[len(first_stanza_tones):]

        # erode and dilate for some iterations, while keeping the stats in good ranges
        # if False:
        #    num_iterations = np.random.randint(0, 2)
        #    for iteration in range(num_iterations):
        #        mode = np.random.randint(0, 4)
        #        if mode == 0:
        #            cipai_first_candidate, cipai_second_candidate = erode_repetitions(cipai_first, cipai_second, num_repetitions=1, erode_front=True, choose_random=True)
        #        elif mode == 1:
        #            cipai_first_candidate, cipai_second_candidate = dilate_repetitions(cipai_first, cipai_second, num_repetitions=1, dilate_front=True, choose_random=True)
        #        elif mode == 2:
        #            cipai_first_candidate, cipai_second_candidate = erode_repetitions(cipai_first, cipai_second, num_repetitions=1, erode_front=False, choose_random=True)
        #        else:
        #            cipai_first_candidate, cipai_second_candidate = dilate_repetitions(cipai_first, cipai_second, num_repetitions=1, dilate_front=False, choose_random=True)

        cipai_first_candidate, cipai_second_candidate = erode_repetitions(cipai_first, cipai_second,
                                                                          num_repetitions=np.random.randint(0, 3),
                                                                          erode_front=True, choose_random=False)
        cipai_first_candidate, cipai_second_candidate = dilate_repetitions(cipai_first_candidate,
                                                                           cipai_second_candidate,
                                                                           num_repetitions=np.random.randint(1, 4),
                                                                           dilate_front=True, choose_random=False)
        cipai_first_candidate, cipai_second_candidate = erode_repetitions(cipai_first_candidate, cipai_second_candidate,
                                                                          num_repetitions=np.random.randint(0, 3),
                                                                          erode_front=False, choose_random=False)
        cipai_first_candidate, cipai_second_candidate = dilate_repetitions(cipai_first_candidate,
                                                                           cipai_second_candidate,
                                                                           num_repetitions=np.random.randint(1, 4),
                                                                           dilate_front=False, choose_random=False)
        _, tone_orig, meter_orig = get_agreements(input_cipai, list(cipai_first + cipai_second))
        _, tone_s, meter_s = get_agreements(input_cipai,
                                            list(cipai_first_candidate + cipai_second_candidate))
        if (tone_orig <= tone_s and meter_orig <= meter_s) or (tone_s > 75 and meter_s > 72):
            cipai_first, cipai_second = cipai_first_candidate, cipai_second_candidate

        if np.random.rand() < 0.5:
            cipai_first_candidate, cipai_second_candidate = remove_some_components(cipai_first, cipai_second)
            _, tone_orig, meter_orig = get_agreements(input_cipai, list(cipai_first + cipai_second))
            _, tone_s, meter_s = get_agreements(input_cipai,
                                                list(cipai_first_candidate + cipai_second_candidate))
            if (tone_orig <= tone_s and meter_orig <= meter_s) or (tone_s > 80 and meter_s > 80):
                cipai_first, cipai_second = cipai_first_candidate, cipai_second_candidate

        if np.random.rand() < 0.8:
            cipai_first_candidate, cipai_second_candidate = add_ending_repetition(cipai_first, cipai_second)
            _, tone_orig, meter_orig = get_agreements(input_cipai, list(cipai_first + cipai_second))
            _, tone_s, meter_s = get_agreements(input_cipai,
                                                list(cipai_first_candidate + cipai_second_candidate))
            if (tone_orig <= tone_s and meter_orig <= meter_s) or (tone_s > 80 and meter_s > 80):
                cipai_first, cipai_second = cipai_first_candidate, cipai_second_candidate

        cipai_first_candidate, cipai_second_candidate = remove_single_repetition(cipai_first, cipai_second)
        _, tone_orig, meter_orig = get_agreements(input_cipai, list(cipai_first + cipai_second))
        _, tone_s, meter_s = get_agreements(input_cipai,
                                            list(cipai_first_candidate + cipai_second_candidate))
        if (tone_orig <= tone_s and meter_orig <= meter_s) or (tone_s > 80 and meter_s > 80):
            cipai_first, cipai_second = cipai_first_candidate, cipai_second_candidate

        first_indices = [idx for idx, r_str in enumerate(cipai_first) if r_str == "r"]
        second_indices = [idx for idx, r_str in enumerate(cipai_second) if r_str == "r"]
        if len(first_indices) != len(second_indices):  # make sure that the repetition structure is valid!
            return ["." for idx, t in enumerate(cipai["tones"])]
        description_string = EnglishTexts.repetitions_default
        return {"repetition": list(cipai_first + cipai_second), "description": description_string}
