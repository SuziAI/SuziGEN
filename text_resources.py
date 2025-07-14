class EnglishTexts:
    repetitions_none: str = (
        "As a repetition structure, I selected the absence of specific repetitions for you. "
        "This happens in {prob:.1f}% of all cases. In Baishidaoren Gequ, "
        "this is the case for the pieces 醉吟商小品 (Zuiyinshangxiaopin) and 淡黃柳 (Danhuangliu)."
    )
    repetitions_intrastrophal: str = (
        "The cipai you provided has a special structure! "
        "The first stanza consists of two parts with equal metrical structure and high agreement of "
        "the Chinese syllable tones which is {tonal_agreement:.1f}%. "
        "Therefore, I chose a repetition pattern for you that has a special form, "
        "which is ABAB/CBCD. In Baishidaoren Gequ, this happens only in the piece "
        "秋宵吟 (Qiuxiaoyin)."
    )
    repetitions_similar_strophes: str = (
        "In the cipai you provided, the first and the second stanzas have the same metrical structure, "
        "and also, the agreement of the Chinese syllable tones inside both stanzas is with "
        "{tonal_agreement:.1f}% very high. Therefore, the repetition I chose for you is a large repetition "
        "in both stanzas. In Baishidaoren Gequ, this appears in the piece 鬲溪梅令 (Geximeiling)."
    )
    repetitions_not_so_similar_strophes: str = (
        "In the cipai you provided, the first and the second stanzas have the same metrical structure, "
        "but the agreement of the Chinese syllable tones inside both stanzas is with "
        "{tonal_agreement:.1f}% not very high. Therefore, for building the repetition pattern I excluded "
        "the ju (sentences) where the agreement is low. In Baishidaoren Gequ, this appears in the piece "
        "杏花天影 (Xinghuatianying)."
    )
    repetitions_default: str = (
        "In your selected cipai, the first and the second stanzas have no obvious metrical similarities, "
        "Therefore, the repetition pattern I chose for you specifically "
        "covers syllables in which the tonal agreement is high. In Baishidaoren Gequ, this is the "
        "case in most of the pieces."
    )

    mode_name: str = (
        "For the mode, I chose {chinese_name} ({name}). I generate this in {probability:.1f}% of all cases. "
    )
    mode_in_baishi: str = (
        "This mode occurs in these pieces of Baishidaoren Gequ: {piece_list}. "
    )
    mode_not_in_baishi: str = (
        "This mode does not occur in Baishidaoren Gequ. "
    )
    mode_final_note: str = (
        "This mode's final note is {final_note}, which is also the final note of these pieces: {final_note_list}."
    )

    second_stanza_pitch_initial_state: str = (
        "Now, I will generate the pitches for you. Since the piece's ending is determined by the mode, "
        "I will first generate the cadential phrase at the end, consisting of three notes. "
        "According to the mode, the piece must end with {final_note}. Therefore, "
        "I chose {cadential_phrase} for you. This happens with a probability of {probability:.1f}%."
    )

    second_stanza_secondary_final: str = (
        "Since we have successfully generated the pitches, it's time to turn our attention to the secondary symbols. "
        "There are two classes of pieces in Baishidaoren Gequ, the ones that end in XIAO_ZHU (5/17) and those that end in DA_DUN (12/17). "
        "I chose {final_secondary}, which accordingly has a probability of {probability:.1f}%."
    )

    second_stanza_secondary_initial_state: str = (
        "Again, I will first generate the last three notes of the second stanza. "
        "This time, I chose {cadential_phrase}, which has a probability of {probability:.1f}%."
    )

    second_stanza_pitch_normal_case: str = (
        "For this type of piece, the second stanza has the most freedom, since the repetition pattern does not yet affect "
        "the generated pitches. The probability for the pitches I chose for you in the second stanza is {probability:,.5g}."
    )

    second_stanza_secondary_normal_case: str = (
        "Again, the second stanza has the most freedom, since the repetition pattern does not yet affect "
        "the generated pitches. The pitches I chose for you in the second stanza have the probability {probability:,.5g}."
    )

    second_stanza_pitch_intrastrophal_case_cd: str = (
        "For this type of piece, the last half of the second stanza (with the form CD) has the most freedom, "
        "since the repetition pattern does not yet affect the generated pitches. "
        "The probability for the pitches I chose for you here is {probability:,.5g}."
    )

    second_stanza_secondary_intrastrophal_case_cd: str = (
        "Again, in the intrastophal case, the CD part has the most freedom, "
        "since the repetition pattern does not yet affect the generated pitches. "
        "The probability for the pitches I chose for you here is {probability:,.5g}."
    )

    second_stanza_pitch_intrastrophal_case_cb: str = (
        "In the first half of the second stanza (with the form CB), the C part has already been determined by "
        "our previous generation step. In addition, the first stanza ends on the three final pitches of B, "
        "so we sample it as the cadential phrase {cadential_phrase}, which has a probability of {cadential_probability:.1f}%. "
        "The other pitches I chose for you in B have the probability {pitch_probability:,.5g}."
    )

    second_stanza_secondary_intrastrophal_case_cb: str = (
        "We already have determined the C part, "
        "so similar to the pitch generation step, we "
        "sample a secondary cadential phrase {cadential_phrase} with a probability of {cadential_probability:.1f}%. "
        "The other symbols in B I have chosen for you has the total probability {pitch_probability:,.5g}."
    )

    second_stanza_pitch_intrastrophal_case_ab: str = (
        "The first stanza has the form ABAB, where we have already determined B in the previous step. "
        "The pitches I generated for A have a probability of {probability:,.5g}."
    )

    second_stanza_secondary_intrastrophal_case_ab: str = (
        "Because of the ABAB form in the first stanza, and we have already generated the B part, "
        "I generated you a suitable A part with probability {probability:,.5g}."
    )

    first_stanza_pitch_initial_state_with_repetition: str = (
        "Since there is no repetition inside the cadential phrase of the first stanza, "
        "I generated {cadential_phrase}, which has a probability of {probability:.1f}%."
    )

    first_stanza_secondary_initial_state_with_repetition: str = (
        "Because there is no repetition pattern inside the first stanza's cadential phrase, "
        "I generated the secondary pattern {cadential_phrase} with a probability of {probability:.1f}%."
    )

    first_stanza_pitch_initial_state_without_repetition: str = (
        "For the first stanza's cadential phrase, "
        "I generated {cadential_phrase}, which for the given repetition pattern has a probability of {probability:.1f}%."
    )

    first_stanza_secondary_initial_state_without_repetition: str = (
        "For the first stanza's cadential phrase, "
        "I generated {cadential_phrase}. For the given repetition pattern it has a probability of {probability:.1f}%."
    )

    first_stanza_pitch_normal_case: str = (
        "Due to the repetitions, the first stanza has restricted possibilities for the pitches that fill the gaps. "
        "I chose some pitches that have a probability of {probability:,.3g}."
    )

    first_stanza_secondary_normal_case: str = (
        "As determined by the repetition pattern, the first stanza has restricted possibilities for the secondary symbols filling the gaps. "
        "Therefore, I chose some secondary symbols that have a combined probability of {probability:,.3g}."
    )

    both_stanzas_pitch_no_repetition_case: str = (
        "Because there is no specific repetition pattern inside this piece, I simply generated all missing pitches for you"
        ", which has a probability of {probability:,.5g}"
        "."
    )

    both_stanzas_secondary_no_repetition_case: str = (
        "There is no specific repetition pattern inside this piece, so I simply generated all missing secondary symbols"
        "with a probability of {probability:,.5g}"
        "."
    )

    final_text: str = (
        "Wow, given the repetition pattern, your personal piece is generated with a chance of around 1 : 1{int_probability:,} ({total_probability:,.5g})!"
    )
