class EnglishTexts:
    repetitions_none: str = (
        "As a repetition structure, I selected the absence of specific repetitions for you. "
        "This happens in {prob:.1f}% of all cases. In Baishidaoren Gequ, "
        "this is the case for the pieces 醉吟商小品 (Zuiyinshangxiaopin) and 淡黃柳 (Danhuangliu)."
    )
    repetitions_intrastrophal: str = (
        "The cipai you provided has a special structure! "
        "The first stanza consists of two parts with equal metrical structure. "
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
