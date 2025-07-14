import numpy as np

from filter_and_conversion import filter_by_mode, filter_by_final
from music import GongdiaoStep, Lvlv, GongdiaoModeList
from text_resources import EnglishTexts


def generate_mode(pieces):
    final_notes = [
        GongdiaoStep.GONG,
        GongdiaoStep.SHANG,
        GongdiaoStep.YU
    ]

    final_probs = [
        6/15,
        4/15,
        5/15
    ]

    gong_lvlvs = [
        Lvlv.HUANGZHONG,
        Lvlv.DALV,
        Lvlv.JIAZHONG,
        Lvlv.ZHONGLV,
        Lvlv.LINZHONG,
        Lvlv.YIZE,
        Lvlv.WUYI
    ]

    gong_lvlv_probs = [
        1/10,
        1/10,
        2/10,
        1/10,
        1/10,
        2/10,
        2/10,
    ]

    mgong = np.random.choice(gong_lvlvs, 1, p=gong_lvlv_probs)[0]
    mfinal = np.random.choice(final_notes, 1, p=final_probs)[0]

    total_probability = final_probs[final_notes.index(mfinal)] * gong_lvlv_probs[gong_lvlvs.index(mgong)]
    mode_properties = {"final_note": mfinal, "gong_lvlv": mgong}
    total_mode = GongdiaoModeList.from_properties(mode_properties)

    ## build description string

    description_string = EnglishTexts.mode_name.format(chinese_name=total_mode.chinese_name, name=total_mode.name, probability=total_probability*100)

    real_pieces_with_same_mode = filter_by_mode(pieces["all"], mode_properties)
    real_pieces_with_same_final = filter_by_final(pieces["all"], mode_properties["final_note"])
    if len(real_pieces_with_same_mode):
        piece_list_string = "".join(f"{piece['title']} ({piece['latinized_title']}), " for piece in real_pieces_with_same_mode)
        piece_list_string = piece_list_string[:-2]  # remove last comma
        description_string += EnglishTexts.mode_in_baishi.format(piece_list=piece_list_string)
    else:
        description_string += EnglishTexts.mode_not_in_baishi

    final_note_string = "".join(
        f"{piece['title']} ({piece['latinized_title']}), " for piece in real_pieces_with_same_final)
    final_note_string = final_note_string[:-2]  # remove last comma
    description_string += EnglishTexts.mode_final_note.format(final_note=mfinal, final_note_list=final_note_string)


    return {"mgong": str(mgong),
            "mfinal": str(mfinal),
            "description": description_string,
            "probability": total_probability}
