# SuziGEN
The style-based composition of melodies to *ci* 词 poetry in *suzipu* notation.


## Setting Up the Environment
To prepare the Python environment, use following configuration (tested with Python 3.12.3):
`pip install -r requirements.txt`

## Usage
```
python3 main.py [-h] --kuiscima-repo-dir KUISCIMA_REPO_DIR --cipai CIPAI [--lyrics LYRICS] [--mode MODE]
                       --output-file-name OUTPUT_FILE_NAME

Stylistic Composition in Jiang Kui's Ci Style

options:
  -h, --help            show this help message and exit
  --kuiscima-repo-dir KUISCIMA_REPO_DIR
                        Path to the folder containing the cloned KuiSCIMA repository
                        (https://github.com/SuziAI/KuiSCIMA)
  --cipai CIPAI         The cipai string of the the cipai that should be generated. The string must be enclosed
                        by quotation marks. 'p' indicates ping, 'z' indicates ze. 'P' indicates rhyme position
                        with ping, 'Z' indicates rhyme position with ze. '.', ',', ':', ';', '!', '?', '。', '，',
                        '！', '？', '：', '；' correspond to ju pause. '、' corresponds to dou pause. '/' indicates
                        the stanzaic division. Example: "zzzpp，zzzppZ。zppZ。/zzppZ，zzpppZ。ppzZ。"
  --lyrics LYRICS       The lyrics accompanying the cipai. The string must be enclosed by quotation marks.(For
                        display purpose in the MIDI file only.)
  --mode MODE           The name of the mode as indicated in class music.GongdiaoModeList. Capitalization,
                        whitespaces, and the corresponding simplified and traditional Chinese characters do not
                        affect the result. If not supplied, the mode will be sampled randomly. Pinyin 'ü' can
                        also be supplied as 'v'.
  --output-file-name OUTPUT_FILE_NAME
                        Path to the output file name. The software generates two files, one .txt (containing the
                        verbal explanation of the generation process) and a .mid containing a MIDI rendition of
                        the generated piece.

```

Exemplary usage:
```
python3 main.py --kuiscima-repo-dir ../KuiSCIMA/ --output-file-name output --lyrics "金谷人歸，綠楊低掃吹笙道。數聲啼鳥，也學相思調。月落潮生，掇送劉郎老。淮南好，甚 時重到？陌上生春草。" --cipai "zzpp，zppzppZ？zppZ，zzppZ。/pzpp，zzppZ。ppZ，zppZ，zzppZ！" --mode xianlvgong
```