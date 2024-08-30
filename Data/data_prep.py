import os
import gruut
import phonemizer
import csv 
from phonemizer.backend import EspeakBackend


text = 'He wound it around the wound, saying "I read it was $10 to read."'
ipa_set = set()
backend = EspeakBackend('en-us', with_stress=True, preserve_punctuation=True)


def gruut_phonemes(text):
    phonetic = ""
    for sent in gruut.sentences(text, lang="en-us"):
        for word in sent:
            if word.phonemes:
                # print(word.text, *word.phonemes)
                
                phonetic += "".join(word.phonemes) + " "
                ipa_set.update(word.phonemes)
    phonetic = phonetic.replace("d͡ʒ", "dʒ")
    phonetic = phonetic.replace("t͡ʃ", "tʃ")
    phonetic = phonetic.replace("t͡ʃ", "tʃ")

    return(phonetic.strip())

def espeak_phonemes(text):
    phonetic = backend.phonemize([text])
    return(phonetic[0])

# LibriTTS/train-clean-460/3072/155948/3072_155948_000007_000011.wav|ʌv kˈoːɹs bˈɑːksɪŋ ʃˌʊd biː ɛŋkˈɜːɹɪdʒd ɪnðɪ ˈɑːɹmi ænd nˈeɪvi.
# /export/eingerman/audio/LibriTTSR

outdir = "/export/eingerman/audio/LibriTTSR/"
indir = "/export/eingerman/audio/"


def process_files(prefix, bad_set, soxfile):
    # read input csv file line by line
    fout_gruut = open(f"{prefix}_list_LibriTTSR_gruut.txt", "w")
    # writer_gruut = csv.writer(fout_gruut, delimiter='|')
    fout_espeak = open(f"{prefix}_list_LibriTTSR_espeak.txt", "w")
    # writer_espeak = csv.writer(fout_espeak, delimiter='|')
                
    with open(f"{prefix}_list_libritts.txt", 'r') as file:
        reader = csv.reader(file, delimiter='¦')
        for nrow, row in enumerate(reader):
            try:
                wav_file_name = row[0]
                speaker_id=wav_file_name.split('/')[2]
                # replace LibriTTS with LibriTTSR in wav_file_name
                wav_file_name = wav_file_name.replace("LibriTTS", "LibriTTSR")
                just_file_name = os.path.split(wav_file_name)[-1]
                just_number = just_file_name.split('.')[0]
                
                if just_number in bad_set:
                    print(f"skipping {wav_file_name} as it is in bad set")
                    continue
                
                normalized_file_name = indir+wav_file_name.replace(".wav",".normalized.txt")
                with open(normalized_file_name, 'r') as f1:
                    text = f1.read().replace('\n', '').strip()
                
                gruut_phonetic = gruut_phonemes(text)
                espeak_phonetic = espeak_phonemes(text)
                print(f"{nrow}\t{wav_file_name}:{gruut_phonetic}:{espeak_phonetic}:{speaker_id}\n")
                fout_gruut.write(f"{wav_file_name}¦{gruut_phonetic}¦{speaker_id}\n")
                fout_espeak.write(f"{wav_file_name}¦{espeak_phonetic}¦{speaker_id}\n")
            
                soxfile.write(f"sox {wav_file_name} -r 16000 -b 16 -c 1 {wav_file_name.replace('train-clean-460','train-clean-460_16K')}\n")
                fout_gruut.flush()
                fout_espeak.flush()
                soxfile.flush()
            except Exception as e:
                print(f"Error {e} processing {wav_file_name}")
                continue
            
    fout_gruut.close()
    fout_espeak.close()        


def process_bad_set():
    bad_set = []
    with open(outdir+"bad_examples.txt",'r') as f:
        for ex in f:
            just_file_name = os.path.split(ex)[-1]
            just_number = just_file_name.split('.')[0]
            bad_set.append(just_number)
    return set(bad_set)

if __name__ == "__main__":
    soxfile = open("sox_commands.txt", "w")

    bad_set=process_bad_set()
    print("Bad set:", len(bad_set))

    process_files("val", bad_set, soxfile)
    process_files("train", bad_set, soxfile)

    soxfile.close()
    print(ipa_set)