import gruut

def gruut_phonemes(text):
    phonetic = ""
    for sent in gruut.sentences(text, lang="en-us"):
        for word in sent:
            if word.phonemes:
                # print(word.text, *word.phonemes)
                phonetic += "".join(word.phonemes) + " " 
    return(phonetic)

text ="""It is now expedient to give some description of Mrs. Allen, that the reader may be able to judge in what manner her actions will hereafter tend to promote the general distress of the work, and how she will, probably, contribute to reduce poor Catherine to all the desperate wretchedness of which a last volume is capable--whether by her imprudence, vulgarity, or jealousy--whether by intercepting her letters, ruining her character, or turning her out of doors.   It is now expedient to give some description of mrs Allen, that the reader may be able to judge in what manner her actions will hereafter tend to promote the general distress of the work, and how she will, probably, contribute to reduce poor Catherine to all the desperate wretchedness of which a last volume is capable-whether by her imprudence, vulgarity, or jealousy-whether by intercepting her letters, ruining her character, or turning her out of doors.""".split('.')
for t in text:
    ph=gruut_phonemes(t) 
    print(ph)              
                