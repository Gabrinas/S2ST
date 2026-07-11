from tinytag import TinyTag
import os

# Rename audios, rename the "new_file" varible to the expected file name,
# also note to change the "ENG" to "YOR" and vice-versa

def read_all(fpath):
    audio_files = os.listdir(fpath); ls = []; lsa = []
    q = 0; lt = len(audio_files); lsc = []; id = [0]
    while q < lt:
        vq = audio_files[q]
        g_n = str(id[-1]).zfill(3)
        return g_n
        file_path = os.path.join(fpath, vq)
        new_file = "DEV_AN_OPEN_SOURCE_FEMALE_ENG_" + str(g_n) + ".wav"
        new_filepath = os.path.join(fpath, new_file)
        os.rename(file_path, new_filepath)
        id.append(int(g_n))
        q += 1
    #ls = list(range(600, 800))
    #rs = [i for i in ls if not i in id]
    #print(rs)
    return 
        
    
    

#folder_path = r"C:\Users\LENOVO\Downloads\SPEAKER_ASR_ENG_5\AUDIOS"
folder_path = r"\\wsl.localhost\Ubuntu\home\gabriel\s2st_project\ENGLISH"
#folder_path = r"https://drive.google.com/drive/folders/1b7WS7JCPoQAh4PMuQpkGPCQ0UOVE5ZnN?usp=drive_link"
print(read_all(folder_path))
    
