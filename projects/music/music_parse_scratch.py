import sys
import music21
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 30000)
pd.set_option('max_colwidth', 4000)
pd.set_option('display.float_format', lambda x: '%.3f' % x)



def main():
    s = music21.converter.parse('/Users/travis/Downloads/Humoresque_No_7__Dvok__Violin_Cello_Piano_Trio.mxl')
    # s = music21.converter.parse('/Users/travis/Downloads/O Holy Night Maggie - Full score - 01 O Holy Night.musicxml')

    # Access elements of the score
    print(s.show('text'))
    print(s[5])
    # print(s[5][2][3].pitch)
    # s.show()  # Graphical representation (if configured)



if __name__ == '__main__':
    main()