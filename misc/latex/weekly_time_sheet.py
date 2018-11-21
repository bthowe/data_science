import os
import sys
import glob
import datetime
import itertools
import subprocess


today = datetime.date.today()
date_shift = 7 - today.weekday()  # Monday is 0
date_shift = 0
dates = [str(today + datetime.timedelta(date_shift + days)) for days in range(0, 6)]

discussion_questions = [
    "Brainstorming research questions.",
    "Moving from a question to a testable hypothesis.",
    "Developing a test of hypothesis.",
    "Making inference and drawing conclusions.",
    "Question based on current event.",
    "Principles from scripture passage or general conference talk."
]

kids = ['Calvin', 'Samuel', 'Kay']

books = ['Algebra 1 / 2', 'Math 7 / 6', 'Math 5 / 4']

jobs = [
    'Garbage, toothbrush paper towel',
    'Toilet',
    'Sink',
    'Mirror, light switch, door knob',
    'Sweep',
    'Scrub shower, no cleanser'
]


header = r'''
\documentclass[10pt,twoside,letterpaper,oldfontcommands,openany]{memoir}
\usepackage{rotating, caption}
\usepackage[margin=0.25in]{geometry}
\newcommand{\tabitem}{~~\llap{\textbullet}~~}
\pagenumbering{gobble}
\begin{document}
'''

footer = r'''\end{document}'''

math_scripture = ''''''
for i in itertools.product(zip(kids, books), dates):
    math_scripture += '''
    \\clearpage
    \\newpage
    \\makeatletter
    \\setlength{{\@fptop}}{{35pt}}
    \\makeatother
    \\begin{{table}}
    \\caption*{{Math Assignment}}
    \\begin{{tabular}}{{| l | l | l | l | l | l |}}
    \\hline
    \\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Name: {0}}} & \\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Book: {1}}} \\\\[20pt]
    \\hline
    \\multicolumn{{3}}{{|l|}}{{Start Chapter: }} & \\multicolumn{{3}}{{|l|}}{{First Problem: }} \\\\[20pt]
    \\hline
    \\multicolumn{{3}}{{|l|}}{{End Chapter: }} & \\multicolumn{{3}}{{|l|}}{{Last Problem: }} \\\\[20pt]
    \\hline
    \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Date: {2}}} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Start Time: }} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{End Time: }} \\\\[20pt]
    \\hline
    \\end{{tabular}}
    \\end{{table}}

    \\clearpage
    \\newpage
    \\makeatletter
    \\setlength{{\@fptop}}{{35pt}}
    \\makeatother
    \\begin{{table}}
    \\caption*{{Scripture Questions and Principles}}
    \\begin{{tabular}}{{| l | l | l | l | l | l |}}
    \\hline
    \\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Name: {0}}} & \\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Date: {2}}} \\\\[20pt]
    \\hline
    \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Start Book: }} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Start Chapter: }} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Start Verse: }} \\\\[20pt]
    \\hline
    \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{End Book: }} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{End Chapter: }} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{End Verse: }} \\\\[20pt]
    \\hline
    \\multicolumn{{6}}{{l}}{{}} \\\\[20pt]
    \\multicolumn{{6}}{{l}}{{Comment:}} \\\\[20pt]
    \\end{{tabular}}
    \\end{{table}}
    '''.format(
        i[0][0], i[0][1], i[1]
    )


time_sheets = ''''''
for name in kids:
    time_sheets += '''
    \\begin{{sidewaystable}}
    \\centering
    \\begin{{tabular}}{{|l|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|}}
    \\multicolumn{{7}}{{l}}{{Name: {12}}} \\\\
    \\multicolumn{{7}}{{l}}{{}} \\\\
    \\multicolumn{{7}}{{l}}{{}} \\\\
    \\cline{{2-13}}
    \\multicolumn{{1}}{{l}}{{}} & \\multicolumn{{2}}{{|c|}}{{Monday}} & \\multicolumn{{2}}{{c|}}{{Tuesday}} & \\multicolumn{{2}}{{c|}}{{Wednesday}} & \\multicolumn{{2}}{{c|}}{{Thursday}} & \\multicolumn{{2}}{{c|}}{{Friday}} & \\multicolumn{{2}}{{c|}}{{Saturday}} \\\\
    \\multicolumn{{1}}{{l}}{{}} & \\multicolumn{{2}}{{|c|}}{{{0}}} & \\multicolumn{{2}}{{c|}}{{{1}}} & \\multicolumn{{2}}{{c|}}{{{2}}} & \\multicolumn{{2}}{{c|}}{{{3}}} & \\multicolumn{{2}}{{c|}}{{{4}}} & \\multicolumn{{2}}{{c|}}{{{5}}} \\\\
    \\cline{{2-13}}
    \\cline{{2-13}}
    \\multicolumn{{1}}{{l|}}{{}} & Start & Stop & Start & Stop & Start & Stop & Start & Stop & Start & Stop & Start & Stop \\\\
    \\hline
    \\hline
    Math & & & & & & & & & & & &\\\\[70pt]
    \\hline
    Reading & & & & & & & & & & & &\\\\[70pt]
    \\hline
    Writing & & & & & & & & & & & &\\\\[70pt]
    \\hline
    Vocabulary & & & & & & & & & & & &\\\\[70pt]
    \\hline
    Discussion &
    \\multicolumn{{2}}{{|p{{3cm}}|}}{{{6}}} &
    \\multicolumn{{2}}{{p{{3cm}}|}}{{{7}}} &
    \\multicolumn{{2}}{{p{{3cm}}|}}{{{8}}} &
    \\multicolumn{{2}}{{p{{3cm}}|}}{{{9}}} &
    \\multicolumn{{2}}{{p{{3cm}}|}}{{{10}}} &
    \\multicolumn{{2}}{{p{{3cm}}|}}{{{11}}}
    \\\\[70pt]
    \\hline
    \\end{{tabular}}
    \\end{{sidewaystable}}
    '''.format(
        dates[0], dates[1], dates[2], dates[3], dates[4], dates[5],  # 0-5
        discussion_questions[0], discussion_questions[1], discussion_questions[2], discussion_questions[3], discussion_questions[4], discussion_questions[5],  # 6-11
        name  # 12
    )

jobs = '''
\\clearpage
\\newpage
\\makeatletter
\\setlength{{\@fptop}}{{5pt}}
\\makeatother
\\begin{{sidewaystable}}
\\small
\\centering
\\begin{{tabular}}{{| l | l | l | l | l | l | l |}}
\\hline\\hline
 & Monday & Tuesday & Wednesday & Thursday & Friday & Saturday \\\\[10pt]
\\hline\\hline
Calvin & \\tabitem {0} & \\tabitem {1} & \\tabitem {2} & \\tabitem {3} & \\tabitem {4} & \\tabitem {5} \\\\
& \\tabitem {6} & \\tabitem {6} & \\tabitem {6} & \\tabitem {6} & \\tabitem {6} & \\tabitem {6} \\\\
& \\tabitem 4:30 & \\tabitem 4:30 & \\tabitem 4:30 & \\tabitem 4:30 & \\tabitem 4:30 & \\tabitem 4:30 \\\\
\\hline\\hline
Samuel & \\tabitem {0} & \\tabitem {1} & \\tabitem {2} & \\tabitem {3} & \\tabitem {4} & \\tabitem {5} \\\\
& \\tabitem {6} & \\tabitem {6} & \\tabitem {6} & \\tabitem {6} & \\tabitem {6} & \\tabitem {6} \\\\
& \\tabitem 4:30 & \\tabitem 4:30 & \\tabitem 4:30 & \\tabitem 4:30 & \\tabitem 4:30 & \\tabitem 4:30 \\\\
\\hline\\hline
Kay & \\tabitem {0} & \\tabitem {1} & \\tabitem {2} & \\tabitem {3} & \\tabitem {4} & \\tabitem {5} \\\\
& \\tabitem {6} & \\tabitem {6} & \\tabitem {6} & \\tabitem {6} & \\tabitem {6} & \\tabitem {6} \\\\
& \\tabitem 4:30 & \\tabitem 4:30 & \\tabitem 4:30 & \\tabitem 4:30 & \\tabitem 4:30 & \\tabitem 4:30 \\\\
\\hline\\hline
Seth & \\tabitem School & \\tabitem School & \\tabitem School & \\tabitem School & \\tabitem School & \\tabitem School \\\\
 & \\tabitem {6} & \\tabitem {6} & \\tabitem {6} & \\tabitem {6} & \\tabitem {6} & \\tabitem {6} \\\\
 & \\tabitem Stairwell & \\tabitem Stairwell & \\tabitem Stairwell & \\tabitem Stairwell & \\tabitem Stairwell & \\tabitem Stairwell \\\\
\\hline\\hline
\\end{{tabular}}
\\end{{sidewaystable}}
'''.format(jobs[0], jobs[1], jobs[2], jobs[3], jobs[4], jobs[5], '5 minute pickup')


content = header + jobs + time_sheets + math_scripture + footer

with open('weekly_time_sheet.tex', 'w') as f:
     f.write(content)

commandLine = subprocess.Popen(['/Library/TeX/Root/bin/x86_64-darwin/pdflatex', 'weekly_time_sheet.tex'])
# commandLine = subprocess.Popen(['pdflatex', 'weekly_time_sheet.tex'])
commandLine.communicate()

os.unlink('weekly_time_sheet.aux')
os.unlink('weekly_time_sheet.log')
os.unlink('weekly_time_sheet.tex')
