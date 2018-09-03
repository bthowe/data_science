import os
import sys
import glob
import datetime
import subprocess


today = datetime.date.today()
dates = [str(today + datetime.timedelta(days)) for days in range(1, 7)]

discussion_questions = [
    "Alma 1:24. Why do people leave the Church?",
    "What is an investment? Should your occupation influence your optimal ivestment decisions?",
    "Why would sellers put other ingredients in bottles of honey or olive oil? What other ingredients?",
    "Why is it difficult to learn how to do things like rollerskating and skate-boarding? (Calvin)",
    "Why do cars use gas? (Kay)",
    "How have cars changed how people live their lives? (Samuel)"
]

header = r'''
\documentclass[10pt,twoside,letterpaper,oldfontcommands,openany]{memoir}
\usepackage{rotating}
\usepackage[margin=0.25in]{geometry}

\usepackage{datenumber}
\newif\iffirst
\newcommand{\pnext}{%
    \addtocounter{datenumber}{1}%
    \setdatebynumber{\thedatenumber}%
    \datedate
}

\pagenumbering{gobble}

\begin{document}
'''

footer = r'''\end{document}'''

main = '''
\\begin{{sidewaystable}}
\\centering
\\begin{{tabular}}{{|l|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|}}
\\multicolumn{{7}}{{l}}{{Name: Calvin}} \\\\
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
\\multicolumn{{2}}{{|p{{3cm}}|}}{{Alma 1:24. Why do people leave the Church?}} & 
\\multicolumn{{2}}{{p{{3cm}}|}}{{What is an investment? Should your occupation influence your optimal ivestment decisions?}} & 
\\multicolumn{{2}}{{p{{3cm}}|}}{{Why would sellers put other ingredients in bottles of honey or olive oil? What other ingredients?}} & 
\\multicolumn{{2}}{{p{{3cm}}|}}{{Why is it difficult to learn how to do things like rollerskating and skate-boarding? (Calvin)}} & 
\\multicolumn{{2}}{{p{{3cm}}|}}{{Why do cars use gas? (Kay)}} & 
\\multicolumn{{2}}{{p{{3cm}}|}}{{How have cars changed how people live their lives? (Samuel)}} 
\\\\[70pt]
\\hline
\\end{{tabular}}
\\end{{sidewaystable}}

\\newpage
\\begin{{sidewaystable}}
\\centering
\\begin{{tabular}}{{|l|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|}}
\\multicolumn{{7}}{{l}}{{Name: Samuel}} \\\\
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

\\newpage
\\begin{{sidewaystable}}
\\centering
\\begin{{tabular}}{{|l|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|p{{1.5cm}}|}}
\\multicolumn{{7}}{{l}}{{Name: Kay}} \\\\
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
'''.format(dates[0], dates[1], dates[2], dates[3], dates[4], dates[5], discussion_questions[0], discussion_questions[1], discussion_questions[2], discussion_questions[3], discussion_questions[4], discussion_questions[5])

content = header + main + footer

with open('weekly_time_sheet.tex','w') as f:
     f.write(content)

commandLine = subprocess.Popen(['pdflatex', 'weekly_time_sheet.tex'])
commandLine.communicate()

os.unlink('weekly_time_sheet.aux')
os.unlink('weekly_time_sheet.log')
os.unlink('weekly_time_sheet.tex')
