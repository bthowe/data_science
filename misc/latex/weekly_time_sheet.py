import os
import sys
import glob
import datetime
import subprocess


today = datetime.date.today()
dates = [str(today + datetime.timedelta(days)) for days in range(1, 7)]

discussion_questions = [
    "What is water made from? Would being able to make water change the world? (Seth)",
    "How do lakes and ponds get and keep their water? (Kay)",
    "How do countries get their names? (Calvin)",
    "What is the rule of law and why is or isn't it significant?",
    "What is illegal fishing, why is it a problem, and how can technology help?",
    "How would a \$2 ventilator affect medicine?"
]

books = ['Algebra 1 / 2', 'Math 7 / 6', 'Math 5 / 4']

jobs = [
    'Garbage',
    'Toilet',
    'Sink',
    'Mirror, light switch, door knob',
    'Sweep',
    'Mop'
]


header = r'''
\documentclass[10pt,twoside,letterpaper,oldfontcommands,openany]{memoir}
\usepackage{rotating}
\usepackage[margin=0.25in]{geometry}
\newcommand{\tabitem}{~~\llap{\textbullet}~~}


\pagenumbering{gobble}

\begin{document}
'''
# \usepackage{datenumber}
# \newif\iffirst
# \newcommand{\pnext}{%
#     \addtocounter{datenumber}{1}%
#     \setdatebynumber{\thedatenumber}%
#     \datedate
# }

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

\\clearpage
\\newpage
\\makeatletter
\\setlength{{\@fptop}}{{5pt}}
\\makeatother
\\begin{{table}}
\\begin{{tabular}}{{| l | l | l | l | l | l |}}
\\hline
\\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Name: Calvin}} & \\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Book: {12}}} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{Start Chapter: }} & \\multicolumn{{3}}{{|l|}}{{First Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{End Chapter: }} & \\multicolumn{{3}}{{|l|}}{{Last Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Date: {0}}} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Start Time: }} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{End Time: }} \\\\[20pt]
\\hline
\\end{{tabular}}
\\end{{table}}

\\clearpage
\\newpage
\\makeatletter
\\setlength{{\@fptop}}{{5pt}}
\\makeatother
\\begin{{table}}
\\begin{{tabular}}{{| l | l | l | l | l | l |}}
\\hline
\\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Name: Calvin}} & \\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Book: {12}}} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{Start Chapter: }} & \\multicolumn{{3}}{{|l|}}{{First Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{End Chapter: }} & \\multicolumn{{3}}{{|l|}}{{Last Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Date: {1}}} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Start Time: }} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{End Time: }} \\\\[20pt]
\\hline
\\end{{tabular}}
\\end{{table}}

\\clearpage
\\newpage
\\makeatletter
\\setlength{{\@fptop}}{{5pt}}
\\makeatother
\\begin{{table}}
\\begin{{tabular}}{{| l | l | l | l | l | l |}}
\\hline
\\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Name: Calvin}} & \\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Book: {12}}} \\\\[20pt]
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
\\setlength{{\@fptop}}{{5pt}}
\\makeatother
\\begin{{table}}
\\begin{{tabular}}{{| l | l | l | l | l | l |}}
\\hline
\\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Name: Calvin}} & \\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Book: {12}}} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{Start Chapter: }} & \\multicolumn{{3}}{{|l|}}{{First Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{End Chapter: }} & \\multicolumn{{3}}{{|l|}}{{Last Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Date: {3}}} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Start Time: }} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{End Time: }} \\\\[20pt]
\\hline
\\end{{tabular}}
\\end{{table}}

\\clearpage
\\newpage
\\makeatletter
\\setlength{{\@fptop}}{{5pt}}
\\makeatother
\\begin{{table}}
\\begin{{tabular}}{{| l | l | l | l | l | l |}}
\\hline
\\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Name: Calvin}} & \\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Book: {12}}} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{Start Chapter: }} & \\multicolumn{{3}}{{|l|}}{{First Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{End Chapter: }} & \\multicolumn{{3}}{{|l|}}{{Last Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Date: {4}}} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Start Time: }} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{End Time: }} \\\\[20pt]
\\hline
\\end{{tabular}}
\\end{{table}}

\\clearpage
\\newpage
\\makeatletter
\\setlength{{\@fptop}}{{5pt}}
\\makeatother
\\begin{{table}}
\\begin{{tabular}}{{| l | l | l | l | l | l |}}
\\hline
\\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Name: Calvin}} & \\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Book: {12}}} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{Start Chapter: }} & \\multicolumn{{3}}{{|l|}}{{First Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{End Chapter: }} & \\multicolumn{{3}}{{|l|}}{{Last Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Date: {5}}} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Start Time: }} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{End Time: }} \\\\[20pt]
\\hline
\\end{{tabular}}
\\end{{table}}


\\clearpage
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


\\clearpage
\\newpage
\\makeatletter
\\setlength{{\@fptop}}{{5pt}}
\\makeatother
\\begin{{table}}
\\begin{{tabular}}{{| l | l | l | l | l | l |}}
\\hline
\\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Name: Samuel}} & \\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Book: {13}}} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{Start Chapter: }} & \\multicolumn{{3}}{{|l|}}{{First Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{End Chapter: }} & \\multicolumn{{3}}{{|l|}}{{Last Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Date: {0}}} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Start Time: }} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{End Time: }} \\\\[20pt]
\\hline
\\end{{tabular}}
\\end{{table}}

\\clearpage
\\newpage
\\makeatletter
\\setlength{{\@fptop}}{{5pt}}
\\makeatother
\\begin{{table}}
\\begin{{tabular}}{{| l | l | l | l | l | l |}}
\\hline
\\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Name: Samuel}} & \\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Book: {13}}} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{Start Chapter: }} & \\multicolumn{{3}}{{|l|}}{{First Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{End Chapter: }} & \\multicolumn{{3}}{{|l|}}{{Last Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Date: {1}}} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Start Time: }} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{End Time: }} \\\\[20pt]
\\hline
\\end{{tabular}}
\\end{{table}}

\\clearpage
\\newpage
\\makeatletter
\\setlength{{\@fptop}}{{5pt}}
\\makeatother
\\begin{{table}}
\\begin{{tabular}}{{| l | l | l | l | l | l |}}
\\hline
\\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Name: Samuel}} & \\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Book: {13}}} \\\\[20pt]
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
\\setlength{{\@fptop}}{{5pt}}
\\makeatother
\\begin{{table}}
\\begin{{tabular}}{{| l | l | l | l | l | l |}}
\\hline
\\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Name: Samuel}} & \\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Book: {13}}} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{Start Chapter: }} & \\multicolumn{{3}}{{|l|}}{{First Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{End Chapter: }} & \\multicolumn{{3}}{{|l|}}{{Last Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Date: {3}}} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Start Time: }} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{End Time: }} \\\\[20pt]
\\hline
\\end{{tabular}}
\\end{{table}}

\\clearpage
\\newpage
\\makeatletter
\\setlength{{\@fptop}}{{5pt}}
\\makeatother
\\begin{{table}}
\\begin{{tabular}}{{| l | l | l | l | l | l |}}
\\hline
\\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Name: Samuel}} & \\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Book: {13}}} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{Start Chapter: }} & \\multicolumn{{3}}{{|l|}}{{First Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{End Chapter: }} & \\multicolumn{{3}}{{|l|}}{{Last Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Date: {4}}} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Start Time: }} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{End Time: }} \\\\[20pt]
\\hline
\\end{{tabular}}
\\end{{table}}

\\clearpage
\\newpage
\\makeatletter
\\setlength{{\@fptop}}{{5pt}}
\\makeatother
\\begin{{table}}
\\begin{{tabular}}{{| l | l | l | l | l | l |}}
\\hline
\\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Name: Samuel}} & \\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Book: {13}}} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{Start Chapter: }} & \\multicolumn{{3}}{{|l|}}{{First Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{End Chapter: }} & \\multicolumn{{3}}{{|l|}}{{Last Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Date: {5}}} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Start Time: }} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{End Time: }} \\\\[20pt]
\\hline
\\end{{tabular}}
\\end{{table}}


\\clearpage
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

\\clearpage
\\newpage
\\makeatletter
\\setlength{{\@fptop}}{{5pt}}
\\makeatother
\\begin{{table}}
\\begin{{tabular}}{{| l | l | l | l | l | l |}}
\\hline
\\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Name: Kay}} & \\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Book: {14}}} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{Start Chapter: }} & \\multicolumn{{3}}{{|l|}}{{First Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{End Chapter: }} & \\multicolumn{{3}}{{|l|}}{{Last Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Date: {0}}} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Start Time: }} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{End Time: }} \\\\[20pt]
\\hline
\\end{{tabular}}
\\end{{table}}

\\clearpage
\\newpage
\\makeatletter
\\setlength{{\@fptop}}{{5pt}}
\\makeatother
\\begin{{table}}
\\begin{{tabular}}{{| l | l | l | l | l | l |}}
\\hline
\\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Name: Kay}} & \\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Book: {14}}} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{Start Chapter: }} & \\multicolumn{{3}}{{|l|}}{{First Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{End Chapter: }} & \\multicolumn{{3}}{{|l|}}{{Last Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Date: {1}}} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Start Time: }} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{End Time: }} \\\\[20pt]
\\hline
\\end{{tabular}}
\\end{{table}}

\\clearpage
\\newpage
\\makeatletter
\\setlength{{\@fptop}}{{5pt}}
\\makeatother
\\begin{{table}}
\\begin{{tabular}}{{| l | l | l | l | l | l |}}
\\hline
\\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Name: Kay}} & \\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Book: {14}}} \\\\[20pt]
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
\\setlength{{\@fptop}}{{5pt}}
\\makeatother
\\begin{{table}}
\\begin{{tabular}}{{| l | l | l | l | l | l |}}
\\hline
\\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Name: Kay}} & \\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Book: {14}}} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{Start Chapter: }} & \\multicolumn{{3}}{{|l|}}{{First Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{End Chapter: }} & \\multicolumn{{3}}{{|l|}}{{Last Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Date: {3}}} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Start Time: }} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{End Time: }} \\\\[20pt]
\\hline
\\end{{tabular}}
\\end{{table}}

\\clearpage
\\newpage
\\makeatletter
\\setlength{{\@fptop}}{{5pt}}
\\makeatother
\\begin{{table}}
\\begin{{tabular}}{{| l | l | l | l | l | l |}}
\\hline
\\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Name: Kay}} & \\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Book: {14}}} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{Start Chapter: }} & \\multicolumn{{3}}{{|l|}}{{First Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{End Chapter: }} & \\multicolumn{{3}}{{|l|}}{{Last Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Date: {4}}} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Start Time: }} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{End Time: }} \\\\[20pt]
\\hline
\\end{{tabular}}
\\end{{table}}

\\clearpage
\\newpage
\\makeatletter
\\setlength{{\@fptop}}{{5pt}}
\\makeatother
\\begin{{table}}
\\begin{{tabular}}{{| l | l | l | l | l | l |}}
\\hline
\\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Name: Kay}} & \\multicolumn{{3}}{{|p{{9.5cm}}|}}{{Book: {14}}} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{Start Chapter: }} & \\multicolumn{{3}}{{|l|}}{{First Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{3}}{{|l|}}{{End Chapter: }} & \\multicolumn{{3}}{{|l|}}{{Last Problem: }} \\\\[20pt]
\\hline
\\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Date: {5}}} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{Start Time: }} & \\multicolumn{{2}}{{|p{{6.33cm}}|}}{{End Time: }} \\\\[20pt]
\\hline
\\end{{tabular}}
\\end{{table}}


\\clearpage
\\newpage
\\begin{{sidewaystable}}
\\LARGE
\\centering
\\begin{{tabular}}{{| l | l | l | l | l | l | l |}}
\\hline\\hline
 & Monday & Tuesday & Wednesday & Thursday & Friday & Saturday \\\\[10pt]
\\hline\\hline
Calvin & \\tabitem {15} & \\tabitem {16} & \\tabitem {17} & \\tabitem {18} & \\tabitem {19} & \\tabitem {20} \\\\
& \\tabitem Room & \\tabitem Room & \\tabitem Room & \\tabitem Room & \\tabitem Room & \\tabitem Room \\\\
& \\tabitem 4:30 & \\tabitem 4:30 & \\tabitem 4:30 & \\tabitem 4:30 & \\tabitem 4:30 & \\tabitem 4:30 \\\\
\\hline\\hline
Samuel & \\tabitem {15} & \\tabitem {16} & \\tabitem {17} & \\tabitem {18} & \\tabitem {19} & \\tabitem {20} \\\\
& \\tabitem Room & \\tabitem Room & \\tabitem Room & \\tabitem Room & \\tabitem Room & \\tabitem Room \\\\
& \\tabitem 4:30 & \\tabitem 4:30 & \\tabitem 4:30 & \\tabitem 4:30 & \\tabitem 4:30 & \\tabitem 4:30 \\\\
\\hline\\hline
Kay & \\tabitem {15} & \\tabitem {16} & \\tabitem {17} & \\tabitem {18} & \\tabitem {19} & \\tabitem {20} \\\\
& \\tabitem Room & \\tabitem Room & \\tabitem Room & \\tabitem Room & \\tabitem Room & \\tabitem Room \\\\
& \\tabitem 4:30 & \\tabitem 4:30 & \\tabitem 4:30 & \\tabitem 4:30 & \\tabitem 4:30 & \\tabitem 4:30 \\\\
\\hline\\hline
Seth & \\tabitem Stairwell & \\tabitem Stairwell & \\tabitem Stairwell & \\tabitem Stairwell & \\tabitem Stairwell & \\tabitem Stairwell \\\\
\\hline\\hline
\\end{{tabular}}
\\end{{sidewaystable}}

'''.format(
    dates[0], dates[1], dates[2], dates[3], dates[4], dates[5], discussion_questions[0], discussion_questions[1], discussion_questions[2], discussion_questions[3], discussion_questions[4], discussion_questions[5],  # 0-11
    books[0], books[1], books[2],  # 12-14
    jobs[0], jobs[1], jobs[2], jobs[3], jobs[4], jobs[5]  # 15-20
)


content = header + main + footer

with open('weekly_time_sheet.tex','w') as f:
     f.write(content)

commandLine = subprocess.Popen(['/Library/TeX/Root/bin/x86_64-darwin/pdflatex', 'weekly_time_sheet.tex'])
# commandLine = subprocess.Popen(['pdflatex', 'weekly_time_sheet.tex'])
commandLine.communicate()

os.unlink('weekly_time_sheet.aux')
os.unlink('weekly_time_sheet.log')
os.unlink('weekly_time_sheet.tex')


# todo: make functions for the boilerplate and the timesheet...lot of redundant code
