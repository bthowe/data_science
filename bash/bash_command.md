sudo
    -when you want to be boss

Ctrl-c
    -kills script

cd some/path
    -change directory

cp file new_file
    -copy files

mv file new/file/location
    -move files
    -also used to rename files

mkdir directory_name
    -creates a directory

rm file_name
    -removes a file

rmdir some_empty_directory
    -removes an empty directory called "some_empty_directory"

cat some_file.txt
    -prints the file to screen

less some_file some_other_file
    -space bar to page down, q to quit, :n to get to next document, :p to get the previous

head -n 3 file.txt
    -display the first 3 lines of file.txt

tail --last=10 file.txt
    -display the last 10 lines of file.txt

ls -RF
    -list everything underneath a directory
    -R means recursive
    -F means printing a / after the name of every directoy and * after the name of runnable programs

man <command>
    -prints the manual for <command>, e.g. man head

cut -f 2-5, 8 -d , values.csv
    -select columns 2 through 5 and column 8, using comma as the separator

history
    -gives a numbered command history
    -re-run a command by typing ! followed (without a space) by the number
    -can also run the most recent use of a certain command by typing ! followed (without a space) by the command
    -a useful command is: $ history | grep -h -v Tooth spring.csv > temp.csv, which takes all lines but the header (which has the word Tooth) and sends the output to temp.csv

grep word file.txt
    -return lines that have the word "word" in them
    -see the many options for this

paste -d , file1.csv file2.csv
    -puts the data from file1 and file2 together separated by a comma

redirect command (i.e., >)
    -e.g. head -n 5 seasonal/summer.csv > top.csv
    -redirects the output to a file called top.csv

pipe command (i.e., |)
    -e.g., cut -d , -f 2 seasonal/summer.csv | grep -v Tooth
    -e.g., cut -d , -f 2 seasonal/autumn.csv | grep -v Tooth | head -n 1

wc -l
    -counts how many lines (-l option)
    -counts how many words (-w option)
    -counts how many characters (-c option)

WILDCARDS
    -*
        -e.g., cut -d , -f 1 seasonal/*
        -would take the first column (delimited by a comma) of all files in the seasonal directory
        -e.g., head -n 3 seasonal/s*
    -?
        -matches a single character
    -[...]
        -matches any one of the characters inside the square brackets
        -e.g., 201[78].txt
    -{...}
        -matches any of the command-separated patterns inside the brackets
        -e.g., {*.txt, *.csv}

sort
    -e.g., cut -f 2 -d ,  seasonal/winter.csv | grep -v Tooth | sort
    -there are a number of options available here

uniq
    -e.g., cut -f 2 -d , seasonal/* | grep -v Tooth | sort | uniq
    -often used in connection with sort...for obvious reasons
    -the option -c causes the count to appear next to the line

echo something
    -print whatever follows to screen
    -e.g., in the above case, it prints "something"
    -e.g., echo $USER
        -print the value of the environment variable USER

create a shell variable
    -e.g., training=seasonal/summer.csv
    -print value
        -echo $training

loops
    -e.g., for suffix in gif jpg png; do echo $suffix; done
    -e.g., for filename in people/*; do echo $filename; done
    -e.g.,  files=seasonal/*.csv
            for f in $files; do echo $f; done
    -e.g., for file in seasonal/*.csv; do head -n 2 $file | tail -n 1; done
    -e.g., for f in seasonal/*.csv; do echo $f; head -n 2 $f | tail -n 1; done

if there are spaces in file names
    -use single quotes around names, e.g., 'July 2017.csv'
    -use backslash, e.g., July\ 2017.csv





to create a menu of commands to run in a shell script (you don't have to save with .sh suffix, but it's a nice convention):
-in a file called headers.sh: head -n 1 seasonal/*.csv
    -this can be done by typing: nano headers.sh
    -then inputing head -n 1 seasonal/*.csv
-run it by typing bash headers.sh
-the command $@ allows you to pass command-line parameters into the script
    -e.g., sort $@ | uniq, in a file called unique-lines.sh
    -e.g. continued, in the command line: $ bash unique-lines.sh seasonal/* > output_file.csv
-$1, $2, etc. allow you to pass specific command-line parameters
    -e.g., cut -d , -f $2 $1, in script called column.sh
    -e.g. continued, bash column.sh seasonal/autumn.csv 1
-can include loops and multiple lines
-can be used like normal shell commands (e.g., the results can be piped):
    -e.g., bash date-range.sh seasonal/* | sort








to create a virtual environment named "test_ve":
```bash
$ virtualenv test_ve
```

to create a virtual environment named "test_ve" which inherits globally installed packages:
```bash
$ virtualenv test_ve --system-site-packages
```

to create a virtual environment named "test_ve" which uses something other than the default interpretor:
```bash
$ virtualenv --python=/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/bin/python2.7 test_ve
$ virtualenv --python=/usr/local/Cellar/python3/3.6.3/Frameworks/Python.framework/Versions/3.6/bin/python3.6 test_ve

virtualenv --python=/usr/local/Cellar/python/2.7.13/Frameworks/Python.framework/Versions/2.7/bin/python2.7 retention --system-site-packages
virtualenv --python=/usr/local/Cellar/python3/3.6.3/Frameworks/Python.framework/Versions/3.6/bin/python3.6 test_ve --system-site-packages
```

to activate a virtual environment named "test_ve":
```bash
$ source test_ve/bin/activate
```

to install python packages to test_ve from a requirements.txt file:
```bash
$ pip install -r /path/to/requirements.txt
$ pip3 install -r /path/to/requirements.txt
```

to deactivate
```bash
$ deactivate
```


list python libraries
```bash
pip freeze
pip freeze > requirements.txt
```

install library to certain installation of python
```bash
pip2.7 install <package name>
pip3.6 install <package name>
```

alias pip and python installations in bash
...in .bash_profile
```bash
alias pip='pip2.7'
alias pip3='pip3.6'
alias python=python2.7
alias python3=python3.6
```

I had to follow the following in order to install xgboost: https://stackoverflow.com/questions/39315156/how-to-install-xgboost-in-python-on-macos
1. cd into the following directory.../venv/my_repo/lib/python3.6/site-packages
2. git clone --recursive https://github.com/dmlc/xgboost
3. cd xgboost
4. cp make/config.mk ./config.mk
5. vi config.mk
6. change # export CC = gcc to export CC = gcc-7 (i.e., uncomment the line and add "-7")
7. change # export CXX = gcc to export CXX = g++-7
8. vi Makefile
9. change export CC = gcc to export CC = gcc-7
10. change export CXX = gcc to export CXX = g++-7
11. make clean_all && make -j4
12. cd python-package; python setup.py install


```bash
curl -i http://localhost:5000/todo/api/v1.0/tasks/2
curl --form addressFile=@/Users/travis.howe/Downloads/test.csv --form benchmark=9 --form vintage=Census2010_Census2010 https://geocoding.geo.census.gov/geocoder/geographies/addressbatch --output geocoderesult.csv
```
For testing endpoints in the command line

```bash
curl -i -H "Content-Type: application/json" -X POST -d '{"title":"Read a book"}' http://localhost:5000/todo/api/v1.0/tasks
```

Copying contents of a directory versus the entire directory 
```bash
cp -r ../I_and_R_treatment\ effect\ evaluation .
```
versus
```bash
cp -r ../I_and_R_treatment\ effect\ evaluation/ .
```
The former copies the directory and everything in it. The latter only copies the files of the directory. So if you want to move a directory into another, creating a subdirectory, use the former. 