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
virtualenv --python=/usr/local/Cellar/python/3.7.5/Frameworks/Python.framework/Versions/3.7/bin/python3.7 test_ve --system-site-packages
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

remember to install pipreqs and create the requirements file in the current directory
```bash
pip install pipreqs
pipreqs .
```

to update a package
```bash
pip install pandas {[[[[--upgrade
```

list python libraries (active ve first)
```bash
pip freeze
pip freeze > requirements.txt
```

install library to certain installation of python
```bash
pip2.7 install <package name>
pip3.6 install <package name>
```bash
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
or 
12. cd python-package; python3 setup.py install
I was having a problem where the library was being recognized but the class XGBCLassifier (or whatever it's called) wasn't. After looking at the interpreter's path I hit the refresh button and this fixed the problem, I think, by putting the xgboost-0.81....egg file in the path.
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
 
 
 
 Update brew and install mongodb
 ```bash
 brew update
 brew install mongodb
 ```
 Create the data directory
 ```bash
 sudo mkdir -p /data/db
 ```
 Set permissions in directory (this much permission is bad, but it's local and for development)
 ```bash 
 sudo chmod -R 777 /data/db
 ```
 Run MongoDB
 ```bash
 mongod
 ```
 See other instances of mongod running
 ```bash
 ps ax | grep mongod
 ```
 Kill an instance
 ```bash
 kill -9 98555
 ```
 Kill all mongodb instances
 ```bash
 killall mongod
 ```
 
Shutdown from command line
```bash
mongod --shutdown
```

Open Mongo shell
```bash
mongo
```
 
Mongo commands
```bash
show dbs
show collections
use <database name>
use <collection name>
db.collection_name.find()   after "use <database name>", shows all documents in this collection (do not tell it to "use <collection name>" first, or else you won't get anything) 
db.bofm.find({'book': '2-ne', 'chapter': '20'})
db.bofm.deleteMany({'book': '2-ne', 'chapter': '21'})
db.bofm.stats().count
db['dc-testament'].stats().count

db.Math_7_6.update({_id: ObjectId("5ba1bc941036475a96128fc6")}, { 
: { date: "2018-09-17"}})

db.Algebra_1_2.deleteOne({ _id: ObjectId("5bd7c1c11036473053367ded")})

//drop collection within a database in use
db.Algebra_1_2_thingy.drop()

// within a directory where the directory named dump is located
mongodump -d vocab -o dump
mongorestore --db vocab --verbose dump/vocab

// update every document in a field: https://stackoverflow.com/questions/7714216/add-new-field-to-every-document-in-a-mongodb-collection
db.your_collection.update(
    {},
    {$set : {"new_field":1}},
    {upsert:false,
    multi:true}
  ) 

to delete a db
use database_name
db.dropDatabase()


// to create a new database. You need at least one document
use scripture_commentary
db.createCollection("Calvin")
db.Calvin.insert({"name": "Calvin", "date": "2018-11-05", "ref": "", "comment": "Why was there famine?"})
```



Make bash script executable...
on first line of the script
```bash
#!/bin/sh
```
I don't add the .sh suffix. In the command line then type
```bash
chmod 755 name_of_script
```
Now I can double click and it will execute.
Alternatively, simply give it a .command suffix (i.e., name_of_script.command)








CRON Jobs
```bash

```
* creating/editing contab file
```bash
env EDITOR=nano crontab -e
```
Control+0, enter, Control+X to write and then exit nano.

Every two minutes
```git
*/2 * * * * bash ~/Projects/github/howeschool_app/test.sh
```
Every day at midnight
```git
0 0 * * * bash ~/Projects/github/howeschool_app/test.sh
```






```bash
python -c "import os"
echo $?
```
if the result is 0 then it is installed

```bash
sudo apt-get install python3-numpy
```
```bash
sudo apt-get install --only-upgrade python3-numpy
```
```bash
pip3 install numpy --upgrade
```
Show location
```bash
pip3 show numpy
```

```bash
pip --no-cache-dir install pandas
```

was having problems getting pandas to load. It was saying numpy was not found.
This solved the problem...
```bash
sudo apt-get install libatlas-base-dev
```



I had to downgrade pymongo to 3.4.0
```bash
pip3 install pymongo===3.4.0
```






/home/pi/.local/bin/gunicorn --bind 0.0.0.0:8001 command:app



login as root
```bash
sudo su
```
exit root
```bash
exit
```

This is a nice tutorial regarding setting up supervisor
https://www.vultr.com/docs/installing-and-configuring-supervisor-on-ubuntu-16-04



Shows where mongo is located
```bash
which mongo
```

Prints mongod process
```bash
pgrep mongod
 ```
 
 



#### Getting profiles and arrangements set up in Iterm2
https://apple.stackexchange.com/questions/22445/how-can-i-save-tabs-in-iterm-2-so-they-restore-the-next-time-the-app-is-run
- Make sure to command + shift + S in order to save the profiles set.

#### Importing libraries in ipython on startup
https://towardsdatascience.com/how-to-automatically-import-your-favorite-libraries-into-ipython-or-a-jupyter-notebook-9c69d89aa343

#### Resolving matplotlib backend issue.
https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python









brew and installing mongodb
https://treehouse.github.io/installation-guides/mac/mongo-mac.html


###SSH into another computer
```bash
ssh pi@xxx.xxx.x.xx
```

###Copy directory from remote computer to local
```bash
scp -r pi@xxx.xxx.x.xx:/media/pi/HOWESCHOOL/database_files .
```

###Export mongo db
```bash
mongoexport --host localhost --db forms --collection Scriptures --csv --out forms_Scriptures.csv --fields scripture,week_start_date,scripture_ref
```
###Import mongo db
```bash
mongoimport -d mydb -c things --type csv --file locations.csv --headerline
```
Here, mydb is the database name, things is the collections, locations.csv is the file path.





###MySQL
####updating the path for mysql
```bash
export PATH=${PATH}:/usr/local/mysql/bin
```
Put that in .bash_profile

Login in shell
```bash 
mysql -u root -p
```

###In MySQL
Quit in shell
```bash 
quit
```

create database
```bash
CREATE DATABASE nets_2015;
```

```sql
CREATE TABLE emp (table column stuff);
```
```sql
DROP TABLE emp;
```
```sql
describe emp;
```
```sql
use NETS_2015;
```
```sql
show tables;
```
```sql
SET GLOBAL local_infile = true;
```sql
SHOW GLOBAL VARIABLES LIKE 'local_infile';
```

https://stackoverflow.com/questions/18437689/error-1148-the-used-command-is-not-allowed-with-this-mysql-version
so in /etc I created a file called my.cnf and added to it
```bash
[client]
loose-local-infile = 1
```
