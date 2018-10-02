source ~/PycharmProjects/venv/data_science/bin/activate

cd ~/Projects/github/data_science/flask_apps/saxon_input

mongod --fork --logpath ../../bash/vocab/mongo_log/mongod.log
open -a "Google Chrome" http://0.0.0.0:8001/
python3 saxon_math_command.py

mongodump -d math_book_info -o dump
mongodump -d math_exercise_origins -o dump
mongodump -d math_performance -o dump

git add dump/.
git commit -m "standard mongo dump"
git push origin master

killall mongod

deactivate